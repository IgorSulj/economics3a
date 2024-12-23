from itertools import count
from pathlib import Path
import pickle
from flax import nnx
import jax
import jax.experimental
import jax.numpy as jnp
import optax
import tqdm
import polars as pl
from huggingface_hub import HfApi
from data_loading import generate_batches, get_features_count, preprocess_history
import transformer
import cnn
import consts

class Model(nnx.Module):
    def __init__(self, *, rngs: nnx.Rngs, dropout: float = 0.1):
        self.lag = consts.LAG
        self.prediction_period = consts.PREDICTION_PERIOD
        self.n_features = 32
        native_features = get_features_count()
        self.cnn = cnn.OneBlock(native_features, 4, self.n_features - native_features, rngs=rngs)
        self.cnn_norm = nnx.LayerNorm(self.lag, rngs=rngs)
        self.transformer = get_itransformer(dropout, rngs)

    def __call__(self, x: jax.Array):
        prices = x[:, :4]
        prices_mean = prices.mean()
        prices_std = prices.std()
        volumes = x[:, -1:]
        x = jnp.hstack([
            (x[:, :-1] - prices_mean) / prices_std,
            (volumes - volumes.mean()) / volumes.std()
        ])
        cnn_features = self.cnn_norm(self.cnn(x).T).T
        x = jnp.hstack([x, cnn_features])
        return self.transformer(x) * prices_std + prices_mean
    
    def forecast(self, x: jax.Array):
        logs = self(x)
        return jnp.exp(logs)

    def dump_to_hf(self, api: HfApi, repo_id: str, name: str):
        model_state = nnx.state(self)
        pickle_bytes = pickle.dumps(model_state)
        api.upload_file(
            repo_id=repo_id,
            path_or_fileobj=pickle_bytes,
            path_in_repo=f"{name}.bin"
        )

    @staticmethod
    def load_from_hf(api: HfApi, repo_id: str, name: str) -> 'Model':
        state = load_state_from_hf(api, repo_id, name)
        return Model.load_from_state(state)

    @staticmethod
    def load_from_state(state: nnx.State) -> 'Model':
        graph = nnx.graphdef(nnx.eval_shape(lambda: Model(rngs=nnx.Rngs(0))))
        return nnx.merge(graph, state)


def load_state_from_hf(api: HfApi, repo_id: str, name: str) -> nnx.State:
    pickle_file = api.hf_hub_download(
        filename=f"{name}.bin",
        repo_id=repo_id,
    )
    pickle_bytes = Path(pickle_file).read_bytes()
    return pickle.loads(pickle_bytes)


def get_itransformer(dropout: float, rngs: nnx.Rngs):
    return transformer.iTransformer(
        consts.LAG,
        consts.PREDICTION_PERIOD,
        consts.N_LAYERS,
        32,
        consts.D_MODEL,
        consts.D_HIDDEN,
        dropout,
        rngs=rngs
    )


@nnx.jit
def loss(model: Model, x: jax.Array, y: jax.Array):
    return optax.l2_loss(model(x), y)


@nnx.jit
def optimize_on(model: Model, opt: nnx.Optimizer, candles: jax.Array, batched_start: jax.Array):
    def loss_fn(model: Model, start: jax.Array):
        x = jax.lax.dynamic_slice_in_dim(candles, start, consts.LAG)
        y = jax.lax.dynamic_slice_in_dim(candles[:, 1], start + consts.LAG, consts.PREDICTION_PERIOD)
        return loss(model, x, y)

    def batch_loss(model: Model, batched_start: jax.Array):
        batch_losses = jax.vmap(loss_fn, in_axes=(None, 0))(model, batched_start)
        return jnp.mean(batch_losses)

    @nnx.scan
    def scan_batches(state: tuple[Model, nnx.Optimizer], batched_start: jax.Array):
        model, opt = state
        loss_, grads = nnx.value_and_grad(batch_loss)(model, batched_start)
        opt.update(grads)
        return (model, opt), loss_

    (model, opt), losses = scan_batches((model, opt), batched_start)
    return jnp.mean(losses)


def optimize_on_ticker(model: Model, opt: nnx.Optimizer, data: pl.DataFrame, key: jax.Array, data_mesh: jax.sharding.Sharding, batch_mesh: jax.sharding.Sharding, batch_size: int = 16):
    candles, batches = generate_batches(data, key, batch_size, data_mesh, batch_mesh)
    if candles is None or batches is None:
        return None
    return optimize_on(model, opt, candles, batches)


def optimize_on_dataset(model: Model, opt: nnx.Optimizer, dataset: pl.DataFrame, key: jax.Array, data_mesh: jax.sharding.Sharding, batch_mesh: jax.sharding.Sharding, batch_size: int = 16):
    avg_loss = nnx.metrics.Average('loss')
    max_loss = -jnp.inf
    max_loss_ticker = None
    model.train()
    with tqdm.tqdm(total=len(dataset)) as iterator:
        curr_state = nnx.state(model)
        for (ticker,), data in dataset.group_by('symbol'):
            key, new_key = jax.random.split(key)
            train_loss = optimize_on_ticker(model, opt, data, key, batch_size=batch_size, data_mesh=data_mesh, batch_mesh=batch_mesh)
            iterator.update(len(data))
            if train_loss is None or not jnp.isfinite(train_loss):
                nnx.update(model, curr_state)
                continue
            curr_state = nnx.state(model)
            avg_loss.update(loss=train_loss)
            if train_loss > max_loss:
                max_loss = train_loss
                max_loss_ticker = ticker
            key = new_key
            iterator.set_postfix(ticker=ticker, train_loss=avg_loss.compute(), max_loss=max_loss, max_loss_ticker=max_loss_ticker)
    return key


@nnx.jit
def validation_loss(model: Model, candles: jax.Array):
    @nnx.vmap(in_axes=(None, 0), out_axes=0)
    def loss_fn(model: Model, start: jax.Array):
        x = jax.lax.dynamic_slice_in_dim(candles, start, consts.LAG)
        y = jax.lax.dynamic_slice_in_dim(candles[:, 1], start + consts.LAG, consts.PREDICTION_PERIOD)
        return loss(model, x, y)
    return loss_fn(model, jnp.arange(candles.shape[0] - consts.LAG - consts.PREDICTION_PERIOD)).mean()


def validation(model: Model, dataset: pl.DataFrame, key: jax.Array, data_mesh: jax.sharding.Sharding):
    metrics = nnx.metrics.Average('loss')
    for _, data in dataset.group_by('symbol'):
        if len(data) < consts.LAG + consts.PREDICTION_PERIOD:
            continue
        candles = preprocess_history(data.select('open', 'close', 'high', 'low', 'volume')).to_jax().to_device(data_mesh)
        key, new_key = jax.random.split(key)
        loss = validation_loss(model, candles)
        metrics.update(loss=loss)
        key = new_key
    return metrics.compute()
