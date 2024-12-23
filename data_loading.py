import json
from pathlib import Path
import requests
import yfinance as yf
import polars as pl
import jax
import jax.numpy as jnp
from consts import LAG, PREDICTION_PERIOD

NASDAQ_API = 'https://api.nasdaq.com/api/screener/stocks?tableonly=true&download=true'
TEST_TICKERS = ('GOOG', 'TSLA', 'SHOP')


def load_tickers():
    cache_path = Path("nasdaq_data.json")
    if cache_path.exists():
        rows = json.loads(cache_path.read_text())['data']['rows']
    else:
        response = requests.get(NASDAQ_API)
        rows = json.loads(response.text)['data']['rows']
    res = pl.DataFrame(rows)
    res = res.sample(len(res), shuffle=True)
    return res


def make_batches(arr, batch_size):
    rounded_size = (arr.shape[0] // batch_size) * batch_size
    arr = arr[:rounded_size]
    arr = arr.reshape((-1, batch_size))
    return arr


MIN_CANDLES = 8


def preprocess_history(history: pl.DataFrame):
    ma7 = pl.col('close').rolling_mean(7)
    ma21 = pl.col('close').rolling_mean(21)
    band_std = pl.col('close').rolling_std(7) * 2
    ema12 = pl.col('close').ewm_mean(span=12)
    ema26 = pl.col('close').ewm_mean(span=26)
    return history.select(
        'open', 'close', 'high', 'low', 'volume',
        ma7=ma7,
        ma21=ma21,
        lower_band=(ma7 - band_std).clip(1e-3),
        upper_band=ma7 + band_std,
        ema12=ema12,
        ema26=ema26,
    ).select(
        (pl.selectors.float().exclude('volume').log()),
        (pl.col('volume').log())
    ).drop_nulls()


def get_features_count():
    empty_df = pl.DataFrame(dict.fromkeys(
        ['open', 'close', 'high', 'low', 'volume'],
        pl.Series(dtype=pl.Float32)
    ))
    return len(preprocess_history(empty_df).columns)


def generate_batches(history: pl.DataFrame, key: jax.Array, batch_size: int, data_mesh: jax.sharding.Sharding, batch_mesh: jax.sharding.Sharding):
    history = preprocess_history(history)
    # Reduce number of lengths and number of functions that JAX needs to compile
    shrunk_len = 2 ** ((len(history) - 1).bit_length() - 1)
    if shrunk_len < 1:
      return None, None
    history = history.slice(-shrunk_len)
    bad_rows = history.filter(
        pl.all_horizontal(
            pl.col('open', 'close', 'high', 'low').pipe(
                lambda col: col.is_nan() | col.is_null() | (col <= 0)
            )
        )
    )
    if len(history) < LAG + PREDICTION_PERIOD + MIN_CANDLES or len(bad_rows) > 0:
        return None, None
    candles = history.to_jax().to_device(data_mesh)
    possible_starts = jnp.array(len(history) - PREDICTION_PERIOD - LAG)
    starts = jax.random.permutation(key, possible_starts)
    batches = make_batches(starts, batch_size).to_device(batch_mesh)
    return candles, batches


def get_last_prediction(ticker_name: str, shard: jax.sharding.Sharding):
    ticker = yf.Ticker(ticker_name)
    history = pl.from_pandas(ticker.history(period="max"), include_index=True)
    if len(history) == 0:
        return None
    history = history.select(
        pl.col('Open', 'Close', 'High', 'Low', 'Volume').name.to_lowercase(),
    )
    history = preprocess_history(history)
    if len(history) < PREDICTION_PERIOD + LAG:
        return None
    x = history[-LAG-PREDICTION_PERIOD:-PREDICTION_PERIOD].to_jax().to_device(shard)
    y = history[-PREDICTION_PERIOD:, 'close'].to_jax().to_device(shard)
    return x, y


def get_last_predictions(*ticker_names: str, shard: jax.sharding.Sharding):
    xs = []
    ys = []
    for ticker_name in ticker_names:
        data = get_last_prediction(ticker_name, shard)
        if data is None:
            continue
        x, y = data
        xs.append(x)
        ys.append(y)
    return jnp.stack(xs), jnp.stack(ys)
