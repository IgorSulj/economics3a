from typing import Callable
from flax import nnx
import jax
import jax.numpy as jnp

import consts


class PositionalEncoding(nnx.Module):
    def __init__(self, seq_len: int, n_features: int, dropout: float, rngs: nnx.Rngs):
        super().__init__()
        self.seq_len = seq_len
        self.n_features = n_features
        self.dropout = nnx.Dropout(dropout, rngs=rngs)

        self.pe = nnx.Param(self.calc_encodings(seq_len, n_features))
    
    @staticmethod
    def calc_encodings(n_inputs: int, n_features: int):
        @nnx.vmap(in_axes=(0, None), out_axes=0)
        @nnx.vmap(in_axes=(None, 0), out_axes=0)
        def calc(pos: jax.Array, feature: jax.Array):
            return jnp.where(
                feature % 2 == 0,
                jnp.sin(pos / 10000**(feature / 2 / n_features)),
                jnp.cos(pos / 10000**((feature - 1) / 2 / n_features)),
            )
        return calc(jnp.arange(n_inputs), jnp.arange(n_features))
    
    def __call__(self, x: jax.Array):
        res = x + self.pe.value
        return self.dropout(res)


class SwiGLU(nnx.Module):
    def __init__(self, d_model: int, d_hidden: int, dropout: float, *, rngs: nnx.Rngs):
        self.linear1 = nnx.Linear(d_model, d_hidden, use_bias=False, rngs=rngs)
        self.linear2 = nnx.Linear(d_model, d_hidden, use_bias=False, rngs=rngs)
        self.dropout = nnx.Dropout(dropout, rngs=rngs)
        self.linear3 = nnx.Linear(d_hidden, d_model, use_bias=False, rngs=rngs)
    
    def __call__(self, x: jax.Array):
        return self.linear3(
            self.dropout(nnx.swish(self.linear1(x)) * self.linear2(x))
        )


class ResidualConnection(nnx.Module):
    def __init__(self, n_features: int, dropout: float, rngs: nnx.Rngs):
        self.dropout = nnx.Dropout(dropout, rngs=rngs)
        self.norm = nnx.LayerNorm(n_features, rngs=rngs)
    
    def __call__(self, sublayer: Callable[[jax.Array], jax.Array], x: jax.Array) -> jax.Array:
        return self.norm(x + self.dropout(sublayer(x)))


class EncoderBlock(nnx.Module):
    def __init__(self, self_attention: nnx.MultiHeadAttention, feed_forward: Callable[[jax.Array], jax.Array], dropout: float, rngs: nnx.Rngs):
        self.self_attention = self_attention
        self.feed_forward = feed_forward
        self.in_features = self.self_attention.in_features
        self.residual1 = ResidualConnection(self.in_features, dropout, rngs=rngs)
        self.residual2 = ResidualConnection(self.in_features, dropout, rngs=rngs)
    
    def __call__(self, x: jax.Array, cross: jax.Array | None = None, src_mask: jax.Array | None = None):
        x = self.residual1(lambda x: self.self_attention(x, cross, mask=src_mask), x)
        x = self.residual2(self.feed_forward, x)
        return x


class EasyEncoderBlock(nnx.Module):
    def __init__(self, n_heads: int, n_features: int, n_hidden: int, dropout: float, *, rngs: nnx.Rngs):
        self_attention = nnx.MultiHeadAttention(n_heads, n_features, decode=False, rngs=rngs)
        feed_forward = SwiGLU(n_features, n_hidden, dropout=dropout, rngs=rngs)
        self.block = EncoderBlock(self_attention, feed_forward, dropout, rngs=rngs)
    
    def __call__(self, x: jax.Array, cross: jax.Array | None = None, src_mask: jax.Array | None = None):
        return self.block(x, cross, src_mask)


class Encoder(nnx.Module):
    def __init__(self, layers: tuple[EasyEncoderBlock, ...], n_features: int, rngs: nnx.Rngs):
        self.layers = layers
        self.norm = nnx.LayerNorm(n_features, rngs=rngs)
    
    @staticmethod
    def with_n_layers(n_layers: int, n_features: int, n_hidden: int, n_heads: int, dropout: float, rngs: nnx.Rngs):
        layers = tuple(
            EasyEncoderBlock(
                n_heads,
                n_features,
                n_hidden,
                dropout,
                rngs=rngs,
            )
            for _ in range(n_layers)
        )
        return Encoder(layers, n_features, rngs=rngs)
    
    def __call__(self, x: jax.Array, mask: jax.Array | None = None):
        for layer in self.layers:
            x = layer(x, mask)
        return self.norm(x)


class iTransformer(nnx.Module):
    def __init__(self, seq_len: int, out_seq_len: int, n_layers: int, n_features: int, d_model: int, n_hidden: int, dropout: float, *, rngs: nnx.Rngs):
        self.embedding = nnx.Linear(seq_len, d_model, rngs=rngs)
        self.pe = PositionalEncoding(n_features, d_model, dropout, rngs=rngs)
        self.encoder = Encoder.with_n_layers(
            n_layers,
            d_model,
            n_hidden,
            consts.N_HEADS,
            dropout,
            rngs=rngs
        )
        self.final_pred = nnx.LinearGeneral((n_features, d_model), out_seq_len, axis=(-2, -1), rngs=rngs)
    
    def __call__(self, x: jax.Array):
        x = self.embedding(x.T)
        x = self.pe(x)
        x = self.encoder(x)
        return self.final_pred(x)
