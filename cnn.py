from typing import Sequence
from flax import nnx
import jax
import jax.numpy as jnp


class MultiCNN1D(nnx.Module):
    def __init__(self, in_features: int, hidden_kernel_features: int, kernel_sizes: Sequence[int], *, rngs: nnx.Rngs):
        self.convs = [
            nnx.Conv(in_features, hidden_kernel_features, (kernel_size,), rngs=rngs)
            for kernel_size in kernel_sizes
        ]
    
    def __call__(self, x: jax.Array):
        results = [conv(x) for conv in self.convs]
        return jnp.hstack(results)


primes = [1, 2, 3, 5, 7, 11, 13, 17, 19]


class OneBlock(nnx.Module):
    def __init__(self, in_features: int, hidden_kernel_features: int, out_features: int, *, rngs: nnx.Rngs):
        n_primes = len(primes)
        self.inner = nnx.Sequential(
            MultiCNN1D(in_features, hidden_kernel_features, primes, rngs=rngs),
            nnx.relu,
            nnx.LayerNorm(hidden_kernel_features * n_primes, rngs=rngs),
            MultiCNN1D(hidden_kernel_features * n_primes, hidden_kernel_features, primes, rngs=rngs),
            nnx.relu,
            nnx.LayerNorm(hidden_kernel_features * n_primes, rngs=rngs),
            nnx.Conv(hidden_kernel_features * n_primes, out_features, (1,), rngs=rngs),
        )
    
    def __call__(self, x: jax.Array) -> jax.Array:
        return self.inner(x)
