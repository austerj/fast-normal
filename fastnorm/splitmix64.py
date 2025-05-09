"""
SplitMix64 generator of pseudo-random unsigned 64-bit integers.

See https://xoshiro.di.unimi.it/splitmix64.c
"""

import numba as nb
import numpy as np
from numba import types

from fastnorm.types import Vector


@nb.jit(types.UniTuple(nb.uint64, 2)(nb.uint64))
def _step(seed: int) -> tuple[int, int]:
    """Generate uniform 64-bit unsigned integer and return incremented seed and resulting value."""
    seed += 0x9E3779B97F4A7C15
    n = seed
    n ^= (n >> 30) * 0xBF58476D1CE4E5B9
    n ^= (n >> 27) * 0x94D049BB133111EB
    n ^= n >> 31
    return seed, n


@nb.jit(nb.float64(nb.uint64))
def to_float(n: int) -> float:
    """Get float in [0, 1) from 53 highest bits of a 64-bit unsigned integer."""
    return (n >> 11) / 2**53


@nb.jit(nb.uint64(nb.uint64[::1], nb.uint64), boundscheck=False)
def fill_ints(array: np.ndarray, seed: int) -> int:
    """Fill array in-place with uniform 64-bit unsigned integers and return latest seed."""
    for i in range(array.shape[0]):
        seed, n = _step(seed)
        array[i] = n
    return seed


@nb.jit(nb.float64(nb.float64[::1], nb.uint64), boundscheck=False)
def fill_floats(array: np.ndarray, seed: int) -> int:
    """Fill array in-place with uniform 64-bit floats in [0,1) and return latest seed."""
    for i in range(array.shape[0]):
        seed, n = _step(seed)
        array[i] = to_float(n)
    return seed


@nb.jit(nb.uint64[::1](nb.uint64, nb.uint64))
def sample_ints(nsamples: int, seed: int) -> Vector[np.uint64]:
    """Sample an array of uniform 64-bit unsigned integers."""
    array = np.empty(nsamples, dtype=np.uint64)
    fill_ints(array, seed)
    return array


@nb.jit(nb.float64[::1](nb.uint64, nb.uint64))
def sample_floats(nsamples: int, seed: int) -> Vector[np.float64]:
    """Sample an array of uniform 64-bit floats in [0, 1)."""
    array = np.empty(nsamples, dtype=np.float64)
    fill_floats(array, seed)
    return array
