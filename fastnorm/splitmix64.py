"""
SplitMix64 generator of pseudo-random unsigned 64-bit integers.

See https://xoshiro.di.unimi.it/splitmix64.c
"""

import numba as nb
import numpy as np
from numba import types


@nb.jit(types.UniTuple(nb.uint64, 2)(nb.uint64))
def _step(seed: int) -> tuple[int, int]:
    """Generate uniform 64-bit unsigned integer and return incremented seed and resulting value."""
    seed += 0x9E3779B97F4A7C15
    n = seed
    n ^= (n >> 30) * 0xBF58476D1CE4E5B9
    n ^= (n >> 27) * 0x94D049BB133111EB
    n ^= n >> 31
    return seed, n


@nb.jit(nb.uint64(nb.uint64[:], nb.uint64), boundscheck=False)
def fill_array(array: np.ndarray, seed: int) -> int:
    """Fill array in-place with uniform 64-bit unsigned integers and return latest seed."""
    for i in range(array.shape[0]):
        seed, n = _step(seed)
        array[i] = n
    return seed


@nb.jit(nb.uint64[:](nb.uint64, nb.uint64))
def sample(nsamples: int, seed: int) -> np.ndarray[tuple[int], np.dtype[np.uint64]]:
    """Sample an array of uniform 64-bit unsigned integers."""
    array = np.empty(nsamples, dtype=np.uint64)
    fill_array(array, seed)
    return array
