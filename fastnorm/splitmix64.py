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
def _fill_array_ints(array: np.ndarray, seed: int) -> int:
    """Fill array in-place with uniform 64-bit unsigned integers and return latest seed."""
    for i in range(array.shape[0]):
        seed, n = _step(seed)
        array[i] = n
    return seed


@nb.jit(nb.float64(nb.float64[:], nb.uint64), boundscheck=False)
def _fill_array_floats(array: np.ndarray, seed: int) -> int:
    """Fill array in-place with uniform 64-bit floats in [0,1) and return latest seed."""
    for i in range(array.shape[0]):
        seed, n = _step(seed)
        # use highest 53 bits of unsigned int for float in [0, 1)
        # see https://prng.di.unimi.it/ (Generating uniform doubles in the unit interval)
        array[i] = nb.float64((n >> 11) * 2 ** (-53))
    return seed


@nb.jit(nb.uint64[:](nb.uint64, nb.uint64))
def sample_ints(nsamples: int, seed: int) -> np.ndarray[tuple[int], np.dtype[np.uint64]]:
    """Sample an array of uniform 64-bit unsigned integers."""
    array = np.empty(nsamples, dtype=np.uint64)
    _fill_array_ints(array, seed)
    return array


@nb.jit(nb.float64[:](nb.uint64, nb.uint64))
def sample_floats(nsamples: int, seed: int) -> np.ndarray[tuple[int], np.dtype[np.float64]]:
    """Sample an array of uniform 64-bit floats in [0, 1)."""
    array = np.empty(nsamples, dtype=np.float64)
    _fill_array_floats(array, seed)
    return array
