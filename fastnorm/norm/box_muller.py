"""
Box-Muller transform generator of standard normal floats.

See https://en.wikipedia.org/wiki/Box-Muller_transform
"""

import math

import numba as nb
import numpy as np

from fastnorm import splitmix64
from fastnorm.types import Vector


# NOTE: taking uint64 array as input so this can be benchmarked independently of the bit generator
@nb.jit(nb.void(nb.float64[:], nb.uint64[:]), boundscheck=False, fastmath=True)
def _fill_from_ints(z: Vector[np.float64], ints: Vector[np.uint64]) -> None:
    """Fill array with standard normal 64-bit floats from array of 64-bit unsigned integers."""
    # assumes integer array length is divisible by 2
    for i in range(0, ints.shape[0], 2):
        u1 = splitmix64.to_float(ints[i])
        u2 = splitmix64.to_float(ints[i + 1])
        r = math.sqrt(-2 * math.log(u1))
        x = 2 * math.pi * u2
        z[i] = r * math.cos(x)
        z[i + 1] = r * math.sin(x)


@nb.jit(nb.float64[:](nb.uint64, nb.uint64))
def sample(nsamples: int, seed: int) -> Vector[np.float64]:
    """Sample an array of standard normal 64-bit floats."""
    _nsamples = 2 * math.ceil(nsamples / 2)  # generate extra sample if nsamples is not divisible by 2
    ints = splitmix64.sample_ints(_nsamples, seed)
    samples = np.empty(ints.shape[0], dtype=np.float64)
    _fill_from_ints(samples, ints)
    return samples[:nsamples]  # drop last sample if nsamples is not divisible by 2
