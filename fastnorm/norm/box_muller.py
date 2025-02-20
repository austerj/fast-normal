"""
Box-Muller transform generator of standard normal floats.

See https://en.wikipedia.org/wiki/Box-Muller_transform
"""

import math

import numba as nb
import numpy as np

from fastnorm import splitmix64
from fastnorm.types import Vector


@nb.jit(nb.void(nb.float64[:]), boundscheck=False, fastmath=True)
def _transform_array(array: np.ndarray) -> None:
    """Transform array of uniform 64-bit floats to standard normal samples via Box-Muller transform."""
    # assumes array length is divisible by 2
    for i in range(0, array.shape[0], 2):
        u1, u2 = array[i], array[i + 1]
        r = math.sqrt(-2 * math.log(u1))
        x = 2 * math.pi * u2
        array[i] = r * math.cos(x)
        array[i + 1] = r * math.sin(x)


@nb.jit(nb.float64[:](nb.uint64, nb.uint64))
def sample(nsamples: int, seed: int) -> Vector[np.float64]:
    """Sample an array of standard normal 64-bit floats."""
    _nsamples = 2 * math.ceil(nsamples / 2)  # generate extra sample if nsamples is not divisible by 2
    array = splitmix64.sample_floats(_nsamples, seed)
    _transform_array(array)
    return array[:nsamples]  # drop last sample if nsamples is not divisible by 2
