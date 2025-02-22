"""
Generator of approximately standard normal floats via truncated distribution lookups.

See {TODO: link to article}
"""

import numba as nb
import numpy as np
from scipy import stats

from fastnorm import splitmix64
from fastnorm.types import Vector


def _invert_cdf(npartitions: int, qmax: float) -> Vector[np.float64]:
    """Equidistant steps of inverse standard normal CDF (quantiles) from mid-point up to provided max."""
    if not (0.5 < qmax < 1.0):
        raise ValueError("Maximal quantile must be strictly between 0.5 and 1.0")
    return stats.norm.ppf([0.5 + i / (npartitions - 1) * (qmax - 0.5) for i in range(npartitions)])


def _filler_from_ints(npartitions: int, qmax: float):
    """Generate filler function for approximately standard normal from uniform 64-bit unsigned integers."""
    if npartitions > 2**10:
        raise ValueError("Number of partitions must fit within 10 bits")

    table: Vector[np.float64] = _invert_cdf(npartitions + 1, qmax)

    # NOTE: taking uint64 array as input so this can be benchmarked independently of the bit generator
    @nb.jit(nb.void(nb.float64[::1], nb.uint64[::1]), boundscheck=False, fastmath=True)
    def fill_from_ints(z: Vector[np.float64], ints: Vector[np.uint64]) -> None:
        """Fill array with approximately standard normal 64-bit floats from array of 64-bit unsigned integers."""
        nsamples = ints.shape[0]
        for i in range(nsamples):
            n = ints[i]
            # truncate to lowest 11 bits, then remove lowest bit (sign)
            idx = (n << (64 - (10 + 1))) >> (64 - 10)
            idx *= npartitions  # multiply by number of unique values...
            idx >>= 10  # ...then scale down to 0, 1, 2, ..., npartitions - 1
            # use highest 53 bits of unsigned int for float in [0, 1)
            f = (n >> 11) * 2 ** (-53)
            # generate uniform sample of absolute value within table segment
            l, u = table[idx], table[idx + 1]
            abs_z = l + f * (u - l)
            # retrieve sign from lowest bit
            z[i] = -abs_z if n & 1 else abs_z

    return fill_from_ints


def sampler(npartitions: int, qmax: float):
    """Generate sampling function of approximately standard normal 64-bit floats."""
    fill_from_ints = _filler_from_ints(npartitions, qmax)

    @nb.jit(nb.float64[::1](nb.uint64, nb.uint64))
    def sample(nsamples: int, seed: int) -> Vector[np.float64]:
        """Sample an array of approximately standard normal 64-bit floats."""
        ints = splitmix64.sample_ints(nsamples, seed)
        samples = np.empty(nsamples, dtype=np.float64)
        fill_from_ints(samples, ints)
        return samples

    return sample
