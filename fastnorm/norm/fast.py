"""
Fast generator of approximately standard normal floats via truncated distribution lookups.

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


def _sampler_from_ints(npartitions: int, qmax: float):
    """Generate sampling function for approximately standard normal from uniform 64-bit unsigned integers."""
    if npartitions > 2**10:
        raise ValueError("Number of partitions must fit within 10 bits")

    table = _invert_cdf(npartitions + 1, qmax)

    # NOTE: taking uint64 array as input so this can be benchmarked independently of the bit generator
    @nb.jit(nb.float64[:](nb.uint64[:]), boundscheck=False, fastmath=True)
    def sample_from_ints(ints: Vector[np.uint64]) -> Vector[np.float64]:
        """Sample an array of approximately standard normal 64-bit floats from array of 64-bit unsigned integers."""
        nsamples = ints.shape[0]
        z = np.empty(nsamples, dtype=np.float64)
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
            # retrieve sign from first bit
            is_negative = bool(n << (64 - 1))
            z[i] = (-is_negative + (not is_negative)) * abs_z  # equivalent: -abs_z if is_negative else abs_z
        return z

    return sample_from_ints


def sampler(npartitions: int, qmax: float):
    """Generate sampling function of approximately standard normal 64-bit floats."""
    sample_from_ints = _sampler_from_ints(npartitions, qmax)

    @nb.jit(nb.float64[:](nb.uint64, nb.uint64))
    def sample(nsamples: int, seed: int) -> Vector[np.float64]:
        """Sample an array of approximately standard normal 64-bit floats."""
        ints = splitmix64.sample_ints(nsamples, seed)
        return sample_from_ints(ints)

    return sample
