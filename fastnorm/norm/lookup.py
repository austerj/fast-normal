"""
Generator of approximately standard normal floats via truncated distribution lookups.

See {TODO: link to article}
"""

import math

import numba as nb
import numpy as np

from fastnorm import splitmix64
from fastnorm.norm.dist import cdfinv
from fastnorm.types import Vector

_DEFAULT_EXPONENT = 10
_HELLINGER_FACTOR = (2 * math.pi) ** (1 / 4)


def _var(quantiles: Vector[np.float64]) -> float:
    """Variance of a symmetric uniform mixture distribution from right quantiles."""
    NPARTITIONS = quantiles.shape[0] - 1
    return 1 / (3 * NPARTITIONS) * sum(a * (a + b) + b**2 for a, b in zip(quantiles[:-1], quantiles[1:]))


def _invert_cdf(nsteps: int, q: float, rescale: bool) -> Vector[np.float64]:
    """Equidistant steps of inverse standard normal CDF from mid-point up to provided max (rescaled to unit variance)."""
    if not (0.5 < q < 1.0):
        raise ValueError("Maximal q probability must be strictly between 0.5 and 1.0")
    # compute quantiles across equidistant steps
    steps = [0.5 + i / (nsteps - 1) * (q - 0.5) for i in range(nsteps)]
    quantiles = np.array([cdfinv(q) for q in steps], dtype=np.float64)
    # rescale to unit variance
    if rescale:
        quantiles /= math.sqrt(_var(quantiles))
    return quantiles


def _hellinger_distance(q: float, npartitions: int) -> float:
    """Hellinger distance between mixture approximation and standard normal distribution."""
    # compute quantiles and adjustment constant from provided parameters
    quantiles = _invert_cdf(npartitions + 1, q, rescale=False)
    c = math.sqrt(_var(quantiles))
    # compute sum term
    erfs = [math.erf(quantile / (2 * c)) for quantile in quantiles]
    numerators = [b - a for a, b in zip(erfs[:-1], erfs[1:])]
    denominators = [math.sqrt(b - a) for a, b in zip(quantiles[:-1], quantiles[1:])]
    h_sum = sum(n / d for n, d in zip(numerators, denominators))
    # finally: compute Hellinger distance
    h_squared = 1 - _HELLINGER_FACTOR * math.sqrt(c / npartitions) * h_sum
    return math.sqrt(h_squared)


def _filler_from_ints(qmax: float, exponent: int = _DEFAULT_EXPONENT, rescale: bool = True):
    """Generate filler function for approximately standard normal from uniform 64-bit unsigned integers."""
    if not (0 < exponent <= 10):
        raise ValueError("Exponent must be strictly between 0 and 10")

    # precompute lookup table of quantiles
    NPARTITIONS = 2**exponent
    Q: Vector[np.float64] = _invert_cdf(NPARTITIONS + 1, qmax, rescale)

    @nb.jit(nb.void(nb.float64[::1], nb.uint64[::1]), boundscheck=False, fastmath=True)
    def fill_from_ints(z: Vector[np.float64], ints: Vector[np.uint64]) -> None:
        """Fill array with approximately standard normal 64-bit floats from array of 64-bit unsigned integers."""
        nsamples = ints.shape[0]
        for i in range(nsamples):
            n = ints[i]
            # remove lowest bit and mask to table index bits
            idx = (n >> 1) & (NPARTITIONS - 1)
            # use highest 53 bits for float in [0, 1)
            f = (n >> 11) * 2**-53
            # generate uniform sample of absolute value within partition
            l, u = Q[idx], Q[idx + 1]
            abs_z = l + f * (u - l)
            # retrieve sign from lowest bit
            z[i] = -abs_z if n & 1 else abs_z

    return fill_from_ints


def sampler(qmax: float, exponent: int = _DEFAULT_EXPONENT, rescale: bool = True):
    """Generate sampling function of approximately standard normal 64-bit floats."""
    fill_from_ints = _filler_from_ints(qmax, exponent, rescale)

    @nb.jit(nb.float64[::1](nb.uint64, nb.uint64))
    def sample(nsamples: int, seed: int) -> Vector[np.float64]:
        """Sample an array of approximately standard normal 64-bit floats."""
        ints = splitmix64.sample_ints(nsamples, seed)
        samples = np.empty(nsamples, dtype=np.float64)
        fill_from_ints(samples, ints)
        return samples

    return sample
