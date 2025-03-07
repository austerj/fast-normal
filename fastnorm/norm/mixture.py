"""
Fast transform for approximately standard normal floats via uniform mixture.

See {TODO: link to article}
"""

import math
import typing
from logging import warning

import numba as nb
import numpy as np
from scipy.optimize import OptimizeResult, minimize_scalar
from scipy.special import erfinv

from fastnorm import splitmix64
from fastnorm.types import Vector

_DEFAULT_EXPONENT = 10
_HELLINGER_FACTOR = (2 * math.pi) ** (1 / 4)
# factor in regression q ~ 1 - 1 / (INVERSE_FACTOR * N)
# see scripts/plots/hellinger.py
INVERSE_FACTOR = 26.679816319102272


def _var(quantiles: Vector[np.float64]) -> float:
    """Variance of a symmetric uniform mixture distribution from right quantiles."""
    NPARTITIONS = quantiles.shape[0] - 1
    return 1 / (3 * NPARTITIONS) * sum(a * (a + b) + b**2 for a, b in zip(quantiles[:-1], quantiles[1:]))


def _invert_cdf(nsteps: int, q: float) -> Vector[np.float64]:
    """Equidistant steps of inverse standard normal CDF from mid-point up to provided q."""
    if not (0.5 < q < 1.0):
        raise ValueError("Maximal q probability must be strictly between 0.5 and 1.0")
    # compute quantiles across equidistant steps
    steps = np.linspace(0.5, q, nsteps, dtype=np.float64)
    # inverting the error function is a non-trivial problem and typically solved by evaluating
    # fitted polynomials, hence we rely on SciPy here
    # see e.g. https://github.com/jeremybarnes/cephes/blob/master/cprob/ndtri.c
    quantiles = math.sqrt(2) * erfinv(2 * steps - 1)
    return quantiles


def hist(npartitions: int, q: float) -> tuple[Vector[np.float64], Vector[np.float64]]:
    """Mixture probability density function as numpy histogram (counts, bins)."""
    # create scaled right quantiles
    quantiles = _invert_cdf(npartitions + 1, q)
    quantiles /= math.sqrt(_var(quantiles))
    # mirror quantiles (skipping midpoint == 0 from left side)
    bins = np.hstack([-quantiles[1:][::-1], quantiles])
    return np.histogram(bins[:-1], bins, density=True)


def hellinger(q: float, npartitions: int) -> float:
    """Hellinger distance between mixture approximation and standard normal distribution."""
    # compute quantiles and variance adjustment from provided parameters
    quantiles = _invert_cdf(npartitions + 1, q)
    c = math.sqrt(_var(quantiles))
    # compute sum term
    erfs = [math.erf(quantile / (2 * c)) for quantile in quantiles]
    numerators = [b - a for a, b in zip(erfs[:-1], erfs[1:])]
    denominators = [math.sqrt(b - a) for a, b in zip(quantiles[:-1], quantiles[1:])]
    h_sum = sum(n / d for n, d in zip(numerators, denominators))
    # finally: compute Hellinger distance
    h_squared = 1 - _HELLINGER_FACTOR * math.sqrt(c / npartitions) * h_sum
    return math.sqrt(h_squared)


def minimize_hellinger(npartitions: int) -> OptimizeResult:
    """Minimize the Hellinger distance as a function of q for the given number of partitions."""
    # 0.9 is well below the optimum even for N=2, so this is a safe lower bound for a minimum
    a, b = 0.9, np.nextafter(1.0, -1)
    cost = lambda q: hellinger(q, npartitions)
    result = minimize_scalar(cost, bounds=(a, b), method="bounded", options={"xatol": 1e-10})
    return typing.cast(OptimizeResult, result)


def optimal_q(npartitions: int) -> float:
    """Retrieve the Hellinger distance minimizing value of q for the given number of partitions."""
    # use relationship from regression
    return 1.0 - 1 / (INVERSE_FACTOR * npartitions)


def filler(q: float | None = None, exponent: int = _DEFAULT_EXPONENT, warn: bool = True):
    """Generate filler function for approximately standard normal 64-bit floats from 64-bit unsigned integers."""
    if not (0 < exponent <= 32) or not isinstance(exponent, int):
        raise ValueError("Exponent must be a positive integer less than or equal to 32")
    elif exponent > 10 and warn:
        warning("Exponents greater than 10 may adversely affect floating point precision")

    # find q that minimizes Hellinger distance if q is not given explicitly
    NPARTITIONS = 2**exponent
    if q is None:
        # use q from regression
        q = typing.cast(float, optimal_q(NPARTITIONS))

    # create lookup table of quantiles and rescale to unit variance
    Q: Vector[np.float64] = _invert_cdf(NPARTITIONS + 1, q)
    Q /= math.sqrt(_var(Q))

    # using at most 53 bits for float to still get uniform floats in [0,1) for exponents below 10
    FLOAT_SHIFT = nb.literally(1 + max(exponent, 10))  # 1 bit reserved for sign
    FLOAT_DIVISOR = nb.literally(2 ** (64 - FLOAT_SHIFT))

    @nb.jit(nb.void(nb.float64[::1], nb.uint64[::1]), boundscheck=False, fastmath=True)
    def fill(z: Vector[np.float64], ints: Vector[np.uint64]) -> None:
        """Fill array with approximately standard normal 64-bit floats from array of 64-bit unsigned integers."""
        nsamples = ints.shape[0]
        for j in range(nsamples):
            U = ints[j]
            # remove lowest bit and mask to table index bits
            i = (U >> 1) & (NPARTITIONS - 1)
            # use remaining (or 53 at most) bits for float in [0, 1)
            a = (U >> FLOAT_SHIFT) / FLOAT_DIVISOR
            # generate uniform sample of absolute value within partition
            l, u = Q[i], Q[i + 1]
            x = l + a * (u - l)
            # retrieve sign from lowest bit
            z[j] = -x if U & 1 else x

    return fill


def sampler(q: float | None = None, exponent: int = _DEFAULT_EXPONENT, warn: bool = True):
    """Generate sampling function of approximately standard normal 64-bit floats."""
    fill = filler(q, exponent, warn)

    @nb.jit(nb.float64[::1](nb.uint64, nb.uint64))
    def sample(nsamples: int, seed: int) -> Vector[np.float64]:
        """Sample an array of approximately standard normal 64-bit floats."""
        ints = splitmix64.sample_ints(nsamples, seed)
        samples = np.empty(nsamples, dtype=np.float64)
        fill(samples, ints)
        return samples

    return sample
