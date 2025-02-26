import math

from scipy.special import erfinv


def pdf(x: float) -> float:
    """Evaluate the standard normal probability density function."""
    return math.exp(-(x**2) / 2) / math.sqrt(2 * math.pi)


def cdf(x: float) -> float:
    """Evaluate the standard normal cumulative distribution function."""
    return (1 + math.erf(x)) / 2


def cdfinv(q: float) -> float:
    """Evaluate the standard normal cumulative distribution function inverse (quantile)."""
    # inverting the error function is a non-trivial problem and typically solved by evaluating
    # fitted polynomials, hence we rely on SciPy here
    # see e.g. https://github.com/jeremybarnes/cephes/blob/master/cprob/ndtri.c
    return math.sqrt(2) * erfinv(2 * q - 1)
