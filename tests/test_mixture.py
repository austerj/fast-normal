import itertools
import math

import numpy as np
import pytest
from scipy.special import erfinv

from fastnorm.norm import box_muller, mixture

MGF_VALS = [math.exp(i**2 / 2) for i in range(3)]


def restrict(array: np.ndarray, qrestrict: float):
    # get quantiles
    lower = np.quantile(array, 1 - qrestrict)
    upper = np.quantile(array, qrestrict)
    # exclude values outside range
    excluded = array
    excluded = excluded[excluded > lower]
    excluded = excluded[excluded < upper]
    return excluded


def estimate_mgf(array: np.ndarray, i: int):
    return np.mean(np.exp(array * i))


def test_mgf():
    qmax = 0.99999
    nsamples = 5_000_000
    samples = mixture.sampler(qmax)(nsamples, seed=0)

    # passes first and second MGF
    for i, moment in enumerate(MGF_VALS[:2]):
        assert estimate_mgf(samples, i) == pytest.approx(moment, rel=1e-2)

    # expectedly FAILS third MGF value - cannot capture tail behavior even with high samples!
    assert estimate_mgf(samples, 2) != pytest.approx(MGF_VALS[2], rel=1e-2)

    # when excluding tail samples, the third MGF estimate matches box-muller
    qrestrict = 0.999
    bm_samples = restrict(box_muller.sample(nsamples, seed=0), qrestrict)
    assert estimate_mgf(restrict(samples, qrestrict), 2) == pytest.approx(estimate_mgf(bm_samples, 2), rel=1e-2)


def test_pseudocode():
    def quantiles_pseudocode(npartitions: int, q: float):
        # compute and scale quantiles via combined iteration (analogous to pseudocode)
        quantiles = np.zeros(npartitions + 1)
        delta = (q - 0.5) / npartitions
        c = 0.0
        for i in range(1, npartitions + 1):
            step = 0.5 + i * delta
            quantiles[i] = math.sqrt(2) * erfinv(2 * step - 1)
            c += quantiles[i] * (quantiles[i] + quantiles[i - 1]) + quantiles[i - 1] ** 2
        c = math.sqrt(c / (3 * npartitions))
        quantiles /= c
        return quantiles

    # check that implementation with comprehensions matches pseudocode in paper
    ns = [1, 2, 3, 4]
    qs = [0.6, 0.7, 0.9, 0.99]
    for n, q in itertools.product(ns, qs):
        npartitions = 2**n
        # compute and scale quantiles via comprehensions
        Q = mixture._invert_cdf(npartitions + 1, q)
        Q /= math.sqrt(mixture._var(Q))
        assert np.allclose(Q, quantiles_pseudocode(npartitions, q))
