import math

import numpy as np
import pytest

from fastnorm.norm import box_muller, lookup

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
    npartitions, qmax = 1024, 0.99999
    nsamples = 5_000_000
    samples = lookup.sampler(npartitions, qmax)(nsamples, seed=0)

    # passes first and second MGF
    for i, moment in enumerate(MGF_VALS[:2]):
        assert estimate_mgf(samples, i) == pytest.approx(moment, rel=1e-2)

    # expectedly FAILS third MGF value - cannot capture tail behavior even with high samples!
    assert estimate_mgf(samples, 2) != pytest.approx(MGF_VALS[2], rel=1e-2)

    # when exclude tail samples, the third MGF estimate matches box-muller
    qrestrict = 0.999
    bm_samples = restrict(box_muller.sample(nsamples, seed=0), qrestrict)
    assert estimate_mgf(restrict(samples, qrestrict), 2) == pytest.approx(estimate_mgf(bm_samples, 2), rel=1e-2)
