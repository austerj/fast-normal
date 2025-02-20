import numpy as np
import pytest

from fastnorm import splitmix64

MOMENTS = [1 / (i + 1) for i in range(10)]


# NOTE: matching a finite number of moments does NOT imply that the distribution is iid uniform - in
# particular this does not assess independence. this serves as more of an implementation test of a
# method presumed to be reasonably sound.
def test_moments():
    samples = splitmix64.sample_floats(50_000, seed=0)
    for i, moment in enumerate(MOMENTS):
        assert np.mean(samples**i) == pytest.approx(moment, rel=1e-2)
