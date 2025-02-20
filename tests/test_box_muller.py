import math

import numpy as np
import pytest

from fastnorm.norm import box_muller

MGF_VALS = [math.exp(i**2 / 2) for i in range(3)]


def test_mgf():
    samples = box_muller.sample(5_000_000, seed=0)
    for i, moment in enumerate(MGF_VALS):
        assert np.mean(np.exp(samples * i)) == pytest.approx(moment, rel=1e-2)
