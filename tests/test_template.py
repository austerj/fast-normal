import pytest


def test_addition():
    assert 1 + 1 == 2


def test_division():
    assert 2 // 2 == 1

    with pytest.raises(ZeroDivisionError):
        x = 2 // 0
