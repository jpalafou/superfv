import numpy as np
import pytest

from superfv.tools.norms import l1_norm, l2_norm, linf_norm


def test_l1_norm():
    arr = np.array([-1, 2, -3, 4])
    assert l1_norm(arr) == pytest.approx(2.5)


def test_l2_norm():
    arr = np.array([3, 4])
    assert l2_norm(arr) == pytest.approx(5.0 / np.sqrt(2))


def test_linf_norm():
    arr = np.array([-1, -9, 5])
    assert linf_norm(arr) == 9
