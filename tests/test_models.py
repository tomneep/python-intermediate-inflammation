"""Tests for statistics functions within the Model layer."""

import pytest
import numpy as np
import numpy.testing as npt

from inflammation.models import daily_mean


@pytest.mark.parametrize(
    ["func", "input_arr", "expected"],
    [
        pytest.param(daily_mean, np.array([[0, 0], [0, 0], [0, 0]]), np.array([0, 0]), id="mean_zeros"),
        pytest.param(daily_mean, np.array([[1, 2], [3, 4], [5, 6]]), np.array([3, 4]), id="mean_integers"),
    ]
)
def test_daily_func(func, input_arr, expected):
    npt.assert_array_almost_equal(func(input_arr), expected)
