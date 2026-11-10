"""The 2025 project's numbers must be reproduced exactly."""
import numpy as np
import pytest

from src.ar import lag_matrix
from src.data import TRUTH_2025_CODE, TRUTH_2025_PDF, load_original
from src.models import SeasonalNaive, TwoYearMean
from src.original import reproduce


@pytest.fixture(scope="module")
def rep():
    return reproduce()


def test_lag_matrix_layout():
    X, target = lag_matrix(np.arange(15), max_lag=12)
    assert X.shape == (3, 12)
    assert list(X[0]) == list(range(11, -1, -1))  # lags 1..12 of y[12]
    assert list(target) == [12, 13, 14]


def test_q12_to_q14_linear_trend(rep):
    assert rep["q12_n"] == 5
    assert rep["q12_adj_r2"] == pytest.approx(0.97446, abs=1e-5)
    assert rep["q12_beta0"] == pytest.approx(58.02, abs=1e-6)
    assert rep["q12_beta1"] == pytest.approx(-4.58, abs=1e-6)
    assert rep["q13_pred_jan2025"] == pytest.approx(3.06, abs=1e-6)
    assert rep["q14_p_beta1"] == pytest.approx(0.0011315, abs=1e-6)


def test_q16_exhaustive_search(rep):
    assert rep["q16_n_combinations"] == 4095
    assert rep["q16_n_train_rows"] == 12
    assert tuple(rep["q16_lags"]) == (1, 2, 5, 7, 8, 9, 10, 11, 12)
    assert rep["q16_adj_r2"] == pytest.approx(0.99967, abs=1e-5)
    assert rep["q16_df_resid"] == 2  # 12 rows - 9 slopes - 1 intercept
    assert np.all(rep["q16_pvalues"][1:] < 0.05)


def test_q17_recursive_forecast(rep):
    np.testing.assert_allclose(rep["q17_pred_2025"], [7.95, 8.65, 11.96, 18.91], atol=0.005)


@pytest.mark.parametrize("truth, expected", [
    (TRUTH_2025_PDF, {"ar": 1.06, "snaive": 1.70, "two_year": 1.91}),   # AR looks best
    (TRUTH_2025_CODE, {"ar": 1.09, "snaive": 0.93, "two_year": 0.81}),  # AR looks worst
])
def test_conclusion_depends_on_truth(rep, truth, expected):
    y = load_original()
    mae = lambda p: np.mean(np.abs(np.asarray(p) - truth))
    assert mae(rep["q17_pred_2025"]) == pytest.approx(expected["ar"], abs=0.005)
    assert mae(SeasonalNaive().forecast(y, 4)) == pytest.approx(expected["snaive"], abs=0.005)
    assert mae(TwoYearMean().forecast(y, 4)) == pytest.approx(expected["two_year"], abs=0.005)
