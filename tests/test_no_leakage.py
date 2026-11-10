"""Evaluation must never use data from after the forecast origin.

Idea: corrupt every value after a cutoff (+100 °C). Any forecast made at an
origin <= cutoff must be bit-for-bit identical; if it changed, the model peeked.
"""
import numpy as np
import pandas as pd
import pytest

from src.data import load_montsouris
from src.evaluate import replay_original_protocol, rolling_origin
from src.run_all import MAIN_MODELS

CUTOFF = pd.Timestamp("2017-03-01")


@pytest.fixture(scope="module")
def runs():
    clean = load_montsouris()
    dirty = clean.copy()
    dirty[dirty.index > CUTOFF] += 100.0
    kw = dict(models=MAIN_MODELS, first_origin="2015-12-01", last_origin="2018-06-01", window=120)
    return rolling_origin(clean, **kw)[0], rolling_origin(dirty, **kw)[0]


def test_targets_are_after_origin(runs):
    fc = runs[0]
    assert (fc["target"] > fc["origin"]).all()
    expected = fc.apply(lambda r: r["origin"] + pd.DateOffset(months=r["h"]), axis=1)
    assert (fc["target"] == expected).all()


def test_forecasts_before_cutoff_unchanged(runs):
    clean, dirty = runs
    before = clean["origin"] <= CUTOFF
    assert set(clean.loc[before, "method"]) == set(MAIN_MODELS)
    np.testing.assert_array_equal(clean.loc[before, "y_pred"], dirty.loc[before, "y_pred"])


def test_corruption_is_visible_after_cutoff(runs):
    """Sanity check that the test has power: later forecasts do change."""
    clean, dirty = runs
    after = clean["origin"] > CUTOFF
    for method in MAIN_MODELS:
        m = after & (clean["method"] == method)
        assert not np.allclose(clean.loc[m, "y_pred"], dirty.loc[m, "y_pred"]), method


def test_replay_uses_only_previous_two_years():
    clean = load_montsouris()
    dirty = clean.copy()
    dirty[dirty.index >= "2025-01-01"] += 100.0
    a, _ = replay_original_protocol(clean, [2025])
    b, _ = replay_original_protocol(dirty, [2025])
    np.testing.assert_array_equal(a["y_pred"], b["y_pred"])
    # and replaying year 2025 on the official series gives the project's numbers
    ar = a[a["method"] == "ar_original"]["y_pred"].to_numpy()
    np.testing.assert_allclose(ar, [7.95, 8.65, 11.96, 18.91], atol=0.005)
