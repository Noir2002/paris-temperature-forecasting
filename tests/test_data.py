"""Stage-1 findings, checked against the committed Météo-France snapshot."""
import numpy as np

from src.data import TRUTH_2025_CODE, TRUTH_2025_PDF, load_montsouris, load_original


def test_series_is_complete_and_monthly():
    s = load_montsouris()
    assert s.index.freqstr == "MS" and s.notna().all()
    assert s.index[0].strftime("%Y-%m") == "1950-01"


def test_data2_is_montsouris_tx():
    o = load_original()
    np.testing.assert_array_equal(load_montsouris().loc[o.index].to_numpy(), o.to_numpy())


def test_true_2025_values_are_the_assignment_ones():
    jan_apr = load_montsouris().loc["2025-01":"2025-04"].to_numpy()
    np.testing.assert_array_equal(jan_apr, TRUTH_2025_PDF)
    assert not np.allclose(jan_apr, TRUTH_2025_CODE)


def test_normals_1991_2020():
    s = load_montsouris().loc["1991":"2020"]
    normals = s.groupby(s.index.month).mean().loc[[1, 2, 3, 4]].round(1)
    np.testing.assert_array_equal(normals.to_numpy(), [7.6, 8.8, 12.8, 16.6])
