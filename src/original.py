"""Reproduce questions 12-17 of the 2025 project on data2.csv (2023-2024).

Run:  python -m src.original
"""
import numpy as np

from .ar import all_subsets, search_subsets
from .data import load_original
from .models import LagAR, LinearTrend


def reproduce():
    y = load_original()                       # 24 months, 2023-01 .. 2024-12

    # Q12-Q14: straight line through the last n months of 2024, extrapolated
    trend = LinearTrend().fit(y)
    jan_2025 = trend.forecast(y, 1)[0]

    # Q16: lags 1..12 -> only 2024 has all 12 lags available -> 12 training rows
    table = search_subsets(y.to_numpy())
    ar = LagAR("original").fit(y)

    # Q17: recursive forecast Jan-Apr 2025
    preds = ar.forecast(y, 4)

    return {
        "q12_n": trend.n,
        "q12_adj_r2": trend.res.rsquared_adj,
        "q12_r2": trend.res.rsquared,
        "q12_beta0": trend.res.params[0],
        "q12_beta1": trend.res.params[1],
        "q13_pred_jan2025": jan_2025,
        "q14_p_beta1": trend.res.pvalues[1],
        "q16_n_combinations": len(all_subsets(12)),
        "q16_n_fitted": len(table),             # subsets with >= 1 residual dof
        "q16_n_train_rows": int(ar.res.nobs),
        "q16_lags": ar.lags,
        "q16_adj_r2": ar.res.rsquared_adj,
        "q16_df_resid": ar.res.df_resid,
        "q16_params": ar.params,
        "q16_pvalues": ar.res.pvalues,
        "q17_pred_2025": preds,
    }


if __name__ == "__main__":
    np.set_printoptions(precision=4, suppress=True)
    for key, value in reproduce().items():
        print(f"{key:20s} {value}")
