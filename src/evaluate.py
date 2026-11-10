"""Evaluation: error metrics, rolling-origin backtest, replay of the 2025 protocol."""
import numpy as np
import pandas as pd

from .models import (Climatology, LagAR, LinearTrend, SeasonalNaive,
                     TwoYearMean)

MONTH = pd.DateOffset(months=1)


def metrics(df, by=("method",)):
    """MAE, RMSE, median absolute error and bias (mean of forecast - truth) per group."""
    err = df["y_pred"] - df["y_true"]
    return err.groupby([df[c] for c in by]).agg(
        MAE=lambda e: e.abs().mean(),
        RMSE=lambda e: np.sqrt((e ** 2).mean()),
        MedAE=lambda e: e.abs().median(),
        bias=lambda e: e.mean(),
        n="size")


def rolling_origin(series, models, first_origin, last_origin, window=360, steps=4,
                   refit_month=12):
    """Rolling-origin backtest.

    For every month-end origin t in [first_origin, last_origin]:
      * history = series up to and including t (nothing later),
      * models are re-fitted on the last `window` months of history once a year
        (when t is a December) and at the first origin,
      * each model forecasts t+1 .. t+steps; we store (origin, h, truth, forecast).
    Returns (forecasts DataFrame, fit-log DataFrame).
    """
    rows, fit_log, fitted = [], [], {}
    for origin in pd.date_range(first_origin, last_origin, freq="MS"):
        history = series.loc[:origin]
        if not fitted or origin.month == refit_month:
            train = history.iloc[-window:]
            fitted = {name: make().fit(train) for name, make in models.items()}
            for name, m in fitted.items():
                if hasattr(m, "lags"):
                    fit_log.append(dict(fit_date=origin, method=name, n_lags=len(m.lags),
                                        lags=m.lags, insample_rmse=m.insample_rmse))
        targets = series.loc[origin + MONTH: origin + steps * MONTH]
        for name, m in fitted.items():
            pred = m.forecast(history, steps)
            for h, (date, truth) in enumerate(targets.items(), start=1):
                rows.append(dict(method=name, origin=origin, target=date, h=h,
                                 y_true=truth, y_pred=pred[h - 1]))
    return pd.DataFrame(rows), pd.DataFrame(fit_log)


def replay_original_protocol(series, years, normals_window=30):
    """Repeat the 2025 project for many years Y: train on the 24 months of Y-2 and
    Y-1 (like data2.csv), forecast Jan-Apr of Y, compare with simple baselines."""
    rows, fits = [], []
    for year in years:
        train = series.loc[f"{year - 2}-01": f"{year - 1}-12"]
        truth = series.loc[f"{year}-01": f"{year}-04"]
        past = series.loc[: f"{year - 1}-12"]
        methods = {
            "ar_original": LagAR("original"),
            "linear_trend_q12": LinearTrend(),
            "seasonal_naive": SeasonalNaive(),
            "two_year_mean": TwoYearMean(),
            "climatology_30y": Climatology(),
        }
        for name, m in methods.items():
            try:
                m.fit(past.iloc[-12 * normals_window:] if name == "climatology_30y" else train)
            except ValueError:  # no lag subset with all p < 0.05
                continue
            pred = m.forecast(train, 4)
            rows += [dict(method=name, year=year, target=d, h=h, y_true=t, y_pred=p)
                     for h, (d, t, p) in enumerate(zip(truth.index, truth, pred), start=1)]
            if name == "ar_original":
                fits.append(dict(year=year, lags=m.lags, n_lags=len(m.lags),
                                 adj_r2=m.res.rsquared_adj, insample_rmse=m.insample_rmse))
    return pd.DataFrame(rows), pd.DataFrame(fits)


def block_bootstrap_ci(d, block=12, n_boot=2000, seed=0):
    """95% CI of mean(d) with a moving-block bootstrap (keeps autocorrelation)."""
    d = np.asarray(d, dtype=float)
    rng = np.random.default_rng(seed)
    n_blocks = int(np.ceil(len(d) / block))
    starts = rng.integers(0, len(d) - block + 1, size=(n_boot, n_blocks))
    idx = (starts[:, :, None] + np.arange(block)).reshape(n_boot, -1)[:, : len(d)]
    return np.percentile(d[idx].mean(axis=1), [2.5, 97.5])


def compare_to(forecasts, reference):
    """Per method and horizon: MAE(method) - MAE(reference), with a block-bootstrap CI.
    Negative = method beats the reference."""
    wide = forecasts.pivot_table(index=["origin", "h"], columns="method", values="y_pred")
    truth = forecasts.groupby(["origin", "h"])["y_true"].first()
    abs_err = wide.sub(truth, axis=0).abs()
    rows = []
    for h, part in abs_err.groupby(level="h"):
        for method in part.columns.drop(reference):
            d = (part[method] - part[reference]).to_numpy()
            lo, hi = block_bootstrap_ci(d)
            rows.append(dict(h=h, method=method, reference=reference,
                             mae_diff=d.mean(), ci_low=lo, ci_high=hi))
    return pd.DataFrame(rows)
