"""Autoregression on lagged values, fitted by OLS.

    y[t] = b0 + sum_{k in lags} b_k * y[t-k] + e[t]

`lags` is a tuple of 1-based lags, e.g. (1, 2, 12). Everything in this file
only ever sees the training array it is given, so it cannot look ahead.
"""
import itertools

import numpy as np
import pandas as pd
import statsmodels.api as sm


def lag_matrix(y, max_lag=12):
    """X[:, k-1] = y[t-k] for k = 1..max_lag, target = y[t]; rows t = max_lag..end."""
    y = np.asarray(y, dtype=float)
    X = np.column_stack([y[max_lag - k: len(y) - k] for k in range(1, max_lag + 1)])
    return X, y[max_lag:]


def all_subsets(max_lag=12):
    """Every non-empty subset of {1..max_lag}: 2**12 - 1 = 4095 for max_lag = 12."""
    return [c for k in range(1, max_lag + 1)
            for c in itertools.combinations(range(1, max_lag + 1), k)]


def fit_ols(X, target, lags):
    """OLS with intercept on the chosen lag columns (statsmodels gives R², AIC, p-values)."""
    cols = [k - 1 for k in lags]
    return sm.OLS(target, sm.add_constant(X[:, cols], has_constant="add")).fit()


_search_cache = {}  # same training array -> same table; avoids refitting 4095 models


def search_subsets(y, max_lag=12):
    """Fit every lag subset on the same rows; return one row of statistics per subset."""
    y = np.asarray(y, dtype=float)
    key = (y.tobytes(), max_lag)
    if key not in _search_cache:
        X, target = lag_matrix(y, max_lag)
        rows = []
        for lags in all_subsets(max_lag):
            if len(target) <= len(lags) + 1:  # need >= 1 residual dof (original rule)
                continue
            r = fit_ols(X, target, lags)
            rows.append(dict(lags=lags, n_lags=len(lags), adj_r2=r.rsquared_adj,
                             aic=r.aic, bic=r.bic, max_p=r.pvalues[1:].max()))
        _search_cache[key] = pd.DataFrame(rows)
    return _search_cache[key]


def select_lags(table, criterion):
    """Pick a subset from `search_subsets` output.

    "original": max adjusted R² among subsets whose slopes all have p < 0.05
                (exactly the rule of the 2025 project; ties -> first, like its loop)
    "adj_r2":   max adjusted R², no p-value filter
    "aic"/"bic": min information criterion
    """
    if criterion == "original":
        ok = table[table["max_p"] < 0.05]
        if ok.empty:
            return None
        return ok.loc[ok["adj_r2"].idxmax(), "lags"]
    if criterion == "adj_r2":
        return table.loc[table["adj_r2"].idxmax(), "lags"]
    if criterion in ("aic", "bic"):
        return table.loc[table[criterion].idxmin(), "lags"]
    raise ValueError(criterion)


def cv_select_lags(y, max_lag=12, n_folds=5, fold_len=12):
    """Time-series cross-validation over all subsets.

    The last `n_folds` blocks of `fold_len` rows are validation blocks. For each
    block: fit on all rows *before* it, score one-step-ahead MSE on it
    (expanding window, never shuffled). Return the subset with the lowest mean MSE.
    """
    X, target = lag_matrix(y, max_lag)
    X1 = np.column_stack([np.ones(len(target)), X])  # column k <-> lag k
    n = len(target)
    best, best_mse = None, np.inf
    for lags in all_subsets(max_lag):
        cols = [0, *lags]
        mses = []
        for f in range(n_folds, 0, -1):
            split = n - f * fold_len
            beta = np.linalg.lstsq(X1[:split, cols], target[:split], rcond=None)[0]
            val = slice(split, split + fold_len)
            mses.append(np.mean((target[val] - X1[val][:, cols] @ beta) ** 2))
        if np.mean(mses) < best_mse:
            best, best_mse = lags, np.mean(mses)
    return best


def recursive_forecast(history, lags, params, steps):
    """Multi-step forecast: each prediction is fed back as an input for the next step."""
    h = list(np.asarray(history, dtype=float))
    out = []
    for _ in range(steps):
        x = np.r_[1.0, [h[-k] for k in lags]]
        out.append(float(x @ np.asarray(params)))
        h.append(out[-1])
    return np.array(out)
