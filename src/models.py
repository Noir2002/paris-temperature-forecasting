"""Forecasting methods, all with the same two-method interface:

    model.fit(train)              -> learn parameters from a monthly pd.Series
    model.forecast(history, steps) -> np.array of the next `steps` months

`history` = every observation up to the forecast origin (never later). `fit`
sees only `train` (the last `window` months of history at refit time).
"""
import warnings

import numpy as np
import statsmodels.api as sm

from .ar import (cv_select_lags, fit_ols, lag_matrix, recursive_forecast,
                 search_subsets, select_lags)


def future_months(history, steps):
    """Calendar months (1..12) of the `steps` months after the last observation."""
    return (history.index[-1].month + np.arange(steps)) % 12 + 1


class SeasonalNaive:
    """y_hat[t+h] = y[t+h-12]: 'same month last year'."""
    def fit(self, train):
        return self

    def forecast(self, history, steps):
        return history.to_numpy()[-12:][:steps]


class TwoYearMean:
    """y_hat[t+h] = mean(y[t+h-12], y[t+h-24])."""
    def fit(self, train):
        return self

    def forecast(self, history, steps):
        y = history.to_numpy()
        return (y[-12:][:steps] + y[-24:-12][:steps]) / 2


class Climatology:
    """y_hat = mean of the same calendar month over the training window
    (30-year window = a rolling 'climate normal')."""
    def fit(self, train):
        self.means = train.groupby(train.index.month).mean()
        return self

    def forecast(self, history, steps):
        return self.means.loc[future_months(history, steps)].to_numpy()


class LagAR:
    """OLS autoregression on a subset of lags 1..12, recursive multi-step forecast.
    criterion: "original" | "adj_r2" | "aic" | "bic" | "cv" (see ar.py)."""
    def __init__(self, criterion="original", max_lag=12):
        self.criterion, self.max_lag = criterion, max_lag

    def fit(self, train):
        y = train.to_numpy()
        if self.criterion == "cv":
            self.lags = cv_select_lags(y, self.max_lag)
        else:
            self.lags = select_lags(search_subsets(y, self.max_lag), self.criterion)
        if self.lags is None:  # no subset passes the p < 0.05 filter
            raise ValueError("no admissible lag subset")
        X, target = lag_matrix(y, self.max_lag)
        self.res = fit_ols(X, target, self.lags)
        self.params = self.res.params
        self.insample_rmse = float(np.sqrt(np.mean(self.res.resid ** 2)))
        return self

    def forecast(self, history, steps):
        return recursive_forecast(history.to_numpy(), self.lags, self.params, steps)


class AnomalyAR:
    """Seasonal mean + AR(p) on the anomalies (y - monthly mean).
    p in 0..12 (contiguous lags 1..p) chosen by AIC on the training window."""
    def fit(self, train, max_p=12):
        self.clim = Climatology().fit(train)
        anom = train.to_numpy() - self.clim.means.loc[train.index.month].to_numpy()
        X, target = lag_matrix(anom, max_p)
        p_max = min(max_p, len(target) - 2)  # keep >= 1 residual dof on short windows
        fits = {p: fit_ols(X, target, tuple(range(1, p + 1))) for p in range(p_max + 1)}
        p = min(fits, key=lambda q: fits[q].aic)
        self.lags, self.params = tuple(range(1, p + 1)), fits[p].params
        self.insample_rmse = float(np.sqrt(np.mean(fits[p].resid ** 2)))
        return self

    def forecast(self, history, steps):
        anom = history.to_numpy() - self.clim.means.loc[history.index.month].to_numpy()
        future_anom = recursive_forecast(anom, self.lags, self.params, steps)
        return future_anom + self.clim.forecast(history, steps)


class Sarima:
    """SARIMA(1,0,0)(0,1,1)_12, specification fixed in advance (not tuned on test data).
    Seasonal difference + seasonal MA ~ an adaptive seasonal mean; AR(1) ~ anomaly persistence."""
    def fit(self, train):
        model = sm.tsa.SARIMAX(train.to_numpy(), order=(1, 0, 0), seasonal_order=(0, 1, 1, 12))
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            self.res = model.fit(disp=False)
        return self

    def forecast(self, history, steps):
        # same fitted parameters, Kalman filter re-run on the history up to the origin
        return np.asarray(self.res.apply(history.to_numpy()).forecast(steps))


def trend_by_last_n(year_values):
    """Q12 of the project: T = b0 + b1 * month_number (0..11) on the last n months
    of one year, n chosen by max adjusted R². (n = 1, 2 give no adjusted R², so 3..12.)"""
    y, x = np.asarray(year_values, dtype=float), np.arange(12)
    fits = {n: sm.OLS(y[-n:], sm.add_constant(x[-n:])).fit() for n in range(3, 13)}
    n_best = max(fits, key=lambda n: fits[n].rsquared_adj)
    return n_best, fits[n_best]


class LinearTrend:
    """Q12 model used as a forecaster: extrapolate the line to month numbers 12, 13, ...
    Only meaningful when the last training month is a December (as in the project)."""
    def fit(self, train):
        self.n, self.res = trend_by_last_n(train.to_numpy()[-12:])
        return self

    def forecast(self, history, steps):
        return self.res.params[0] + self.res.params[1] * np.arange(12, 12 + steps)
