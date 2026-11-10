"""Run every experiment and write all tables/figures to results/.

    python -m src.run_all          (~2 minutes, no network: uses the committed data snapshot)

1. reproduce the 2025 project (Q12-Q17) on data2.csv
2. 2025 ex-post check with the verified truth (and with the code's values)
3. replay the original protocol for every year 1980-2026
4. rolling-origin backtest 2001-2026, 30-year window, horizons 1-4
5. same backtest with 2/5/10/20/30-year windows (does more data stop overfitting?)
"""
import numpy as np
import pandas as pd

from . import plots
from .data import (ROOT, TRUTH_2025_CODE, TRUTH_2025_PDF, load_montsouris,
                   load_original)
from .evaluate import (compare_to, metrics, replay_original_protocol,
                       rolling_origin)
from .models import (AnomalyAR, Climatology, LagAR, LinearTrend, Sarima,
                     SeasonalNaive, TwoYearMean)
from .original import reproduce

RESULTS = ROOT / "results"
FIRST_ORIGIN, LAST_ORIGIN = "2000-12-01", "2026-09-01"  # targets 2001-01 .. 2026-10

MAIN_MODELS = {
    "seasonal_naive": SeasonalNaive,
    "climatology_30y": Climatology,
    "ar_original": lambda: LagAR("original"),
    "ar_adj_r2": lambda: LagAR("adj_r2"),
    "ar_aic": lambda: LagAR("aic"),
    "ar_bic": lambda: LagAR("bic"),
    "ar_cv": lambda: LagAR("cv"),
    "anomaly_ar": AnomalyAR,
    "sarima": Sarima,
}
SWEEP_MODELS = {
    "climatology": Climatology,
    "ar_original": lambda: LagAR("original"),
    "ar_bic": lambda: LagAR("bic"),
    "anomaly_ar": AnomalyAR,
}


def md_table(df, digits=2):
    """DataFrame -> GitHub markdown table (no extra dependency)."""
    df = df.reset_index()
    fmt = lambda v: f"{v:.{digits}f}" if isinstance(v, (float, np.floating)) else str(v)
    lines = ["| " + " | ".join(map(str, df.columns)) + " |",
             "|" + "---|" * len(df.columns)]
    lines += ["| " + " | ".join(fmt(v) for v in row) + " |" for row in df.itertuples(index=False)]
    return "\n".join(lines)


def errors(pred, truth):
    e = np.asarray(pred, float) - np.asarray(truth, float)
    return np.abs(e).mean(), np.sqrt((e ** 2).mean())


def expost_2025(series, original, ar_preds, rolling_fc):
    """Jan-Apr 2025 forecasts vs the official truth and vs the code's values."""
    normals = series["1991":"2020"].groupby(series["1991":"2020"].index.month).mean()
    preds = {
        "ar_original_9lags": ar_preds,
        "linear_trend_q12": LinearTrend().fit(original).forecast(original, 4),
        "seasonal_naive": SeasonalNaive().forecast(original, 4),
        "two_year_mean": TwoYearMean().forecast(original, 4),
        "normals_1991_2020": normals.loc[[1, 2, 3, 4]].to_numpy(),
    }
    rows = [dict(method=k, block="as in the project", **dict(zip(plots.MONTHS, v)))
            for k, v in preds.items()]
    at_2024 = rolling_fc[(rolling_fc["origin"] == "2024-12-01")
                         & (rolling_fc["method"] != "seasonal_naive")]  # already above
    for m, g in at_2024.groupby("method", sort=False):
        rows.append(dict(method=m, block="backtest model, trained 1995-2024",
                         **dict(zip(plots.MONTHS, g.sort_values("h")["y_pred"]))))
    table = pd.DataFrame(rows).set_index("method")
    p = table[plots.MONTHS].to_numpy(float)
    for name, truth in [("official", TRUTH_2025_PDF), ("code", TRUTH_2025_CODE)]:
        table[f"MAE_{name}"], table[f"RMSE_{name}"] = zip(*(errors(r, truth) for r in p))
    return table


def main():
    RESULTS.mkdir(exist_ok=True)
    series, original = load_montsouris(), load_original()
    out = {}

    print("1/5 reproduce original project")
    rep = reproduce()
    def plain(v):  # numpy arrays -> short python lists for the CSV
        if not np.ndim(v):
            return v
        return [round(float(x), 6) if isinstance(x, (float, np.floating)) else int(x) for x in v]
    pd.Series({k: plain(v) for k, v in rep.items()}, name="value") \
        .to_csv(RESULTS / "01_original_reproduction.csv")

    print("2/5 rolling-origin backtest, 30-year window (slowest step)")
    fc, fit_log = rolling_origin(series, MAIN_MODELS, FIRST_ORIGIN, LAST_ORIGIN, window=360)
    fc.to_csv(RESULTS / "04_rolling_forecasts.csv.gz", index=False,
              compression={"method": "gzip", "mtime": 0})  # byte-identical reruns
    fit_log.to_csv(RESULTS / "04_rolling_fit_log.csv", index=False)
    roll = metrics(fc, by=("method", "h"))
    roll.to_csv(RESULTS / "04_rolling_metrics.csv")
    vs_clim = compare_to(fc, "climatology_30y")
    vs_clim.to_csv(RESULTS / "04_rolling_vs_climatology.csv", index=False)
    lag_summary = fit_log.groupby("method").agg(mean_n_lags=("n_lags", "mean"),
                                                mean_insample_rmse=("insample_rmse", "mean"),
                                                max_root_max=("max_root", "max"))
    lag_summary["oos_rmse_h1"] = roll.xs(1, level="h")["RMSE"]
    lag_summary["lags_at_2024_12"] = fit_log[fit_log["fit_date"] == "2024-12-01"] \
        .set_index("method")["lags"].astype(str)
    lag_summary.to_csv(RESULTS / "04_rolling_lag_selection.csv")

    print("3/5 2025 ex-post")
    expost = expost_2025(series, original, rep["q17_pred_2025"], fc)
    expost.to_csv(RESULTS / "02_expost_2025.csv")

    print("4/5 replay of the original protocol, 1980-2026")
    rp, rp_fits = replay_original_protocol(series, range(1980, 2027))
    rp.to_csv(RESULTS / "03_replay_forecasts.csv", index=False)
    rp_fits.to_csv(RESULTS / "03_replay_ar_fits.csv", index=False)
    rp_metrics = metrics(rp)
    by_year = rp.assign(ae=(rp.y_pred - rp.y_true).abs()) \
        .groupby(["year", "method"])["ae"].mean().unstack()
    for m in by_year.columns.drop("ar_original"):
        rp_metrics.loc[m, "years_AR_better"] = int((by_year["ar_original"] < by_year[m]).sum())
    rp_metrics.to_csv(RESULTS / "03_replay_metrics.csv")
    ar_rows = rp[rp["method"] == "ar_original"]
    worst = ar_rows.loc[ar_rows["y_pred"].abs().idxmax()]

    print("5/5 training-window sweep")
    sweep = []
    for years in [2, 5, 10, 20, 30]:
        f, log = rolling_origin(series, SWEEP_MODELS, FIRST_ORIGIN, LAST_ORIGIN, window=12 * years)
        m = metrics(f).join(log.groupby("method")[["insample_rmse", "n_lags"]].mean())
        m["RMSE_h1"] = metrics(f[f["h"] == 1])["RMSE"]  # 1-step, comparable to in-sample
        sweep.append(m.assign(years=years).set_index("years", append=True))
    sweep = pd.concat(sweep).reorder_levels(["years", "method"]).sort_index()
    sweep.to_csv(RESULTS / "05_window_sweep.csv")

    print("figures")
    plots.fig_data(series, original, TRUTH_2025_PDF, TRUTH_2025_CODE, RESULTS / "fig1_data.png")
    plots.fig_expost_2025(expost, TRUTH_2025_PDF, TRUTH_2025_CODE, RESULTS / "fig2_expost_2025.png")
    plots.fig_replay(by_year, rp_fits, RESULTS / "fig3_replay_original_protocol.png")
    plots.fig_rolling(roll["MAE"].unstack(), RESULTS / "fig4_rolling_mae_by_horizon.png")
    plots.fig_window_sweep(sweep, RESULTS / "fig5_window_sweep.png")

    summary = [
        "# Results summary (generated by `python -m src.run_all`, do not edit)\n",
        "## 1. Reproduction of the 2025 project (data2.csv)\n",
        md_table(pd.read_csv(RESULTS / "01_original_reproduction.csv", index_col=0), 4),
        "\n## 2. Jan-Apr 2025 ex-post (truth: official = PDF; code = 8/10/14/18)\n",
        md_table(expost.round(2)),
        "\n## 3. Original protocol replayed 1980-2026 (Jan-Apr of each year)\n",
        md_table(rp_metrics.round(2)),
        f"\nLargest |AR forecast| in the replay: {worst['y_pred']:.1f} °C "
        f"(target {worst['target']:%Y-%m}, truth {worst['y_true']:.1f} °C).\n",
        "\nAR fits on 12 training rows, one per replayed year:\n",
        md_table(rp_fits[["n_lags", "adj_r2", "insample_rmse", "max_root"]].describe().round(3), 3),
        f"\nExplosive fits (max |root| > 1): {(rp_fits['max_root'] > 1).sum()} of {len(rp_fits)}; "
        f"AR better than 30-year climatology in years: "
        f"{', '.join(map(str, by_year.index[by_year['ar_original'] < by_year['climatology_30y']]))}.\n",
        "\n## 4. Rolling-origin backtest 2001-2026, 30-year window\n",
        "### MAE by horizon\n", md_table(roll["MAE"].unstack().round(3), 3),
        "\n### RMSE by horizon\n", md_table(roll["RMSE"].unstack().round(3), 3),
        "\n### Bias (forecast - truth) by horizon\n", md_table(roll["bias"].unstack().round(3), 3),
        "\n### MAE difference vs 30-year climatology (negative = better), 95% block-bootstrap CI\n",
        md_table(vs_clim.set_index(["h", "method"]).drop(columns="reference").round(3), 3),
        "\n### Lag selection (averaged over the 26 yearly refits)\n",
        md_table(lag_summary.round(3), 3),
        "\n## 5. Training-window sweep (same test period)\n",
        md_table(sweep.round(3), 3),
    ]
    (RESULTS / "summary.md").write_text("\n".join(summary) + "\n")
    print(f"done -> {RESULTS.relative_to(ROOT)}/")


if __name__ == "__main__":
    main()
