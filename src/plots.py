"""Figures for results/. One colour per entity, the same in every figure:
blue = climatology, orange = the original AR method, aqua = seasonal models,
black = official truth, gray = the values hard-coded in the original code / naive."""
import matplotlib
import matplotlib.ticker

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

BLUE, ORANGE, AQUA = "#2a78d6", "#eb6834", "#1baf7a"
INK, INK2, GRAY, GRID, SURFACE = "#0b0b0b", "#52514e", "#9b9a95", "#e6e5e1", "#fcfcfb"
MONTHS = ["Jan", "Feb", "Mar", "Apr"]

plt.rcParams.update({
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
    "axes.edgecolor": GRID, "axes.labelcolor": INK2, "axes.titlecolor": INK,
    "axes.titlesize": 11, "axes.titleweight": "bold", "axes.titlelocation": "left",
    "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.8,
    "xtick.color": INK2, "ytick.color": INK2, "font.size": 9.5,
    "lines.linewidth": 2, "lines.markersize": 6, "legend.frameon": False,
})


def _plain_log_ticks(ax, ticks):
    ax.yaxis.set_major_locator(matplotlib.ticker.FixedLocator(ticks))
    ax.yaxis.set_major_formatter(matplotlib.ticker.FormatStrFormatter("%g"))
    ax.yaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())


def _label_end(ax, x, y, text, color, dy=0.0):
    ax.annotate(text, (x, y), xytext=(6, dy), textcoords="offset points",
                va="center", fontsize=8.5, color=INK2)


def fig_data(series, original, truth_pdf, truth_code, path):
    fig, (a, b) = plt.subplots(1, 2, figsize=(11, 3.6), gridspec_kw={"width_ratios": [1.6, 1]})
    a.plot(series.index, series, color="#cfcec9", lw=0.7, label="monthly TX")
    a.plot(series.index, series.rolling(12).mean(), color=INK, lw=1.5, label="12-month mean")
    a.set_title("Paris-Montsouris, monthly mean of daily max (°C), 1950–2026")
    a.legend(loc="upper left")
    idx25 = series.loc["2025-01":"2025-04"].index
    b.plot(original.index, original, color=INK, marker="o", label="data2.csv = Montsouris TX")
    b.plot(idx25, truth_pdf, color=INK, marker="o", mfc=SURFACE, mew=1.8, ls="-",
           label="2025 official (= assignment PDF)")
    b.plot(idx25, truth_code, color=GRAY, marker="X", ls="--", label="2025 in original code")
    b.set_title("Course data and Jan–Apr 2025")
    b.set_ylim(5, 37)
    b.legend(loc="upper left", fontsize=8)
    b.tick_params(axis="x", rotation=30)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def fig_expost_2025(expost, truth_pdf, truth_code, path):
    fig, (a, b) = plt.subplots(1, 2, figsize=(11, 3.8))
    x = np.arange(4)
    a.plot(x, truth_pdf, color=INK, marker="o", label="official (Météo-France = PDF)")
    a.plot(x, truth_code, color=GRAY, marker="X", ls="--", label="values in original code")
    ar = expost.loc["ar_original_9lags", MONTHS].to_numpy(float)
    a.plot(x, ar, color=ORANGE, marker="s", label="9-lag AR forecast (original)")
    a.set_xticks(x, MONTHS)
    a.set_ylabel("°C")
    a.set_title("Jan–Apr 2025: two 'truths' and the original forecast")
    a.legend(loc="upper left", fontsize=8)

    part = expost[expost["block"] == "as in the project"].drop("linear_trend_q12")
    y = np.arange(len(part))
    for i, (_, row) in enumerate(part.iterrows()):
        b.plot([row["MAE_official"], row["MAE_code"]], [i, i], color=GRID, lw=3, zorder=1)
    b.scatter(part["MAE_official"], y, color=INK, s=50, zorder=2, label="vs official truth")
    b.scatter(part["MAE_code"], y, facecolor=SURFACE, edgecolor=GRAY, lw=1.8, s=50, zorder=2,
              label="vs values in code")
    b.set_yticks(y, part.index)
    b.set_xlabel("MAE over Jan–Apr 2025 (°C)")
    trend = expost.loc["linear_trend_q12", "MAE_official"]
    b.set_title("The MAE ranking flips with the truth used")
    b.text(0.98, 0.97, f"not shown: Q12 linear trend,\nMAE {trend:.1f} °C (official truth)",
           transform=b.transAxes, ha="right", va="top", fontsize=8, color=INK2)
    b.legend(loc="lower right", fontsize=8)
    b.invert_yaxis()
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def fig_replay(by_year, fits, path):
    fig, ax = plt.subplots(figsize=(11, 3.8))
    ar, clim = by_year["ar_original"], by_year["climatology_30y"]
    ax.semilogy(ar.index, ar, color=ORANGE, marker="o", ls="-", lw=1,
                label="original AR: out-of-sample MAE (Jan–Apr)")
    ax.semilogy(fits["year"], fits["insample_rmse"].clip(lower=1e-3), color=ORANGE, marker="o",
                mfc=SURFACE, ls=":", lw=1, label="original AR: in-sample RMSE (12 rows)")
    ax.semilogy(clim.index, clim, color=BLUE, marker="s", ls="-", lw=1,
                label="30-year climatology: out-of-sample MAE")
    ax.annotate("2025 (the course year)", (2025, ar.loc[2025]), xytext=(-60, -28),
                textcoords="offset points", arrowprops=dict(arrowstyle="-", color=INK2),
                fontsize=8.5, color=INK2)
    ax.set_ylim(top=5e4)
    _plain_log_ticks(ax, [1e-3, 1e-2, 0.1, 1, 10, 100, 1000])
    ax.set_ylabel("°C (log scale)")
    ax.set_title("Original protocol replayed: train on 24 months, forecast Jan–Apr, 1980–2026")
    ax.legend(loc="upper left", fontsize=8, ncol=3)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def fig_rolling(mae, path):
    """mae: DataFrame index=method, columns=h."""
    fig, ax = plt.subplots(figsize=(7.5, 4.2))
    h = mae.columns.to_numpy()
    ar_rules = ["ar_original", "ar_adj_r2", "ar_aic", "ar_bic", "ar_cv"]
    ax.fill_between(h, mae.loc[ar_rules].min(), mae.loc[ar_rules].max(), color=ORANGE,
                    alpha=0.18, lw=0, label="AR, 5 lag-selection rules (range)")
    styles = {
        "seasonal_naive": (GRAY, "--", "D", "seasonal naive"),
        "ar_original": (ORANGE, "-", "o", "AR, original rule"),
        "climatology_30y": (BLUE, "-", "s", "30-year climatology"),
        "sarima": (AQUA, "--", "^", "SARIMA(1,0,0)(0,1,1)12"),
        "anomaly_ar": (AQUA, "-", "o", "climatology + AR anomalies"),
    }
    offsets = {"climatology_30y": 11, "anomaly_ar": 0, "sarima": -11}
    for m, (c, ls, mk, lab) in styles.items():
        ax.plot(h, mae.loc[m], color=c, ls=ls, marker=mk, label=lab)
        _label_end(ax, h[-1], mae.loc[m].iloc[-1], lab, c, dy=offsets.get(m, 0))
    ax.set_xticks(h, [f"h={k}" for k in h])
    ax.set_xlim(h[0] - 0.2, h[-1] + 1.9)
    ax.set_ylabel("MAE (°C)")
    ax.set_title("Rolling-origin test 2001–2026, 30-year training window")
    ax.legend(loc="upper left", fontsize=7.5, ncol=2)
    ax.set_ylim(top=ax.get_ylim()[1] + 0.12)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def fig_window_sweep(sweep, path):
    """sweep: rows (years, method) with MAE, RMSE, insample_rmse."""
    fig, (a, b) = plt.subplots(1, 2, figsize=(11, 3.8))
    ar = sweep.xs("ar_original", level="method")
    a.semilogy(ar.index, ar["insample_rmse"], color=ORANGE, marker="o", mfc=SURFACE, ls=":",
               label="in-sample RMSE")
    a.semilogy(ar.index, ar["RMSE_h1"], color=ORANGE, marker="o", label="out-of-sample RMSE (h=1)")
    a.set_xticks(ar.index)
    _plain_log_ticks(a, [0.2, 0.5, 1, 2, 5, 10, 20, 50])
    a.set_xlabel("training window (years)")
    a.set_ylabel("°C (log scale)")
    a.set_title("Original AR rule: 1-step error, in vs out of sample")
    a.legend(fontsize=8)
    styles = {"ar_original": (ORANGE, "-", "o", "AR, original rule"),
              "ar_bic": (GRAY, "--", "D", "AR, BIC"),
              "climatology": (BLUE, "-", "s", "climatology (same window)"),
              "anomaly_ar": (AQUA, "-", "^", "climatology + AR anomalies")}
    for m, (c, ls, mk, lab) in styles.items():
        s = sweep.xs(m, level="method")["MAE"]
        b.semilogy(s.index, s, color=c, ls=ls, marker=mk, label=lab)
    b.set_xticks(ar.index)
    _plain_log_ticks(b, [1, 1.5, 2, 3, 5, 10, 15])
    b.set_xlabel("training window (years)")
    b.set_ylabel("out-of-sample MAE, h=1–4 (°C, log)")
    b.set_title("Test 2001–2026, varying training length")
    b.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
