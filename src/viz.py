"""Static charts for Crashscape (matplotlib/seaborn → ``reports/figures``).

All figures share one quiet style: hairline grid, no top/right spines, thin
marks, and a fixed, colour-blind-validated palette:

* single series → ``BLUE`` (categorical slot 1); a second series → ``ORANGE``
* Severity 1–4 (ordered) → a light→dark blue ramp
* magnitude heatmaps → a single-hue sequential blue ramp
* never two y-axes; count and rate get separate panels.

Usage::

    python -m src.viz        # compute tables on the full clean data + draw every chart
"""

from __future__ import annotations

from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.ticker import FuncFormatter, PercentFormatter

from src import analysis, config

# --------------------------------------------------------------------------- #
# Palette & style
# --------------------------------------------------------------------------- #
SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK_2 = "#52514e"
MUTED = "#898781"
GRID = "#e1e0d9"
AXIS = "#c3c2b7"
NEUTRAL = "#c3c2b7"  # de-emphasised marks

BLUE = "#2a78d6"     # categorical slot 1
ORANGE = "#eb6834"   # categorical slot 2

BLUE_RAMP = ["#cde2fb", "#b7d3f6", "#9ec5f4", "#86b6ef", "#6da7ec", "#5598e7",
             "#3987e5", "#2a78d6", "#256abf", "#1c5cab", "#184f95", "#104281", "#0d366b"]
SEVERITY_COLORS = {1: "#86b6ef", 2: "#3987e5", 3: "#1c5cab", 4: "#0d366b"}
SEQ_CMAP = LinearSegmentedColormap.from_list("crash_blues", BLUE_RAMP)


def set_style() -> None:
    """Apply the shared matplotlib style."""
    mpl.rcParams.update({
        "figure.facecolor": SURFACE,
        "axes.facecolor": SURFACE,
        "savefig.facecolor": SURFACE,
        "font.family": ["Helvetica Neue", "Arial", "DejaVu Sans"],
        "font.size": 10,
        "text.color": INK,
        "axes.labelcolor": INK_2,
        "axes.titlesize": 13,
        "axes.titleweight": "bold",
        "axes.titlelocation": "left",
        "axes.titlepad": 22,
        "axes.edgecolor": AXIS,
        "axes.linewidth": 0.8,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": True,
        "axes.axisbelow": True,
        "grid.color": GRID,
        "grid.linewidth": 0.8,
        "grid.linestyle": "-",
        "xtick.color": MUTED,
        "ytick.color": MUTED,
        "xtick.labelcolor": INK_2,
        "ytick.labelcolor": INK_2,
        "xtick.major.size": 0,
        "ytick.major.size": 0,
        "legend.frameon": False,
        "legend.fontsize": 9,
        "lines.linewidth": 2,
        "lines.solid_capstyle": "round",
        "lines.solid_joinstyle": "round",
    })


def compact(x: float, _pos: int | None = None) -> str:
    """Compact number formatter: 1,200 → 1.2K, 3,400,000 → 3.4M."""
    for div, suffix in ((1e6, "M"), (1e3, "K")):
        if abs(x) >= div:
            v = x / div
            return f"{v:.0f}{suffix}" if v >= 10 or v == int(v) else f"{v:.1f}{suffix}"
    return f"{x:.0f}"


COMPACT = FuncFormatter(compact)


def _subtitle(ax: plt.Axes, text: str) -> None:
    """Add a muted one-line subtitle under the axes title.

    Title and subtitle are excluded from ``tight_layout`` so a long headline on
    the first panel doesn't squeeze its neighbours; ``savefig(bbox_inches=
    "tight")`` still keeps them inside the image.
    """
    sub = ax.annotate(text, xy=(0, 1), xycoords="axes fraction", xytext=(0, 5),
                      textcoords="offset points", fontsize=9, color=INK_2, va="bottom")
    sub.set_in_layout(False)
    ax.title.set_in_layout(False)


def _source(fig: plt.Figure, extra: str = "") -> None:
    """Footer with the data source."""
    note = "Source: US Accidents (2016–Mar 2023), Moosavi et al."
    fig.text(0.01, 0.005, note + (f" · {extra}" if extra else ""), fontsize=7.5,
             color=MUTED, ha="left", va="bottom")


def save(fig: plt.Figure, name: str) -> Path:
    """Save ``fig`` to ``reports/figures/<name>.png`` and close it."""
    config.ensure_dirs()
    path = config.FIGURES_DIR / f"{name}.png"
    fig.savefig(path, dpi=150, bbox_inches="tight", pad_inches=0.25)
    plt.close(fig)
    return path


def _hbar(ax: plt.Axes, labels, values, color=BLUE, fmt=compact, highlight=None) -> None:
    """Horizontal bars (largest on top) with a value label at each tip."""
    labels, values = list(labels)[::-1], np.asarray(values)[::-1]
    colors = [color if highlight is None or highlight(l) else NEUTRAL for l in labels]
    bars = ax.barh(labels, values, height=0.62, color=colors)
    ax.grid(axis="y", visible=False)
    ax.xaxis.set_major_formatter(FuncFormatter(lambda x, p: fmt(x)))
    pad = values.max() * 0.01
    for b, v in zip(bars, values):
        ax.text(b.get_width() + pad, b.get_y() + b.get_height() / 2, fmt(v),
                va="center", ha="left", fontsize=8.5, color=INK_2)
    ax.set_xlim(0, values.max() * 1.14)
    ax.spines["left"].set_visible(False)


# --------------------------------------------------------------------------- #
# Time
# --------------------------------------------------------------------------- #
def plot_hourly(t: pd.DataFrame) -> Path:
    """Accident share by hour of day, weekday vs weekend."""
    fig, ax = plt.subplots(figsize=(10, 4.8))
    for part, color in (("Weekday", BLUE), ("Weekend", ORANGE)):
        s = t[t["week_part"] == part].sort_values("hour")
        ax.plot(s["hour"], s["share_pct"], color=color, label=part)
        ax.scatter(s["hour"].iloc[-1], s["share_pct"].iloc[-1], s=40, color=color,
                   edgecolor=SURFACE, linewidth=2, zorder=3)
    wk = t[t["week_part"] == "Weekday"].set_index("hour")["share_pct"]
    for h in (wk.loc[5:10].idxmax(), wk.loc[14:19].idxmax()):
        ax.annotate(f"{h}:00 · {wk[h]:.1f}%", xy=(h, wk[h]), xytext=(0, 8),
                    textcoords="offset points", ha="center", fontsize=8.5, color=INK_2)
    ax.set_xticks(range(0, 24, 2), [f"{h}:00" for h in range(0, 24, 2)])
    ax.yaxis.set_major_formatter(PercentFormatter(decimals=0))
    ax.set_xlim(-0.5, 23.5)
    ax.set_ylim(0, t["share_pct"].max() * 1.12)
    ax.set_ylabel("Share of week-part's accidents")
    ax.set_title("Weekdays peak at the commute; weekends peak in the afternoon")
    _subtitle(ax, "Share of accidents by local hour (each line sums to 100%)")
    ax.legend(loc="upper left")
    _source(fig)
    return save(fig, "01_hourly_weekday_vs_weekend")


def plot_day_of_week(t: pd.DataFrame) -> Path:
    """Accident count by day of week (weekend de-emphasised)."""
    fig, ax = plt.subplots(figsize=(9, 4.5))
    colors = [BLUE if d not in ("Saturday", "Sunday") else NEUTRAL for d in t["day_of_week"]]
    bars = ax.bar(t["day_of_week"].astype(str), t["accidents"], width=0.6, color=colors)
    for b, v in zip(bars, t["accidents"]):
        ax.text(b.get_x() + b.get_width() / 2, v, compact(v), ha="center", va="bottom",
                fontsize=8.5, color=INK_2)
    ax.yaxis.set_major_formatter(COMPACT)
    ax.grid(axis="x", visible=False)
    ax.set_title("Weekend days see about half the accidents of a weekday")
    _subtitle(ax, "Total accidents by day of week, 2016–Mar 2023")
    _source(fig)
    return save(fig, "02_day_of_week")


def plot_month(t: pd.DataFrame) -> Path:
    """Accident count by calendar month (full years only)."""
    fig, ax = plt.subplots(figsize=(9, 4.5))
    bars = ax.bar(t["month_name"].astype(str), t["accidents"], width=0.6, color=BLUE)
    top = t["accidents"].idxmax()
    ax.text(bars[top].get_x() + bars[top].get_width() / 2, t["accidents"][top],
            compact(t["accidents"][top]), ha="center", va="bottom", fontsize=8.5, color=INK_2)
    ax.yaxis.set_major_formatter(COMPACT)
    ax.grid(axis="x", visible=False)
    ax.set_title("Accidents climb through autumn and peak in December")
    _subtitle(ax, "Total accidents by month, full years 2016–2022 (partial 2023 excluded)")
    _source(fig)
    return save(fig, "03_month")


def plot_hour_dow_heatmap(m: pd.DataFrame) -> Path:
    """Hour × day-of-week heatmap of accident counts."""
    fig, ax = plt.subplots(figsize=(12, 4.4))
    sns.heatmap(m, cmap=SEQ_CMAP, linewidths=1.5, linecolor=SURFACE, ax=ax,
                cbar_kws={"format": COMPACT, "shrink": 0.85, "pad": 0.015, "label": "Accidents"})
    ax.set_xlabel("Hour of day (local)")
    ax.set_ylabel("")
    ax.tick_params(axis="y", rotation=0)
    ax.grid(False)
    ax.set_title("Weekday 7–8 AM and 4–5 PM are the busiest accident hours of the week")
    _subtitle(ax, "Accidents by day of week × hour, 2016–Mar 2023")
    _source(fig)
    return save(fig, "04_hour_dow_heatmap")


def plot_year_trend(t: pd.DataFrame) -> Path:
    """Two panels: accidents per year, and high-severity share per year."""
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(12, 4.5))
    full, part = t[~t["partial_year"]], t[t["partial_year"]]
    a1.bar(full["year"], full["accidents"], width=0.6, color=BLUE)
    a1.bar(part["year"], part["accidents"], width=0.6, color=NEUTRAL, label="2023: Jan–Mar only")
    for y, v in zip(t["year"], t["accidents"]):
        a1.text(y, v, compact(v), ha="center", va="bottom", fontsize=8.5, color=INK_2)
    a1.yaxis.set_major_formatter(COMPACT)
    a1.grid(axis="x", visible=False)
    a1.set_xticks(t["year"])
    a1.legend(loc="upper left")
    a1.set_title("Recorded accidents grew 4× from 2016 to 2022")
    _subtitle(a1, "Accidents per year")

    a2.plot(t["year"], t["high_severity_pct"], color=BLUE, marker="o", markersize=6,
            markeredgecolor=SURFACE, markeredgewidth=2)
    for _, r in t.iloc[[0, t["high_severity_pct"].idxmin(), -1]].iterrows():
        a2.annotate(f"{r.high_severity_pct:.0f}%", (r.year, r.high_severity_pct),
                    xytext=(0, 9), textcoords="offset points", ha="center",
                    fontsize=8.5, color=INK_2)
    a2.set_xticks(t["year"])
    a2.yaxis.set_major_formatter(PercentFormatter(decimals=0))
    a2.set_ylim(0, t["high_severity_pct"].max() * 1.2)
    a2.set_title("…while the share of high-severity accidents fell")
    _subtitle(a2, "Share of accidents with Severity ≥ 3")
    fig.tight_layout(w_pad=3)
    _source(fig, "growth partly reflects wider data-source coverage")
    return save(fig, "05_year_trend")


# --------------------------------------------------------------------------- #
# Weather
# --------------------------------------------------------------------------- #
def plot_weather(t: pd.DataFrame) -> Path:
    """Two panels sharing the weather order: accident count, and high-severity rate."""
    t = t.sort_values("accidents", ascending=False)
    overall = (t["high_severity_pct"] * t["accidents"]).sum() / t["accidents"].sum()
    rate = "high_severity_adj_pct" if "high_severity_adj_pct" in t else "high_severity_pct"
    above = set(t.loc[t[rate] > overall + 1, "weather_bucket"].astype(str))
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(12, 4.8), sharey=True)
    _hbar(a1, t["weather_bucket"].astype(str), t["accidents"])
    a1.set_title("Most accidents happen in clear or cloudy weather")
    _subtitle(a1, "Accidents by weather condition")
    _hbar(a2, t["weather_bucket"].astype(str), t[rate],
          fmt=lambda v: f"{v:.0f}%", highlight=lambda l: l in above)
    a2.axvline(overall, color=INK_2, linewidth=1)
    a2.text(overall, -0.62, f" all-weather avg {overall:.0f}%", fontsize=8, color=INK_2,
            va="center")
    a2.set_ylim(-0.85, len(t) - 0.5)
    a2.set_title("…but storms, snow and rain make them more severe")
    _subtitle(a2, "Year-adjusted share with Severity ≥ 3 (highlighted: above average)")
    fig.tight_layout(w_pad=3)
    _source(fig)
    return save(fig, "06_weather_count_and_severity")


_VAR_TITLES = {"Visibility(mi)": "Visibility (miles)", "Temperature(F)": "Temperature (°F)",
               "Precipitation(in)": "Precipitation (inches/hr)"}


def plot_weather_numeric_rates(t: pd.DataFrame) -> Path:
    """High-severity rate across bins of visibility, temperature, precipitation."""
    fig, axes = plt.subplots(1, 3, figsize=(13, 4.3), sharey=True)
    for ax, (var, g) in zip(axes, t.groupby("variable", sort=False)):
        rate = "high_severity_adj_pct" if "high_severity_adj_pct" in g else "high_severity_pct"
        ticks = [f"{str(b).replace('0.', '.')}\nn={compact(n)}" if var == "Precipitation(in)"
                 else f"{b}\nn={compact(n)}" for b, n in zip(g["bin"], g["accidents"])]
        bars = ax.bar(ticks, g[rate], width=0.6, color=BLUE)
        for b, v in zip(bars, g[rate]):
            ax.text(b.get_x() + b.get_width() / 2, v, f"{v:.0f}%", ha="center",
                    va="bottom", fontsize=8.5, color=INK_2)
        ax.set_xlabel(_VAR_TITLES[var])
        ax.grid(axis="x", visible=False)
        ax.yaxis.set_major_formatter(PercentFormatter(decimals=0))
        ax.tick_params(axis="x", labelsize=8)
    axes[0].set_ylabel("Year-adjusted share with Severity ≥ 3")
    axes[0].set_title("Heavier precipitation and sub-freezing temperatures raise severity")
    _subtitle(axes[0], "Year-adjusted high-severity share across weather readings "
                       "(precipitation: reported values only)")
    fig.tight_layout(w_pad=2)
    _source(fig)
    return save(fig, "07_weather_numeric_vs_severity")


def plot_weather_distributions(df: pd.DataFrame) -> Path:
    """Weather readings per Severity level.

    Temperature is continuous, so it gets a density outline. Visibility and
    precipitation are reported in a few discrete steps, so cumulative curves
    (share of accidents at or below each value) compare them more honestly.
    """
    fig, axes = plt.subplots(1, 3, figsize=(13, 4.3))
    temp_bins = np.arange(-20, 111, 5)
    for sev, color in SEVERITY_COLORS.items():
        sub = df[df["Severity"] == sev]
        axes[0].hist(sub["Temperature(F)"].clip(-20, 110), bins=temp_bins, density=True,
                     histtype="step", linewidth=1.8, color=color, label=f"Severity {sev}")
        vis = np.sort(sub["Visibility(mi)"].clip(0, 10).to_numpy())
        axes[1].plot(vis, np.arange(1, len(vis) + 1) / len(vis) * 100, color=color,
                     drawstyle="steps-post")
        pr = sub.loc[~sub["precip_missing"] & (sub["Precipitation(in)"] > 0), "Precipitation(in)"]
        pr = np.sort(pr.to_numpy())
        axes[2].plot(pr, np.arange(1, len(pr) + 1) / len(pr) * 100, color=color,
                     drawstyle="steps-post")
    axes[0].set_xlabel(_VAR_TITLES["Temperature(F)"])
    axes[0].set_yticks([])
    axes[0].spines["left"].set_visible(False)
    axes[1].set_xlabel(_VAR_TITLES["Visibility(mi)"] + " · capped at 10")
    axes[1].set_xlim(0, 10)
    axes[1].set_ylim(0, 30)
    axes[1].set_ylabel("Cumulative share of accidents")
    axes[1].yaxis.set_major_formatter(PercentFormatter(decimals=0))
    axes[2].set_xscale("log")
    axes[2].set_xlabel(_VAR_TITLES["Precipitation(in)"] + " · non-zero only")
    axes[2].yaxis.set_major_formatter(PercentFormatter(decimals=0))
    axes[2].set_ylim(0, 100)
    axes[0].set_title("Weather readings overlap heavily; Severity 4 skews colder")
    _subtitle(axes[0], "Temperature density · visibility below 10 mi (zoomed) · "
                       "cumulative precipitation, by Severity")
    axes[0].legend(loc="upper left")
    fig.tight_layout(w_pad=2.5)
    _source(fig, "500k stratified sample, unadjusted")
    return save(fig, "08_weather_distributions_by_severity")


# --------------------------------------------------------------------------- #
# Road conditions
# --------------------------------------------------------------------------- #
def plot_road_share(t: pd.DataFrame) -> Path:
    """Share of accidents occurring at each road feature."""
    fig, ax = plt.subplots(figsize=(9, 5))
    _hbar(ax, t["feature"].str.replace("_", " "), t["share_pct"], fmt=lambda v: f"{v:.1f}%")
    ax.set_title("Traffic signals, crossings and junctions are the top accident sites")
    _subtitle(ax, "Share of all accidents with each road feature nearby")
    _source(fig, "an accident can have several features")
    return save(fig, "09_road_feature_share")


def plot_road_severity(t: pd.DataFrame, min_n: int = 1000) -> Path:
    """Dumbbell: high-severity share with vs without each road feature."""
    adj = "_adj" if "high_sev_adj_pct_with" in t else ""
    w, wo, lift = f"high_sev{adj}_pct_with", f"high_sev{adj}_pct_without", f"severity_lift{adj}_pts"
    t = t[t["accidents"] >= min_n].sort_values(lift)
    fig, ax = plt.subplots(figsize=(9, 5))
    y = np.arange(len(t))
    ax.hlines(y, t[wo], t[w], color=GRID, linewidth=3)
    ax.scatter(t[wo], y, s=60, color=NEUTRAL, edgecolor=SURFACE,
               linewidth=2, zorder=3, label="Without feature")
    ax.scatter(t[w], y, s=60, color=BLUE, edgecolor=SURFACE,
               linewidth=2, zorder=3, label="With feature")
    for yi, (_, r) in zip(y, t.iterrows()):
        ax.text(max(r[w], r[wo]) + 0.6, yi, f"{r[lift]:+.0f} pts", va="center",
                fontsize=8.5, color=INK_2)
    ax.set_yticks(y, t["feature"].str.replace("_", " "))
    ax.xaxis.set_major_formatter(PercentFormatter(decimals=0))
    ax.set_xlim(0, t[[w, wo]].max().max() * 1.15)
    ax.grid(axis="y", visible=False)
    ax.spines["left"].set_visible(False)
    ax.legend(loc="center right")
    ax.set_title("Signals, crossings and stops see milder accidents; junctions see worse ones")
    _subtitle(ax, "Year-adjusted share with Severity ≥ 3, with vs without each feature")
    _source(fig, f"features with ≥{min_n:,} accidents")
    return save(fig, "10_road_feature_severity")


# --------------------------------------------------------------------------- #
# Day / night & severity
# --------------------------------------------------------------------------- #
def _stacked_severity(ax: plt.Axes, mix: pd.DataFrame) -> None:
    """100% stacked horizontal bars of Severity mix (rows = groups)."""
    mix = mix.iloc[::-1]
    left = np.zeros(len(mix))
    labels = [str(i) for i in mix.index]
    for sev in [1, 2, 3, 4]:
        vals = mix[sev].to_numpy() if sev in mix.columns else np.zeros(len(mix))
        ax.barh(labels, vals, left=left, height=0.62, color=SEVERITY_COLORS[sev],
                edgecolor=SURFACE, linewidth=1.5, label=f"Severity {sev}")
        for i, (l, v) in enumerate(zip(left, vals)):
            if v >= 6:
                ax.text(l + v / 2, i, f"{v:.0f}%", ha="center", va="center", fontsize=8,
                        color="white" if sev >= 2 else INK)
        left += vals
    ax.set_xlim(0, 100)
    ax.xaxis.set_major_formatter(PercentFormatter(decimals=0))
    ax.grid(axis="y", visible=False)
    ax.spines["left"].set_visible(False)


def plot_day_night(dn: pd.DataFrame, mix: pd.DataFrame) -> Path:
    """Day vs night: accident counts and severity mix."""
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(12, 3.6), gridspec_kw={"width_ratios": [1, 1.6]})
    _hbar(a1, dn["day_night"].astype(str), dn["accidents"])
    a1.set_title("Two-thirds of accidents happen in daylight")
    _subtitle(a1, "Accidents by light condition")
    _stacked_severity(a2, mix)
    s4 = mix[4] if 4 in mix.columns else mix["4"]
    a2.set_title(f"Severity ≥ 3 is similar, but night has more Severity 4 "
                 f"({s4['Night']:.1f}% vs {s4['Day']:.1f}%)")
    _subtitle(a2, "Severity mix within day and night")
    a2.legend(loc="upper center", bbox_to_anchor=(0.5, -0.15), ncol=4)
    fig.tight_layout(w_pad=3)
    _source(fig)
    return save(fig, "11_day_vs_night")


def plot_severity_distribution(t: pd.DataFrame) -> Path:
    """Overall Severity level distribution."""
    fig, ax = plt.subplots(figsize=(7, 4.2))
    bars = ax.bar(t["Severity"].astype(str), t["accidents"], width=0.55,
                  color=[SEVERITY_COLORS[s] for s in t["Severity"]])
    for b, v, p in zip(bars, t["accidents"], t["share_pct"]):
        ax.text(b.get_x() + b.get_width() / 2, v, f"{compact(v)} · {p:.1f}%", ha="center",
                va="bottom", fontsize=8.5, color=INK_2)
    ax.yaxis.set_major_formatter(COMPACT)
    ax.grid(axis="x", visible=False)
    ax.set_xlabel("Severity (1 = minor traffic delay … 4 = major delay)")
    ax.set_title("Four in five accidents are Severity 2")
    _subtitle(ax, "Accidents by Severity level")
    _source(fig)
    return save(fig, "15_severity_distribution")


def plot_severity_shift(by_weather: pd.DataFrame, by_tod: pd.DataFrame) -> Path:
    """Severity mix across weather buckets and time-of-day bands."""
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(13, 4.6))
    order = by_weather.assign(hs=by_weather.get(3, 0) + by_weather.get(4, 0)).sort_values("hs", ascending=False)
    _stacked_severity(a1, order.drop(columns="hs"))
    a1.set_title("Severity mix shifts with weather…")
    _subtitle(a1, "Severity share within each weather condition (unadjusted)")
    _stacked_severity(a2, by_tod)
    a2.set_title("…and with the time of day")
    _subtitle(a2, "Severity share within each time-of-day band")
    handles, labels = a1.get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=4, bbox_to_anchor=(0.5, -0.04))
    fig.tight_layout(w_pad=3, rect=(0, 0.05, 1, 1))
    _source(fig)
    return save(fig, "16_severity_shift")


# --------------------------------------------------------------------------- #
# Geography
# --------------------------------------------------------------------------- #
def plot_top_states(t: pd.DataFrame) -> Path:
    """Top-10 states by accident count."""
    fig, ax = plt.subplots(figsize=(9, 4.8))
    _hbar(ax, t["State"].astype(str), t["accidents"])
    ax.set_title(f"California alone records {t['share_pct'].iloc[0]:.0f}% of all US accidents")
    _subtitle(ax, "Top 10 states by accident count, 2016–Mar 2023")
    _source(fig)
    return save(fig, "12_top_states")


def plot_top_cities(t: pd.DataFrame) -> Path:
    """Top-20 cities by accident count."""
    fig, ax = plt.subplots(figsize=(9, 7))
    _hbar(ax, t["label"], t["accidents"])
    ax.set_title(f"{t['label'].iloc[0]} leads every US city")
    _subtitle(ax, "Top 20 cities by accident count, 2016–Mar 2023")
    _source(fig)
    return save(fig, "13_top_cities")


def plot_state_per_capita(t: pd.DataFrame, n: int = 15) -> Path:
    """Accidents per 100k residents, top-``n`` states."""
    t = t.head(n)
    fig, ax = plt.subplots(figsize=(9, 5.5))
    _hbar(ax, t["State"].astype(str), t["per_100k"], fmt=lambda v: f"{v:,.0f}")
    ax.set_title(f"Per resident, {t['State'].iloc[0]} has the highest recorded accident rate")
    _subtitle(ax, f"Accidents per 100k residents (2020 Census), top {n} states, 2016–Mar 2023")
    _source(fig, "rates also reflect how well each state's traffic feeds are covered")
    return save(fig, "14_state_per_capita")


# --------------------------------------------------------------------------- #
# Runner
# --------------------------------------------------------------------------- #
def make_all_figures(tables: dict[str, pd.DataFrame], sample: pd.DataFrame) -> list[Path]:
    """Draw every chart from the analysis tables (+ a row-level sample for densities)."""
    set_style()
    paths = [
        plot_hourly(tables["by_hour_weekpart"]),
        plot_day_of_week(tables["by_day_of_week"]),
        plot_month(tables["by_month"]),
        plot_hour_dow_heatmap(tables["hour_dow_matrix"]),
        plot_year_trend(tables["by_year"]),
        plot_weather(tables["by_weather"]),
        plot_weather_numeric_rates(tables["numeric_bins_vs_severity"]),
        plot_weather_distributions(sample),
        plot_road_share(tables["road_features"]),
        plot_road_severity(tables["road_features"]),
        plot_day_night(tables["day_night"], tables["severity_by_day_night"]),
        plot_top_states(tables["top_states"]),
        plot_top_cities(tables["top_cities"]),
        plot_state_per_capita(tables["state_per_capita"]),
        plot_severity_distribution(tables["severity_distribution"]),
        plot_severity_shift(tables["severity_by_weather"], tables["severity_by_time_of_day"]),
    ]
    return paths


def main() -> None:
    """Compute tables on the full clean data and draw all charts."""
    from src.data_loader import load_parquet

    df = load_parquet(config.CLEAN_PARQUET, columns=analysis.ANALYSIS_COLUMNS)
    tables = analysis.run_all(df)
    sample = load_parquet(config.SAMPLE_PARQUET, columns=analysis.ANALYSIS_COLUMNS, verbose=False)
    for p in make_all_figures(tables, sample):
        print("✔", p.relative_to(config.ROOT_DIR))


if __name__ == "__main__":
    main()
