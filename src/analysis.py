"""Aggregations behind every Crashscape chart.

Each function takes a cleaned accidents DataFrame (full data or a sample) and
returns a small tidy DataFrame. ``run_all()`` computes all of them and writes
them to ``reports/tables/`` so the charts, notebook and README share numbers.

Note on Severity: in this dataset Severity (1–4) measures the accident's
*impact on traffic* (length of the delay), not injury severity.

Year adjustment: the share of Severity ≥ 3 accidents falls from ~35% (2016–18)
to ~7% (2022) as data sources changed, and some fields (e.g. precipitation)
are mostly missing before 2019. Raw severity rates per condition therefore mix
"which condition" with "which year". ``high_severity_adj_pct`` removes that by
direct standardisation: observed rate ÷ the rate expected from the group's
year mix, rescaled to the overall rate. Use it when comparing conditions.
"""

from __future__ import annotations

import pandas as pd

from src import config

# 2020 US Census resident population (used to normalise accidents per capita).
STATE_POPULATION_2020: dict[str, int] = {
    "AL": 5024279, "AK": 733391, "AZ": 7151502, "AR": 3011524, "CA": 39538223,
    "CO": 5773714, "CT": 3605944, "DE": 989948, "DC": 689545, "FL": 21538187,
    "GA": 10711908, "HI": 1455271, "ID": 1839106, "IL": 12812508, "IN": 6785528,
    "IA": 3190369, "KS": 2937880, "KY": 4505836, "LA": 4657757, "ME": 1362359,
    "MD": 6177224, "MA": 7029917, "MI": 10077331, "MN": 5706494, "MS": 2961279,
    "MO": 6154913, "MT": 1084225, "NE": 1961504, "NV": 3104614, "NH": 1377529,
    "NJ": 9288994, "NM": 2117522, "NY": 20201249, "NC": 10439388, "ND": 779094,
    "OH": 11799448, "OK": 3959353, "OR": 4237256, "PA": 13002700, "RI": 1097379,
    "SC": 5118425, "SD": 886667, "TN": 6910840, "TX": 29145505, "UT": 3271616,
    "VT": 643077, "VA": 8631393, "WA": 7705281, "WV": 1793716, "WI": 5893718,
    "WY": 576851,
}

# Columns the analysis needs (column pruning when loading the full parquet).
ANALYSIS_COLUMNS: list[str] = [
    "Severity", "high_severity", "year", "month", "month_name", "day_of_week",
    "hour", "is_weekend", "time_of_day", "day_night", "weather_bucket",
    "Visibility(mi)", "Temperature(F)", "Precipitation(in)", "precip_missing",
    "City", "State", *config.ROAD_FEATURES, "any_road_feature",
]

# The dataset ends on 31 Mar 2023, so 2023 is a partial year.
LAST_FULL_YEAR: int = 2022


YEAR_BASE_COL = "_year_base"


def add_year_baseline(df: pd.DataFrame) -> pd.DataFrame:
    """Attach each row's year-level high-severity rate (used for year adjustment)."""
    df[YEAR_BASE_COL] = df.groupby("year")["high_severity"].transform("mean").astype("float32")
    df.attrs["overall_high_sev"] = float(df["high_severity"].mean())
    return df


def _adjusted_pct(hs_mean: pd.Series | float, base_mean: pd.Series | float,
                  overall: float) -> pd.Series | float:
    """Year-standardised high-severity rate (%)."""
    return hs_mean / base_mean * overall * 100


def _severity_summary(g: pd.core.groupby.DataFrameGroupBy) -> pd.DataFrame:
    """Accident count, mean Severity, high-severity rate (%) and, if the frame
    has a year baseline, the year-adjusted rate per group."""
    aggs = {"accidents": ("Severity", "size"),
            "avg_severity": ("Severity", "mean"),
            "high_severity_pct": ("high_severity", "mean")}
    has_base = YEAR_BASE_COL in g.obj.columns
    if has_base:
        aggs["_base"] = (YEAR_BASE_COL, "mean")
    out = g.agg(**aggs)
    if has_base:
        overall = g.obj.attrs.get("overall_high_sev", g.obj["high_severity"].mean())
        out["high_severity_adj_pct"] = _adjusted_pct(out["high_severity_pct"], out["_base"], overall)
        out = out.drop(columns="_base")
    out["high_severity_pct"] *= 100
    return out


# --------------------------------------------------------------------------- #
# Time
# --------------------------------------------------------------------------- #
def by_hour_weekpart(df: pd.DataFrame) -> pd.DataFrame:
    """Share of each week-part's accidents (weekday vs weekend) by hour.

    Shares (%) are used instead of raw counts because there are 5 weekdays but
    only 2 weekend days; the curve *shape* is what we compare.
    """
    t = (df.assign(week_part=df["is_weekend"].map({False: "Weekday", True: "Weekend"}))
           .groupby(["week_part", "hour"], observed=True).size()
           .rename("accidents").reset_index())
    t["share_pct"] = t["accidents"] / t.groupby("week_part")["accidents"].transform("sum") * 100
    # Average accidents per single day of that type, per hour (5 vs 2 days).
    t["per_day_avg_index"] = t["accidents"] / t["week_part"].map({"Weekday": 5, "Weekend": 2})
    return t


def by_day_of_week(df: pd.DataFrame) -> pd.DataFrame:
    """Accident count and severity stats by day of week."""
    return _severity_summary(df.groupby("day_of_week", observed=False)).reset_index()


def by_month(df: pd.DataFrame) -> pd.DataFrame:
    """Accidents by calendar month, full years only (2016–2022).

    2023 is excluded because it only covers January–March and would inflate them.
    """
    full = df[df["year"] <= LAST_FULL_YEAR]
    return _severity_summary(full.groupby("month_name", observed=False)).reset_index()


def hour_dow_matrix(df: pd.DataFrame) -> pd.DataFrame:
    """Hour × day-of-week accident counts (rows: day, cols: hour 0–23)."""
    return (df.groupby(["day_of_week", "hour"], observed=False).size()
              .unstack("hour").reindex(config.DAY_ORDER))


def by_year(df: pd.DataFrame) -> pd.DataFrame:
    """Accidents and severity by year, with a partial-year flag for 2023."""
    t = _severity_summary(df.groupby("year")).reset_index()
    t["partial_year"] = t["year"] > LAST_FULL_YEAR
    return t


def by_time_of_day(df: pd.DataFrame) -> pd.DataFrame:
    """Accidents and severity by time-of-day band."""
    return _severity_summary(df.groupby("time_of_day", observed=False)).reset_index()


# --------------------------------------------------------------------------- #
# Weather
# --------------------------------------------------------------------------- #
def by_weather(df: pd.DataFrame) -> pd.DataFrame:
    """Accident count, mean Severity and high-severity rate by weather bucket."""
    t = _severity_summary(df.groupby("weather_bucket", observed=False)).reset_index()
    t["share_pct"] = t["accidents"] / t["accidents"].sum() * 100
    return t.sort_values("accidents", ascending=False).reset_index(drop=True)


_NUMERIC_BINS: dict[str, tuple[list[float], list[str]]] = {
    "Visibility(mi)": ([-0.01, 0.5, 1, 2, 5, 9.99, 30],
                       ["<0.5", "0.5–1", "1–2", "2–5", "5–10", "10+"]),
    "Temperature(F)": ([-41, 20, 32, 50, 70, 90, 126],
                       ["<20", "20–32", "32–50", "50–70", "70–90", "90+"]),
    "Precipitation(in)": ([-0.01, 0, 0.02, 0.05, 0.1, 0.3, 5],
                          ["0", "0–0.02", "0.02–0.05", "0.05–0.1", "0.1–0.3", "0.3+"]),
}


def numeric_bins_vs_severity(df: pd.DataFrame) -> pd.DataFrame:
    """High-severity rate across bins of visibility, temperature and precipitation.

    Precipitation rows where the reading was missing (imputed as 0) are
    excluded so "0" means a real zero reading.
    """
    frames = []
    for col, (bins, labels) in _NUMERIC_BINS.items():
        sub = df[~df["precip_missing"]] if col == "Precipitation(in)" else df
        b = pd.cut(sub[col], bins=bins, labels=labels)
        t = _severity_summary(sub.groupby(b, observed=False)).reset_index()
        t = t.rename(columns={col: "bin"}).assign(variable=col)
        frames.append(t)
    cols = ["variable", "bin", "accidents", "avg_severity", "high_severity_pct"]
    out = pd.concat(frames, ignore_index=True)
    return out[cols + [c for c in ("high_severity_adj_pct",) if c in out.columns]]


# --------------------------------------------------------------------------- #
# Road conditions
# --------------------------------------------------------------------------- #
def road_features(df: pd.DataFrame) -> pd.DataFrame:
    """Share of accidents at each road feature and severity with vs without it."""
    has_base = YEAR_BASE_COL in df.columns
    overall = df.attrs.get("overall_high_sev", df["high_severity"].mean())
    rows = []
    for feat in config.ROAD_FEATURES:
        mask = df[feat]
        row = {
            "feature": feat,
            "accidents": int(mask.sum()),
            "share_pct": mask.mean() * 100,
            "high_sev_pct_with": df.loc[mask, "high_severity"].mean() * 100 if mask.any() else float("nan"),
            "high_sev_pct_without": df.loc[~mask, "high_severity"].mean() * 100,
            "avg_severity_with": df.loc[mask, "Severity"].mean() if mask.any() else float("nan"),
        }
        if has_base:
            for part, m in (("with", mask), ("without", ~mask)):
                row[f"high_sev_adj_pct_{part}"] = (
                    _adjusted_pct(df.loc[m, "high_severity"].mean(),
                                  df.loc[m, YEAR_BASE_COL].mean(), overall)
                    if m.any() else float("nan"))
        rows.append(row)
    t = pd.DataFrame(rows)
    t["severity_lift_pts"] = t["high_sev_pct_with"] - t["high_sev_pct_without"]
    if has_base:
        t["severity_lift_adj_pts"] = t["high_sev_adj_pct_with"] - t["high_sev_adj_pct_without"]
    return t.sort_values("share_pct", ascending=False).reset_index(drop=True)


# --------------------------------------------------------------------------- #
# Day vs night, severity
# --------------------------------------------------------------------------- #
def severity_mix(df: pd.DataFrame, by: str) -> pd.DataFrame:
    """Percentage of each Severity level within each group of ``by`` (rows sum to 100)."""
    return (pd.crosstab(df[by], df["Severity"], normalize="index") * 100).round(3)


def day_night(df: pd.DataFrame) -> pd.DataFrame:
    """Accidents and severity stats for day vs night."""
    return _severity_summary(df.groupby("day_night", observed=False)).reset_index()


def severity_distribution(df: pd.DataFrame) -> pd.DataFrame:
    """Overall count and share of each Severity level."""
    t = df["Severity"].value_counts().sort_index().rename("accidents").to_frame()
    t["share_pct"] = t["accidents"] / t["accidents"].sum() * 100
    return t.reset_index()


# --------------------------------------------------------------------------- #
# Geography
# --------------------------------------------------------------------------- #
def top_states(df: pd.DataFrame, n: int = 10) -> pd.DataFrame:
    """Top-``n`` states by accident count, with each state's share of *all* accidents."""
    t = _severity_summary(df.groupby("State", observed=True)).nlargest(n, "accidents").reset_index()
    t["share_pct"] = t["accidents"] / len(df) * 100
    return t


def top_cities(df: pd.DataFrame, n: int = 20) -> pd.DataFrame:
    """Top-``n`` cities by accident count (city labelled with its state)."""
    t = _severity_summary(df.groupby(["City", "State"], observed=True)).nlargest(n, "accidents")
    t = t.reset_index()
    t["label"] = t["City"].astype(str) + ", " + t["State"].astype(str)
    return t


def state_per_capita(df: pd.DataFrame) -> pd.DataFrame:
    """Accidents per 100k residents (2020 Census) per state, whole 2016–2023 period.

    If ``df`` is a sample, counts are scaled back up to the full-data total via
    the ``scale`` attribute stored in ``df.attrs`` (defaults to 1).
    """
    scale = df.attrs.get("scale", 1.0)
    counts = df.groupby("State", observed=True).size().mul(scale).rename("accidents")
    pop = pd.Series(STATE_POPULATION_2020, name="population")
    t = pd.concat([counts, pop], axis=1, join="inner")
    t["per_100k"] = t["accidents"] / t["population"] * 1e5
    return t.sort_values("per_100k", ascending=False).reset_index(names="State")


# --------------------------------------------------------------------------- #
# Runner
# --------------------------------------------------------------------------- #
def run_all(df: pd.DataFrame, save: bool = True) -> dict[str, pd.DataFrame]:
    """Compute every analysis table; optionally write each to ``reports/tables``."""
    if YEAR_BASE_COL not in df.columns:
        df = add_year_baseline(df)
    tables: dict[str, pd.DataFrame] = {
        "by_hour_weekpart": by_hour_weekpart(df),
        "by_day_of_week": by_day_of_week(df),
        "by_month": by_month(df),
        "hour_dow_matrix": hour_dow_matrix(df),
        "by_year": by_year(df),
        "by_time_of_day": by_time_of_day(df),
        "by_weather": by_weather(df),
        "numeric_bins_vs_severity": numeric_bins_vs_severity(df),
        "road_features": road_features(df),
        "day_night": day_night(df),
        "severity_distribution": severity_distribution(df),
        "severity_by_weather": severity_mix(df, "weather_bucket"),
        "severity_by_day_night": severity_mix(df, "day_night"),
        "severity_by_time_of_day": severity_mix(df, "time_of_day"),
        "top_states": top_states(df),
        "top_cities": top_cities(df),
        "state_per_capita": state_per_capita(df),
    }
    if save:
        config.ensure_dirs()
        for name, t in tables.items():
            keep_index = name in {"hour_dow_matrix"} or name.startswith("severity_by_")
            t.round(4).to_csv(config.TABLES_DIR / f"{name}.csv", index=keep_index)
    return tables
