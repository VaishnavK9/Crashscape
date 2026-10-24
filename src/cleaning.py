"""Clean the raw accident table and derive time, weather and road features.

Pipeline (``clean()``):

1. Parse ``Start_Time`` / ``End_Time`` (mixed formats, some with nanoseconds).
2. Derive hour, day of week, month, year, weekend flag, season, time-of-day
   band and ``duration_min``.
3. Resolve day/night from ``Sunrise_Sunset`` (falling back to the hour).
4. Group ~145 ``Weather_Condition`` strings into 8 buckets.
5. Null out physically implausible sensor readings, then impute weather
   numerics with the State × month median (precipitation → 0 + missing flag).
6. Summarise the boolean road-feature columns.

Usage::

    python -m src.cleaning     # clean full data → write clean parquet, sample, mini
"""

from __future__ import annotations

import time

import numpy as np
import pandas as pd

from src import config
from src.data_loader import load_parquet, report_memory, stratified_sample

# --------------------------------------------------------------------------- #
# Weather bucketing
# --------------------------------------------------------------------------- #
# Ordered (bucket, keywords) rules: the first match wins, so mixed conditions
# such as "Light Rain / Windy" land in the more hazardous bucket (Rain).
_WEATHER_RULES: list[tuple[str, tuple[str, ...]]] = [
    ("Thunderstorm", ("thunder", "t-storm", "storm")),
    ("Snow", ("snow", "sleet", "ice pellets", "wintry", "freezing rain",
              "freezing drizzle", "hail")),
    ("Rain", ("rain", "drizzle", "shower")),
    ("Fog", ("fog", "mist", "haze", "smoke", "dust", "sand", "ash")),
    ("Windy", ("windy", "squall", "tornado", "funnel")),
    ("Cloudy", ("cloud", "overcast")),
    ("Clear", ("fair", "clear")),
]


def bucket_weather(condition: str | float | None) -> str:
    """Map a raw ``Weather_Condition`` string onto one of 8 weather buckets.

    >>> bucket_weather("Light Rain / Windy")
    'Rain'
    >>> bucket_weather("Mostly Cloudy")
    'Cloudy'
    """
    if not isinstance(condition, str):
        return "Other"
    s = condition.lower()
    for bucket, keywords in _WEATHER_RULES:
        if any(k in s for k in keywords):
            return bucket
    return "Other"


def add_weather_bucket(df: pd.DataFrame) -> pd.DataFrame:
    """Add an ordered categorical ``weather_bucket`` column.

    The rule is applied once per *unique* condition (~145) rather than per row.
    """
    cond = df["Weather_Condition"].astype("category")
    mapping = {c: bucket_weather(c) for c in cond.cat.categories}
    df["weather_bucket"] = pd.Categorical(
        cond.map(mapping).astype("object").fillna("Other"),
        categories=config.WEATHER_BUCKETS,
    )
    return df


# --------------------------------------------------------------------------- #
# Time features
# --------------------------------------------------------------------------- #
_TIME_OF_DAY_BINS = [-1, 5, 9, 15, 19, 23]
_TIME_OF_DAY_LABELS = ["Night (0-5)", "Morning rush (6-9)", "Midday (10-15)",
                       "Evening rush (16-19)", "Late evening (20-23)"]
_SEASON_BY_MONTH = {12: "Winter", 1: "Winter", 2: "Winter", 3: "Spring", 4: "Spring",
                    5: "Spring", 6: "Summer", 7: "Summer", 8: "Summer", 9: "Fall",
                    10: "Fall", 11: "Fall"}
MAX_DURATION_MIN: int = 7 * 24 * 60  # anything longer than a week is a data artifact


def parse_times(df: pd.DataFrame) -> pd.DataFrame:
    """Parse ``Start_Time``/``End_Time`` (mixed formats) to datetimes.

    Times in the dataset are local to the accident location, so the derived
    hour is the local clock hour.
    """
    for col in ("Start_Time", "End_Time"):
        df[col] = pd.to_datetime(df[col], format="mixed", errors="coerce")
    return df


def add_time_features(df: pd.DataFrame) -> pd.DataFrame:
    """Derive calendar/time-of-day features and ``duration_min`` from ``Start_Time``."""
    st = df["Start_Time"].dt
    df["year"] = st.year.astype("int16")
    df["month"] = st.month.astype("int8")
    # Names built from integer codes (much faster than strftime on millions of rows).
    df["month_name"] = pd.Categorical.from_codes(df["month"] - 1, config.MONTH_ORDER,
                                                 ordered=True)
    df["dow"] = st.day_of_week.astype("int8")  # Monday = 0
    df["day_of_week"] = pd.Categorical.from_codes(df["dow"], config.DAY_ORDER,
                                                  ordered=True)
    df["hour"] = st.hour.astype("int8")
    df["is_weekend"] = df["dow"] >= 5
    df["time_of_day"] = pd.cut(df["hour"], bins=_TIME_OF_DAY_BINS,
                               labels=_TIME_OF_DAY_LABELS, ordered=True)
    df["season"] = pd.Categorical(df["month"].map(_SEASON_BY_MONTH),
                                  categories=["Winter", "Spring", "Summer", "Fall"],
                                  ordered=True)

    dur = (df["End_Time"] - df["Start_Time"]).dt.total_seconds() / 60
    dur = dur.where((dur > 0) & (dur <= MAX_DURATION_MIN))
    df["duration_min"] = dur.astype("float32")
    return df


def add_day_night(df: pd.DataFrame) -> pd.DataFrame:
    """Add ``day_night`` (Day/Night) from ``Sunrise_Sunset``, filling gaps by hour."""
    fallback = np.where(df["hour"].between(6, 18), "Day", "Night")
    dn = df["Sunrise_Sunset"].astype("object").where(df["Sunrise_Sunset"].notna(), fallback)
    df["day_night"] = pd.Categorical(dn, categories=["Day", "Night"])
    return df


# --------------------------------------------------------------------------- #
# Weather numerics: outliers + imputation
# --------------------------------------------------------------------------- #
# Physically plausible ranges; values outside become NaN before imputation.
PLAUSIBLE_RANGES: dict[str, tuple[float, float]] = {
    "Temperature(F)": (-40, 125),
    "Humidity(%)": (0, 100),
    "Pressure(in)": (25, 32),
    "Visibility(mi)": (0, 30),
    "Wind_Speed(mph)": (0, 100),
    "Precipitation(in)": (0, 5),
}

_WIND_DIR_MAP = {"CALM": "Calm", "VAR": "Variable", "VARIABLE": "Variable",
                 "NORTH": "N", "SOUTH": "S", "EAST": "E", "WEST": "W"}


def clean_weather_numerics(df: pd.DataFrame) -> pd.DataFrame:
    """Null implausible readings and impute weather numerics.

    * ``Precipitation(in)``: NaN usually means "no precipitation reported", so
      it is set to 0 and ``precip_missing`` records the gap.
    * Other numerics: median of the same State × month, then the State median,
      then the global median.
    """
    for col, (lo, hi) in PLAUSIBLE_RANGES.items():
        df[col] = df[col].where(df[col].between(lo, hi))

    df["precip_missing"] = df["Precipitation(in)"].isna()
    df["Precipitation(in)"] = df["Precipitation(in)"].fillna(0).astype("float32")

    impute_cols = [c for c in config.WEATHER_NUMERIC if c != "Precipitation(in)"]
    for col in impute_cols:
        if not df[col].isna().any():
            continue
        by_state_month = df.groupby(["State", "month"], observed=True)[col].transform("median")
        by_state = df.groupby("State", observed=True)[col].transform("median")
        df[col] = (df[col].fillna(by_state_month).fillna(by_state)
                          .fillna(df[col].median()).astype("float32"))

    wd = df["Wind_Direction"].astype("object").str.strip()
    wd = wd.str.upper().map(_WIND_DIR_MAP).fillna(wd)
    df["Wind_Direction"] = wd.fillna("Unknown").astype("category")
    return df


# --------------------------------------------------------------------------- #
# Road features
# --------------------------------------------------------------------------- #
def add_road_features(df: pd.DataFrame) -> pd.DataFrame:
    """Make road-feature flags plain bools and add count / any-feature summaries."""
    for col in config.ROAD_FEATURES:
        df[col] = df[col].fillna(False).astype(bool)
    df["road_feature_count"] = df[config.ROAD_FEATURES].sum(axis=1).astype("int8")
    df["any_road_feature"] = df["road_feature_count"] > 0
    return df


# --------------------------------------------------------------------------- #
# Orchestration
# --------------------------------------------------------------------------- #
def clean(df: pd.DataFrame, verbose: bool = True) -> pd.DataFrame:
    """Run the full cleaning + feature pipeline on a raw accidents DataFrame."""
    t0 = time.perf_counter()
    n0 = len(df)

    def log(step: str) -> None:
        if verbose:
            print(f"  ✓ {step:<34} ({time.perf_counter() - t0:5.1f}s)")

    df = parse_times(df)
    df = df.dropna(subset=["Start_Time", "Start_Lat", "Start_Lng"]).copy()
    log(f"parse times (dropped {n0 - len(df):,} rows)")
    df = add_time_features(df)
    log("time features")
    df = add_day_night(df)
    log("day / night")
    df = add_weather_bucket(df)
    log("weather buckets")
    df = clean_weather_numerics(df)
    log("weather numerics + imputation")
    df = add_road_features(df)
    log("road features")

    df["high_severity"] = df["Severity"] >= config.HIGH_SEVERITY_THRESHOLD
    for col in ("City", "County", "State", "Street", "Timezone", "Weather_Condition"):
        df[col] = df[col].astype("category")
    df = df.drop(columns=["Sunrise_Sunset"]).reset_index(drop=True)
    if verbose:
        report_memory(df, "clean")
    return df


def build_clean_datasets() -> None:
    """Clean the full Parquet file and write the clean full / sample / mini files."""
    config.ensure_dirs()
    print("Cleaning full dataset ...")
    raw = load_parquet(config.FULL_PARQUET)
    df = clean(raw)
    del raw

    df.to_parquet(config.CLEAN_PARQUET, index=False, compression="zstd")
    _report_file(config.CLEAN_PARQUET)

    sample = stratified_sample(df, n=config.SAMPLE_SIZE)
    sample.to_parquet(config.SAMPLE_PARQUET, index=False, compression="zstd")
    report_memory(sample, "sample")
    _report_file(config.SAMPLE_PARQUET)

    mini = stratified_sample(sample, n=config.MINI_SAMPLE_SIZE)
    mini.to_parquet(config.MINI_PARQUET, index=False, compression="zstd")
    _report_file(config.MINI_PARQUET)


def _report_file(path) -> None:
    print(f"✔ Wrote {path.relative_to(config.ROOT_DIR)} "
          f"({path.stat().st_size / 1024**2:,.1f} MB)")


if __name__ == "__main__":
    build_clean_datasets()
