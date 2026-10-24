"""Central configuration for Crashscape: paths, sampling and column settings."""

from __future__ import annotations

from pathlib import Path

# --------------------------------------------------------------------------- #
# Paths
# --------------------------------------------------------------------------- #
ROOT_DIR: Path = Path(__file__).resolve().parents[1]

DATA_DIR: Path = ROOT_DIR / "data"
RAW_DIR: Path = DATA_DIR / "raw"
PROCESSED_DIR: Path = DATA_DIR / "processed"

RAW_ZIP: Path = RAW_DIR / "US_Accidents_March23.csv.zip"
RAW_CSV: Path = RAW_DIR / "US_Accidents_March23.csv"

FULL_PARQUET: Path = PROCESSED_DIR / "accidents.parquet"          # all rows, raw columns
SAMPLE_RAW_PARQUET: Path = PROCESSED_DIR / "accidents_sample_raw.parquet"  # sample, raw columns
CLEAN_PARQUET: Path = PROCESSED_DIR / "accidents_clean.parquet"   # all rows, cleaned + features
SAMPLE_PARQUET: Path = PROCESSED_DIR / "accidents_sample.parquet" # stratified sample, cleaned
MINI_PARQUET: Path = PROCESSED_DIR / "accidents_mini.parquet"     # small committable sample

REPORTS_DIR: Path = ROOT_DIR / "reports"
FIGURES_DIR: Path = REPORTS_DIR / "figures"
MAPS_DIR: Path = REPORTS_DIR / "maps"
TABLES_DIR: Path = REPORTS_DIR / "tables"

# --------------------------------------------------------------------------- #
# Sampling
# --------------------------------------------------------------------------- #
RANDOM_STATE: int = 42
SAMPLE_SIZE: int = 500_000
MINI_SAMPLE_SIZE: int = 50_000
CHUNK_SIZE: int = 1_000_000
STRATIFY_COLS: list[str] = ["State", "Severity"]

# Max file size (MB) we allow inside the git repo.
GIT_MAX_FILE_MB: int = 25

# --------------------------------------------------------------------------- #
# Columns
# --------------------------------------------------------------------------- #
ROAD_FEATURES: list[str] = [
    "Amenity", "Bump", "Crossing", "Give_Way", "Junction", "No_Exit", "Railway",
    "Roundabout", "Station", "Stop", "Traffic_Calming", "Traffic_Signal",
]

WEATHER_NUMERIC: list[str] = [
    "Temperature(F)", "Humidity(%)", "Pressure(in)", "Visibility(mi)",
    "Wind_Speed(mph)", "Precipitation(in)",
]

USECOLS: list[str] = [
    "ID", "Severity", "Start_Time", "End_Time", "Start_Lat", "Start_Lng",
    "Distance(mi)", "Street", "City", "County", "State", "Timezone",
    *WEATHER_NUMERIC, "Wind_Direction", "Weather_Condition",
    *ROAD_FEATURES, "Sunrise_Sunset",
]

# Read dtypes. Strings with low cardinality → category; floats → float32.
READ_DTYPES: dict[str, str] = {
    "ID": "string",
    "Severity": "int8",
    "Start_Time": "string",
    "End_Time": "string",
    "Start_Lat": "float32",
    "Start_Lng": "float32",
    "Distance(mi)": "float32",
    "Street": "category",
    "City": "category",
    "County": "category",
    "State": "category",
    "Timezone": "category",
    **{c: "float32" for c in WEATHER_NUMERIC},
    "Wind_Direction": "category",
    "Weather_Condition": "category",
    **{c: "boolean" for c in ROAD_FEATURES},
    "Sunrise_Sunset": "category",
}

WEATHER_BUCKETS: list[str] = [
    "Clear", "Cloudy", "Rain", "Snow", "Fog", "Thunderstorm", "Windy", "Other",
]

DAY_ORDER: list[str] = [
    "Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday",
]
MONTH_ORDER: list[str] = [
    "Jan", "Feb", "Mar", "Apr", "May", "Jun",
    "Jul", "Aug", "Sep", "Oct", "Nov", "Dec",
]

HIGH_SEVERITY_THRESHOLD: int = 3

# City used for the clustered-marker map.
FOCUS_CITY: tuple[str, str] = ("Los Angeles", "CA")


def ensure_dirs() -> None:
    """Create all output directories if they don't already exist."""
    for d in (RAW_DIR, PROCESSED_DIR, FIGURES_DIR, MAPS_DIR, TABLES_DIR):
        d.mkdir(parents=True, exist_ok=True)
