"""Load the US Accidents CSV memory-efficiently and convert it to Parquet.

The raw CSV is ~3GB / ~7.7M rows, so it is never loaded in one go. Instead:

1. Only the columns in ``config.USECOLS`` are read, in chunks of ``CHUNK_SIZE``.
2. Each chunk is written straight into a single Parquet file via a
   ``pyarrow.parquet.ParquetWriter`` (strings are dictionary-encoded on disk).
3. A reproducible stratified sample (State × Severity) is drawn from the Parquet
   file for fast iteration and for the dashboard.

Usage::

    python -m src.data_loader            # convert + sample
    python -m src.data_loader --sample   # only (re)build the sample
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from src import config

# Columns stored as categories in memory. On disk they are plain strings, which
# Parquet dictionary-encodes, so chunks with different category sets stay
# schema-compatible.
CATEGORY_COLS: list[str] = [c for c, t in config.READ_DTYPES.items() if t == "category"]


def memory_mb(df: pd.DataFrame) -> float:
    """Return the deep memory usage of ``df`` in megabytes."""
    return df.memory_usage(deep=True).sum() / 1024**2


def report_memory(df: pd.DataFrame, label: str = "DataFrame") -> None:
    """Print row/column counts and deep memory usage of ``df``."""
    print(f"[{label}] {len(df):,} rows × {df.shape[1]} cols | "
          f"memory: {memory_mb(df):,.1f} MB")


def _csv_source() -> Path:
    """Return the CSV path if it is unzipped, otherwise the zip (pandas reads it directly)."""
    if config.RAW_CSV.exists():
        return config.RAW_CSV
    if config.RAW_ZIP.exists():
        return config.RAW_ZIP
    raise FileNotFoundError(
        f"Dataset not found. Put US_Accidents_March23.csv (or its .zip) in {config.RAW_DIR}.\n"
        "Download: kaggle datasets download -d sobhanmoosavi/us-accidents -p data/raw --unzip"
    )


def _csv_read_dtypes() -> dict[str, str]:
    """Dtypes used while parsing the CSV (categories are read as strings)."""
    return {c: ("string" if t == "category" else t) for c, t in config.READ_DTYPES.items()}


def iter_csv_chunks(chunksize: int = config.CHUNK_SIZE):
    """Yield DataFrame chunks of the raw CSV, restricted to ``config.USECOLS``."""
    src = _csv_source()
    print(f"Reading {src.name} in chunks of {chunksize:,} rows ...")
    yield from pd.read_csv(
        src,
        usecols=config.USECOLS,
        dtype=_csv_read_dtypes(),
        chunksize=chunksize,
        engine="c",
        low_memory=True,
    )


def convert_csv_to_parquet(out_path: Path = config.FULL_PARQUET,
                           chunksize: int = config.CHUNK_SIZE) -> Path:
    """Stream the raw CSV into a single compressed Parquet file.

    Args:
        out_path: Destination Parquet path.
        chunksize: Rows per CSV chunk.

    Returns:
        The path of the written Parquet file.
    """
    config.ensure_dirs()
    t0 = time.perf_counter()
    writer: pq.ParquetWriter | None = None
    schema: pa.Schema | None = None
    total = 0
    try:
        for i, chunk in enumerate(iter_csv_chunks(chunksize)):
            chunk = chunk[config.USECOLS]
            table = pa.Table.from_pandas(chunk, preserve_index=False)
            if writer is None:
                schema = table.schema
                writer = pq.ParquetWriter(out_path, schema, compression="zstd")
            writer.write_table(table.cast(schema))
            total += len(chunk)
            print(f"  chunk {i + 1:>2}: {total:>10,} rows written "
                  f"({time.perf_counter() - t0:,.0f}s)")
    finally:
        if writer is not None:
            writer.close()
    size_mb = out_path.stat().st_size / 1024**2
    print(f"✔ Wrote {out_path.relative_to(config.ROOT_DIR)} | {total:,} rows | "
          f"{size_mb:,.1f} MB | {time.perf_counter() - t0:,.0f}s")
    return out_path


def to_categories(df: pd.DataFrame, cols: list[str] | None = None) -> pd.DataFrame:
    """Convert string columns to ``category`` dtype in place and return ``df``."""
    for c in cols or CATEGORY_COLS:
        if c in df.columns and not isinstance(df[c].dtype, pd.CategoricalDtype):
            df[c] = df[c].astype("category")
    return df


def load_parquet(path: Path = config.FULL_PARQUET,
                 columns: list[str] | None = None,
                 verbose: bool = True) -> pd.DataFrame:
    """Load a Crashscape Parquet file with memory-efficient dtypes.

    Args:
        path: Parquet file to load.
        columns: Optional subset of columns (column pruning keeps memory low).
        verbose: Print the memory usage after loading.
    """
    if not path.exists():
        raise FileNotFoundError(f"{path} not found. Run `python -m src.data_loader` first.")
    df = pd.read_parquet(path, columns=columns)
    df = to_categories(df)
    if verbose:
        report_memory(df, path.name)
    return df


def stratified_sample(df: pd.DataFrame,
                      n: int = config.SAMPLE_SIZE,
                      strata: list[str] | None = None,
                      random_state: int = config.RANDOM_STATE) -> pd.DataFrame:
    """Draw a proportional stratified sample of roughly ``n`` rows.

    Each stratum (default State × Severity) keeps the same fraction of rows, so
    the sample preserves the joint distribution of the strata.
    """
    strata = strata or config.STRATIFY_COLS
    # Oversample by 0.5% so per-stratum rounding never leaves us short of n.
    frac = min(1.0, n / len(df) * 1.005)
    sample = (
        df.groupby(strata, observed=True, group_keys=False)
          .sample(frac=frac, random_state=random_state)
    )
    # Trim to exactly n (uniform, so stratum shares are preserved) and shuffle.
    sample = sample.sample(n=min(n, len(sample)), random_state=random_state)
    return sample.reset_index(drop=True)


def build_sample(n: int = config.SAMPLE_SIZE,
                 out_path: Path = config.SAMPLE_RAW_PARQUET) -> pd.DataFrame:
    """Create and save a stratified sample (raw columns) from the full Parquet file."""
    df = load_parquet(config.FULL_PARQUET)
    sample = stratified_sample(df, n=n)
    sample.to_parquet(out_path, index=False, compression="zstd")
    report_memory(sample, "sample")
    print(f"✔ Wrote {out_path.relative_to(config.ROOT_DIR)} | "
          f"{out_path.stat().st_size / 1024**2:,.1f} MB")
    _compare_distributions(df, sample)
    return sample


def _compare_distributions(full: pd.DataFrame, sample: pd.DataFrame) -> None:
    """Print Severity and top-state shares for full data vs sample."""
    sev = pd.DataFrame({
        "full_%": full["Severity"].value_counts(normalize=True).sort_index() * 100,
        "sample_%": sample["Severity"].value_counts(normalize=True).sort_index() * 100,
    }).round(2)
    print("\nSeverity share (full vs sample):\n", sev)
    top = full["State"].value_counts(normalize=True).head(5).index
    st = pd.DataFrame({
        "full_%": full["State"].value_counts(normalize=True)[top] * 100,
        "sample_%": sample["State"].value_counts(normalize=True)[top] * 100,
    }).round(2)
    print("\nTop-5 state share (full vs sample):\n", st)


def main() -> None:
    """CLI entry point."""
    parser = argparse.ArgumentParser(description="Convert US Accidents CSV to Parquet + sample.")
    parser.add_argument("--sample", action="store_true", help="Only rebuild the sample.")
    parser.add_argument("--n", type=int, default=config.SAMPLE_SIZE, help="Sample size.")
    args = parser.parse_args()

    if not args.sample:
        convert_csv_to_parquet()
    build_sample(n=args.n)


if __name__ == "__main__":
    main()
