from __future__ import annotations

import random
from pathlib import Path
from typing import Iterable

import pandas as pd

REQUIRED_COLUMNS = {"open", "high", "low", "close", "volume"}


def _detect_datetime_column(df: pd.DataFrame) -> str:
    for candidate in ("timestamp", "datetime", df.columns[0]):
        if candidate in df.columns:
            return candidate
    raise ValueError("No datetime column detected in dataset")


def _set_datetime_index(df: pd.DataFrame, column: str) -> pd.DataFrame:
    df[column] = pd.to_datetime(df[column], utc=False, errors="coerce")
    if df[column].isna().any():
        raise ValueError("Datetime column contains invalid entries")
    df = df.set_index(column)
    return df.sort_index()


def _validate_columns(df: pd.DataFrame) -> None:
    missing = REQUIRED_COLUMNS - set(df.columns)
    if missing:
        raise ValueError(f"Data file missing required columns: {sorted(missing)}")


def _validate_frequency(index: pd.DatetimeIndex) -> None:
    if len(index) < 3:
        return
    diffs = index.to_series().diff().dropna()
    expected = pd.Timedelta(minutes=15)
    off = diffs[~(diffs == expected)]
    if len(off) > len(diffs) * 0.05:
        raise ValueError("Dataset is not consistently 15-minute cadence")


def _clean_frame(df: pd.DataFrame) -> pd.DataFrame:
    df = df[~df.index.duplicated(keep="last")]
    df = df.sort_index()
    df = df.dropna(subset=list(REQUIRED_COLUMNS))
    return df


def load_random_data(
    data_dir: str = "data",
    *,
    sample_k: int = 1,
    random_seed: int | None = None,
) -> tuple[pd.DataFrame, list[Path]]:
    """Randomly sample CSVs (recursively) and return combined price data."""
    if sample_k < 1:
        raise ValueError("sample_k must be >= 1")

    directory = Path(data_dir)
    if not directory.exists() or not directory.is_dir():
        raise FileNotFoundError(f"Data directory not found: {directory}")

    csv_files = sorted(p for p in directory.rglob("*.csv") if p.is_file())
    if not csv_files:
        raise FileNotFoundError(f"No CSV files found in {directory}")

    if sample_k > len(csv_files):
        raise ValueError(f"Requested sample_k={sample_k} but only {len(csv_files)} file(s) available")
    
    if random_seed:
        random.seed(random_seed)
    start = random.randint(0, len(csv_files) - sample_k)
    chosen = csv_files[start : start + sample_k]
    print("[data-loader] sampled files:", ", ".join(str(p) for p in chosen))

    frames: list[pd.DataFrame] = []
    for path in chosen:
        df = pd.read_csv(path)
        df.columns = [c.lower() for c in df.columns]
        dt_col = _detect_datetime_column(df)
        df = _set_datetime_index(df, dt_col)
        _validate_columns(df)
        _validate_frequency(df.index)
        frames.append(_clean_frame(df))

    combined = pd.concat(frames, axis=0).sort_index()
    combined = combined[~combined.index.duplicated(keep="last")]
    combined = combined.dropna(subset=list(REQUIRED_COLUMNS))

    return combined, chosen
