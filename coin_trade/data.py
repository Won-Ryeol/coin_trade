from __future__ import annotations

import random
from pathlib import Path
from typing import Iterable
import numpy as np

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
    if diffs.empty:
        return
    minutes = diffs.dt.total_seconds() / 60.0
    mod = np.mod(minutes, 15.0)
    valid = (mod < 1e-6) | ((15.0 - mod) < 1e-6)
    if not (minutes == 15).any():
        raise ValueError("Dataset is not consistently 15-minute cadence")
    if not valid.all():
        invalid = minutes[~valid]
        if len(invalid) > len(minutes) * 0.1:
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
    min_rows: int = 60,
    max_attempts: int = 10,
) -> tuple[pd.DataFrame, list[Path]]:
    """Randomly sample CSVs (recursively) and return combined price data."""
    if sample_k < 1:
        raise ValueError("sample_k must be >= 1")
    if min_rows < 1:
        raise ValueError("min_rows must be >= 1")

    directory = Path(data_dir)
    if not directory.exists() or not directory.is_dir():
        raise FileNotFoundError(f"Data directory not found: {directory}")

    csv_files = sorted(p for p in directory.rglob("*.csv") if p.is_file() and p.stat().st_size > 200)
    if not csv_files:
        raise FileNotFoundError(f"No CSV files found in {directory}")

    if sample_k > len(csv_files):
        raise ValueError(f"Requested sample_k={sample_k} but only {len(csv_files)} file(s) available")

    rng = random.Random(random_seed)
    candidates = csv_files.copy()
    attempts = 0
    frames: list[pd.DataFrame] = []
    chosen: list[Path] = []

    while attempts < max_attempts and candidates and len(chosen) < sample_k:
        pick_count = min(sample_k - len(chosen), len(candidates))
        trial = rng.sample(candidates, k=pick_count)
        print(f"[data-loader] sampled files (attempt {attempts + 1}):", ", ".join(str(p) for p in trial))

        for path in trial:
            try:
                df = pd.read_csv(path)
            except pd.errors.EmptyDataError:
                print(f"[data-loader] skipping empty file: {path}")
                continue
            if df.empty:
                print(f"[data-loader] skipping empty file: {path}")
                continue

            df.columns = [c.lower() for c in df.columns]
            dt_col = _detect_datetime_column(df)
            df = _set_datetime_index(df, dt_col)
            _validate_columns(df)
            _validate_frequency(df.index)
            cleaned = _clean_frame(df)
            if cleaned.empty:
                print(f"[data-loader] skipping file with no usable rows: {path}")
                continue
            if len(cleaned) < min_rows:
                print(f"[data-loader] skipping short file (<{min_rows} rows): {path}")
                continue

            frames.append(cleaned)
            chosen.append(path)

        candidates = [p for p in candidates if p not in trial]
        attempts += 1

    if not frames:
        raise ValueError("No usable CSV rows found in sampled files")

    combined = pd.concat(frames, axis=0).sort_index()
    combined = combined[~combined.index.duplicated(keep="last")]
    combined = combined.dropna(subset=list(REQUIRED_COLUMNS))

    return combined, chosen
