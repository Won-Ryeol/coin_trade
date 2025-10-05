from __future__ import annotations

import random
from pathlib import Path
import numpy as np

import pandas as pd

REQUIRED_COLUMNS = {"open", "high", "low", "close", "volume"}

MAX_FILE_START_GAP = pd.Timedelta(days=3)

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
    """Randomly select a contiguous block of CSVs and return combined price data."""
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
    if random_seed is not None:
        random.seed(random_seed)

    start_indices = list(range(len(csv_files) - sample_k + 1))
    rng.shuffle(start_indices)

    last_error: Exception | None = None

    def _load_frame(path: Path) -> pd.DataFrame | None:
        try:
            df = pd.read_csv(path)
        except pd.errors.EmptyDataError:
            print(f"[data-loader] skipping empty file: {path}")
            return None
        if df.empty:
            print(f"[data-loader] skipping empty file: {path}")
            return None

        df.columns = [c.lower() for c in df.columns]
        dt_col = _detect_datetime_column(df)
        df = _set_datetime_index(df, dt_col)
        _validate_columns(df)
        _validate_frequency(df.index)
        cleaned = _clean_frame(df)
        if cleaned.empty:
            print(f"[data-loader] skipping file with no usable rows: {path}")
            return None
        if len(cleaned) < min_rows:
            print(f"[data-loader] skipping short file (<{min_rows} rows): {path}")
            return None
        return cleaned

    attempts = 0
    for start in start_indices or [0]:
        attempts += 1
        if max_attempts > 0 and attempts > max_attempts:
            break

        block_paths = csv_files[start : start + sample_k]
        print(f"[data-loader] sampled block (attempt {attempts}): " + ", ".join(str(p) for p in block_paths))

        frames: list[pd.DataFrame] = []
        chosen: list[Path] = []
        start_times: list[pd.Timestamp] = []
        block_valid = True

        for path in block_paths:
            try:
                cleaned = _load_frame(path)
            except ValueError as exc:
                print(f"[data-loader] skipping file due to error: {path} ({exc})")
                last_error = exc
                block_valid = False
                break

            if cleaned is None:
                block_valid = False
                break

            frames.append(cleaned)
            chosen.append(path)
            start_times.append(cleaned.index[0])

        if not block_valid or len(frames) != sample_k:
            continue

        contiguous = True
        for prev, cur in zip(start_times, start_times[1:]):
            gap = cur - prev
            if gap <= pd.Timedelta(0) or gap > MAX_FILE_START_GAP:
                print(f"[data-loader] block rejected due to gap between {prev} and {cur}")
                contiguous = False
                break

        if not contiguous:
            continue

        combined = pd.concat(frames, axis=0).sort_index()
        combined = combined[~combined.index.duplicated(keep="last")]
        combined = combined.dropna(subset=list(REQUIRED_COLUMNS))
        return combined, chosen

    if last_error is not None:
        raise last_error

    raise ValueError("No contiguous CSV block found that satisfies sampling constraints")

