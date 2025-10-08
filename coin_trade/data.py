from __future__ import annotations

import random
from pathlib import Path
from typing import Callable

import pandas as pd

from coin_trade.config import MAX_FILE_GAP_DAYS, PRICE_COLUMNS
from coin_trade.utils.dataframe import (
    clean_price_frame,
    detect_datetime_column,
    ensure_required_columns,
    normalise_columns,
    set_datetime_index,
    validate_frequency,
)

MAX_FILE_START_GAP = pd.Timedelta(days=MAX_FILE_GAP_DAYS)


def _read_csv(path: Path) -> pd.DataFrame:
    try:
        return pd.read_csv(path)
    except pd.errors.EmptyDataError:
        print(f"[data-loader] skipping empty file: {path}")
        return pd.DataFrame()


def _prepare_frame(frame: pd.DataFrame, *, min_rows: int) -> pd.DataFrame | None:
    if frame.empty:
        return None

    normalise_columns(frame)
    dt_col = detect_datetime_column(frame)
    frame = set_datetime_index(frame, dt_col)
    ensure_required_columns(frame, PRICE_COLUMNS)
    validate_frequency(frame.index)
    cleaned = clean_price_frame(frame, PRICE_COLUMNS)
    if cleaned.empty:
        return None
    if len(cleaned) < min_rows:
        return None
    return cleaned


def load_random_data(
    data_dir: str = "data",
    *,
    sample_k: int = 1,
    random_seed: int | None = None,
    min_rows: int = 60,
    max_attempts: int = 10,
    reader: Callable[[Path], pd.DataFrame] = _read_csv,
) -> tuple[pd.DataFrame, list[Path]]:
    """Randomly select a contiguous block of CSVs and return combined price data."""
    if sample_k < 1:
        raise ValueError("sample_k must be >= 1")
    if min_rows < 1:
        raise ValueError("min_rows must be >= 1")

    directory = Path(data_dir)
    if not directory.exists() or not directory.is_dir():
        raise FileNotFoundError(f"Data directory not found: {directory}")

    csv_files = sorted(
        p
        for p in directory.rglob("*.csv")
        if p.is_file() and p.stat().st_size > 200
    )
    if not csv_files:
        raise FileNotFoundError(f"No CSV files found in {directory}")

    if sample_k > len(csv_files):
        raise ValueError(
            f"Requested sample_k={sample_k} but only {len(csv_files)} file(s) available"
        )

    rng = random.Random(random_seed)
    if random_seed is not None:
        random.seed(random_seed)

    start_indices = list(range(len(csv_files) - sample_k + 1))
    rng.shuffle(start_indices)

    last_error: Exception | None = None

    attempts = 0
    for start in start_indices or [0]:
        attempts += 1
        if max_attempts > 0 and attempts > max_attempts:
            break

        block_paths = csv_files[start : start + sample_k]
        joined = ", ".join(str(p) for p in block_paths)
        print(f"[data-loader] sampled block (attempt {attempts}): {joined}")

        frames: list[pd.DataFrame] = []
        chosen: list[Path] = []
        start_times: list[pd.Timestamp] = []
        block_valid = True

        for path in block_paths:
            try:
                raw = reader(path)
            except ValueError as exc:
                print(f"[data-loader] skipping file due to error: {path} ({exc})")
                last_error = exc
                block_valid = False
                break

            prepared = _prepare_frame(raw, min_rows=min_rows)
            if prepared is None:
                print(f"[data-loader] skipping file with no usable rows: {path}")
                block_valid = False
                break

            frames.append(prepared)
            chosen.append(path)
            start_times.append(prepared.index[0])

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
        combined = combined.dropna(subset=list(PRICE_COLUMNS))
        return combined, chosen

    if last_error is not None:
        raise last_error

    raise ValueError("No contiguous CSV block found that satisfies sampling constraints")
