from __future__ import annotations

from collections.abc import Iterable

import numpy as np
import pandas as pd

from coin_trade.config import BAR_INTERVAL_MINUTES, PRICE_COLUMNS


def normalise_columns(frame: pd.DataFrame) -> pd.DataFrame:
    frame.columns = [str(c).lower() for c in frame.columns]
    return frame


def detect_datetime_column(frame: pd.DataFrame) -> str:
    for candidate in ("timestamp", "datetime", frame.columns[0]):
        if candidate in frame.columns:
            return candidate
    raise ValueError("No datetime column detected in dataset")


def set_datetime_index(frame: pd.DataFrame, column: str) -> pd.DataFrame:
    frame[column] = pd.to_datetime(frame[column], utc=False, errors="coerce")
    if frame[column].isna().any():
        raise ValueError("Datetime column contains invalid entries")
    frame = frame.set_index(column)
    return frame.sort_index()


def ensure_required_columns(frame: pd.DataFrame, required: Iterable[str] = PRICE_COLUMNS) -> None:
    missing = set(required) - set(frame.columns)
    if missing:
        raise ValueError(f"Data file missing required columns: {sorted(missing)}")


def validate_frequency(index: pd.DatetimeIndex, minutes: int = BAR_INTERVAL_MINUTES) -> None:
    if len(index) < 3:
        return
    diffs = index.to_series().diff().dropna()
    if diffs.empty:
        return
    values = diffs.dt.total_seconds() / 60.0
    if not np.isclose(values, minutes).any():
        raise ValueError(f"Dataset is not consistently {minutes}-minute cadence")
    mod = np.mod(values, float(minutes))
    valid = (mod < 1e-6) | ((minutes - mod) < 1e-6)
    if not valid.all():
        invalid = values[~valid]
        if len(invalid) > len(values) * 0.1:
            raise ValueError(f"Dataset is not consistently {minutes}-minute cadence")


def clean_price_frame(frame: pd.DataFrame, required: Iterable[str] = PRICE_COLUMNS) -> pd.DataFrame:
    frame = frame[~frame.index.duplicated(keep="last")]
    frame = frame.sort_index()
    frame = frame.dropna(subset=list(required))
    return frame
