from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from coin_trade.backtester import BacktestResult
from coin_trade.config import PRICE_COLUMNS
from coin_trade.utils.dataframe import (
    clean_price_frame,
    detect_datetime_column,
    ensure_required_columns,
    normalise_columns,
    set_datetime_index,
)


def load_price_data(path: str) -> pd.DataFrame:
    data_path = Path(path)
    if not data_path.exists():
        raise FileNotFoundError(path)

    if data_path.suffix.lower() == ".csv":
        frame = pd.read_csv(data_path)
    elif data_path.suffix.lower() in {".parquet", ".pq"}:
        frame = pd.read_parquet(data_path)
    else:
        raise ValueError("Data file must be CSV or Parquet")

    normalise_columns(frame)

    if isinstance(frame.index, pd.DatetimeIndex):
        processed = frame
    else:
        dt_col = detect_datetime_column(frame)
        processed = set_datetime_index(frame, dt_col)

    ensure_required_columns(processed, PRICE_COLUMNS)
    processed = clean_price_frame(processed, PRICE_COLUMNS)
    return processed


def maybe_filter(df: pd.DataFrame, start: str | None, end: str | None) -> pd.DataFrame:
    if start:
        df = df[df.index >= pd.to_datetime(start)]
    if end:
        df = df[df.index <= pd.to_datetime(end)]
    return df


def ensure_output_dir(path: str) -> Path:
    output = Path(path)
    output.mkdir(parents=True, exist_ok=True)
    return output


def save_artifacts(result: BacktestResult, output_dir: Path, *, prefix: str = "") -> None:
    trades_path = output_dir / f"{prefix}trades.csv"
    equity_path = output_dir / f"{prefix}equity.csv"
    metrics_path = output_dir / f"{prefix}metrics.json"

    result.trades.to_csv(trades_path, index=False)
    result.equity.to_csv(equity_path, index=True)
    metrics_path.write_text(json.dumps(result.metrics.metrics, indent=2), encoding="utf-8")
