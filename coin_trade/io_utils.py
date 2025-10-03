from __future__ import annotations

import json
from pathlib import Path
from typing import Optional

import pandas as pd

from coin_trade.backtester import BacktestResult

REQUIRED_COLUMNS = {"open", "high", "low", "close", "volume"}


def load_price_data(path: str) -> pd.DataFrame:
    data_path = Path(path)
    if not data_path.exists():
        raise FileNotFoundError(path)

    if data_path.suffix.lower() == ".csv":
        df = pd.read_csv(data_path)
    elif data_path.suffix.lower() in {".parquet", ".pq"}:
        df = pd.read_parquet(data_path)
    else:
        raise ValueError("Data file must be CSV or Parquet")

    df.columns = [c.lower() for c in df.columns]

    if isinstance(df.index, pd.DatetimeIndex):
        pass
    elif "timestamp" in df.columns:
        df["timestamp"] = pd.to_datetime(df["timestamp"], utc=False)
        df = df.set_index("timestamp")
    elif "datetime" in df.columns:
        df["datetime"] = pd.to_datetime(df["datetime"], utc=False)
        df = df.set_index("datetime")
    else:
        first_col = df.columns[0]
        df[first_col] = pd.to_datetime(df[first_col], utc=False)
        df = df.set_index(first_col)

    missing = REQUIRED_COLUMNS - set(df.columns)
    if missing:
        raise ValueError(f"Data file missing required columns: {missing}")

    return df.sort_index()


def maybe_filter(df: pd.DataFrame, start: Optional[str], end: Optional[str]) -> pd.DataFrame:
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
