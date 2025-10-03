
from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from typing import Literal, Optional

import numpy as np
import pandas as pd

TradeExitPriority = Literal["tp", "sl"]


@dataclass
class TradeBuilderConfig:
    priority: TradeExitPriority = "tp"
    fee_rate: float = 0.0005
    slippage_rate: float = 0.0


def _resolve_entry_price(entry_time: pd.Timestamp, entry_df: pd.DataFrame) -> float:
    if "open" not in entry_df.columns:
        raise ValueError("entry_df must contain an 'open' column.")
    if not isinstance(entry_df.index, pd.DatetimeIndex):
        raise ValueError("entry_df must have a DatetimeIndex when provided.")

    pos = entry_df.index.searchsorted(entry_time)
    if pos >= len(entry_df):
        raise ValueError(f"Entry data does not contain prices for {entry_time} or later.")
    return float(entry_df.iloc[pos]["open"])


def build_trades(
    df: pd.DataFrame,
    *,
    signal_col: str = "buy_signal",
    tp_col: str = "tp_pct",
    sl_col: str = "sl_pct",
    priority: TradeExitPriority = "tp",
    fee_rate: float = 0.0005,
    slippage_rate: float = 0.0,
    entry_df: Optional[pd.DataFrame] = None,
    max_stopouts_per_day: Optional[int] = None,
    daily_loss_cap_pct: Optional[float] = None,
    time_stop_bars: Optional[int] = None,
) -> pd.DataFrame:
    """Construct trades using next-bar open entries with risk controls."""
    if priority not in ("tp", "sl"):
        raise ValueError("priority must be 'tp' or 'sl'")
    if not isinstance(df.index, pd.DatetimeIndex):
        raise ValueError("Input DataFrame must use a DatetimeIndex.")

    df = df.sort_index()
    if entry_df is not None:
        entry_df = entry_df.sort_index()

    signals = df[signal_col].fillna(False).to_numpy(dtype=bool)
    highs = df["high"].to_numpy(float)
    lows = df["low"].to_numpy(float)
    opens = df["open"].to_numpy(float)
    closes = df["close"].to_numpy(float)
    tp_pcts = df[tp_col].to_numpy(float)
    sl_pcts = df[sl_col].to_numpy(float)

    n = len(df)
    trade_list: list[dict[str, object]] = []
    last_exit_idx = -1
    stopouts_by_day: defaultdict[object, int] = defaultdict(int)
    pnl_by_day: defaultdict[object, float] = defaultdict(float)

    signal_indices = np.flatnonzero(signals)
    trade_id = 0

    for sig_idx in signal_indices:
        entry_idx = sig_idx + 1
        if entry_idx >= n or entry_idx <= last_exit_idx:
            continue

        tp_pct = tp_pcts[sig_idx]
        sl_pct = sl_pcts[sig_idx]
        if not np.isfinite(tp_pct) or not np.isfinite(sl_pct):
            continue

        entry_time = df.index[entry_idx]
        entry_day = entry_time.date()

        if max_stopouts_per_day is not None and stopouts_by_day[entry_day] >= max_stopouts_per_day:
            continue
        if daily_loss_cap_pct is not None and pnl_by_day[entry_day] <= -abs(daily_loss_cap_pct):
            continue

        if entry_df is None:
            entry_price = float(opens[entry_idx])
        else:
            entry_price = _resolve_entry_price(entry_time, entry_df)

        tp_price = entry_price * (1.0 + tp_pct)
        sl_price = entry_price * (1.0 - sl_pct)

        max_exit_idx = n - 1
        if time_stop_bars is not None:
            max_exit_idx = min(max_exit_idx, entry_idx + max(time_stop_bars, 0))

        exit_idx = None
        exit_price = None
        outcome = "open"

        for j in range(entry_idx, max_exit_idx + 1):
            hit_tp = highs[j] >= tp_price
            hit_sl = lows[j] <= sl_price

            if hit_tp and hit_sl:
                if priority == "tp":
                    exit_idx, exit_price, outcome = j, tp_price, "tp"
                else:
                    exit_idx, exit_price, outcome = j, sl_price, "sl"
                break
            if hit_tp:
                exit_idx, exit_price, outcome = j, tp_price, "tp"
                break
            if hit_sl:
                exit_idx, exit_price, outcome = j, sl_price, "sl"
                break

        if exit_idx is None:
            exit_idx = max_exit_idx
            exit_price = float(closes[exit_idx])
            outcome = "time" if time_stop_bars is not None else "open"

        holding_bars = exit_idx - entry_idx + 1
        gross_return_pct = exit_price / entry_price - 1.0

        entry_effective = entry_price * (1.0 + fee_rate + slippage_rate)
        exit_effective = exit_price * (1.0 - fee_rate - slippage_rate)
        net_return_pct = exit_effective / entry_effective - 1.0
        cost_pct = gross_return_pct - net_return_pct

        trade_list.append(
            {
                "trade_id": trade_id,
                "signal_idx": int(sig_idx),
                "entry_idx": int(entry_idx),
                "exit_idx": int(exit_idx),
                "signal_time": df.index[sig_idx],
                "entry_time": entry_time,
                "exit_time": df.index[exit_idx],
                "entry_price": float(entry_price),
                "exit_price": float(exit_price),
                "tp_price": float(tp_price),
                "sl_price": float(sl_price),
                "tp_pct": float(tp_pct),
                "sl_pct": float(sl_pct),
                "gross_return_pct": float(gross_return_pct),
                "net_return_pct": float(net_return_pct),
                "cost_pct": float(cost_pct),
                "outcome": outcome,
                "holding_bars": int(holding_bars),
            }
        )

        pnl_by_day[entry_day] += float(net_return_pct)
        if outcome == "sl":
            stopouts_by_day[entry_day] += 1

        last_exit_idx = exit_idx
        trade_id += 1

    return pd.DataFrame(trade_list)
