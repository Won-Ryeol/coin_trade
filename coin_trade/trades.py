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
    enable_partial_take_profit: bool = False
    partial_tp_fraction: float = 0.0
    partial_tp_rr: float = 1.0
    breakeven_buffer_pct: float = 0.0
    enable_trailing_stop: bool = False
    trailing_stop_activation_rr: float = 1.0
    trailing_stop_atr_multiple: float = 2.5


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
    enable_partial_take_profit: bool = False,
    partial_tp_fraction: float = 0.0,
    partial_tp_rr: float = 1.0,
    breakeven_buffer_pct: float = 0.0,
    enable_trailing_stop: bool = False,
    trailing_stop_activation_rr: float = 1.0,
    trailing_stop_atr_multiple: float = 2.5,
) -> pd.DataFrame:
    """Construct trades using next-bar open entries with optional partials and trailing stops."""
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
    if "ATR" in df.columns:
        atr_values = df["ATR"].to_numpy(dtype=float)
    else:
        atr_values = np.full(len(df), np.nan, dtype=float)

    n = len(df)
    trade_list: list[dict[str, object]] = []
    last_exit_idx = -1
    stopouts_by_day: defaultdict[object, int] = defaultdict(int)
    pnl_by_day: defaultdict[object, float] = defaultdict(float)

    signal_indices = np.flatnonzero(signals)
    trade_id = 0

    base_partial_fraction = float(np.clip(partial_tp_fraction, 0.0, 1.0))
    base_partial_enabled = enable_partial_take_profit and base_partial_fraction > 0.0
    base_trailing_enabled = enable_trailing_stop and trailing_stop_atr_multiple > 0
    base_partial_rr = float(max(partial_tp_rr, 0.0))
    base_trailing_rr = float(max(trailing_stop_activation_rr, 0.0))

    for sig_idx in signal_indices:
        entry_idx = sig_idx + 1
        if entry_idx >= n or entry_idx <= last_exit_idx:
            continue

        tp_pct = tp_pcts[sig_idx]
        sl_pct = sl_pcts[sig_idx]
        if not np.isfinite(tp_pct) or not np.isfinite(sl_pct):
            continue

        sig_type = df.iloc[sig_idx].get("signal_type") if "signal_type" in df.columns else None

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

        exit_idx: Optional[int] = None
        exit_price: Optional[float] = None
        exit_reason = "open"

        current_sl_price = float(sl_price)
        stop_source = "initial"
        risk_pct = float(sl_pct)
        partial_enabled = base_partial_enabled and risk_pct > 0
        trailing_enabled = base_trailing_enabled and risk_pct > 0
        partial_fraction = base_partial_fraction if partial_enabled else 0.0
        partial_price = entry_price * (1.0 + risk_pct * base_partial_rr) if partial_enabled else np.nan
        breakeven_price = entry_price * (1.0 + breakeven_buffer_pct)
        trailing_activation_price = entry_price * (1.0 + risk_pct * base_trailing_rr) if trailing_enabled else np.nan

        partial_done = False
        partial_exit_idx = np.nan
        partial_exit_price = np.nan
        partial_exit_time = pd.NaT
        partial_fraction_realized = 0.0
        position_fraction = 1.0

        trail_active = False
        trail_started = False
        event_log: list[str] = []

        for j in range(entry_idx, max_exit_idx + 1):
            current_time = df.index[j]
            bar_high = highs[j]
            bar_low = lows[j]

            if trailing_enabled and not trail_active:
                activation_price = trailing_activation_price if np.isfinite(trailing_activation_price) else tp_price
                if bar_high >= activation_price:
                    trail_active = True
                    trail_started = True
                    event_log.append(f"trail_on@{current_time.isoformat()}")

            if partial_enabled and (not partial_done) and bar_high >= partial_price:
                partial_done = True
                partial_exit_idx = float(j)
                partial_exit_price = float(partial_price)
                partial_exit_time = current_time
                partial_fraction_realized = partial_fraction
                position_fraction = max(0.0, position_fraction - partial_fraction)
                event_log.append(f"partial@{current_time.isoformat()}:{partial_fraction_realized:.4f}")

                be_target = max(current_sl_price, breakeven_price)
                if be_target > current_sl_price + 1e-12:
                    current_sl_price = be_target
                    stop_source = "breakeven"
                    event_log.append(f"breakeven@{current_sl_price:.8f}")

                if trailing_enabled:
                    trail_active = True
                    trail_started = True

                if position_fraction <= 1e-6:
                    exit_idx = j
                    exit_price = float(partial_price)
                    exit_reason = "tp"
                    break

            if trail_active and trailing_enabled:
                atr_val = atr_values[j]
                if np.isfinite(atr_val) and atr_val > 0:
                    candidate = bar_high - trailing_stop_atr_multiple * atr_val
                    candidate = min(candidate, bar_high)
                    if candidate > current_sl_price + 1e-12:
                        current_sl_price = candidate
                        stop_source = "trailing"
                        event_log.append(f"trail@{current_time.isoformat()}:{current_sl_price:.8f}")

            hit_tp = bar_high >= tp_price
            hit_sl = bar_low <= current_sl_price

            if hit_tp and hit_sl:
                if priority == "tp":
                    exit_idx, exit_price, exit_reason = j, float(tp_price), "tp"
                else:
                    exit_idx, exit_price = j, float(current_sl_price)
                    exit_reason = "sl" if stop_source == "initial" else ("breakeven" if stop_source == "breakeven" else "trail")
                break
            if hit_tp:
                exit_idx, exit_price, exit_reason = j, float(tp_price), "tp"
                break
            if hit_sl:
                exit_idx, exit_price = j, float(current_sl_price)
                if stop_source == "initial":
                    exit_reason = "sl"
                elif stop_source == "breakeven":
                    exit_reason = "breakeven"
                else:
                    exit_reason = "trail"
                break

        if exit_idx is None:
            exit_idx = max_exit_idx
            exit_price = float(closes[exit_idx])
            exit_reason = "time" if time_stop_bars is not None else "open"

        holding_bars = exit_idx - entry_idx + 1
        entry_effective = entry_price * (1.0 + fee_rate + slippage_rate)

        legs: list[tuple[float, float]] = []
        if partial_fraction_realized > 0 and np.isfinite(partial_exit_price):
            legs.append((partial_fraction_realized, float(partial_exit_price)))
        remaining_fraction = max(0.0, 1.0 - partial_fraction_realized)
        if remaining_fraction > 0:
            legs.append((remaining_fraction, float(exit_price)))

        gross_return_pct = 0.0
        net_return_pct = 0.0
        for frac, price in legs:
            gross_leg = price / entry_price - 1.0
            net_leg = (price * (1.0 - fee_rate - slippage_rate)) / entry_effective - 1.0
            gross_return_pct += frac * gross_leg
            net_return_pct += frac * net_leg
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
                "outcome": exit_reason,
                "holding_bars": int(holding_bars),
                "signal_type": sig_type,
                "partial_exit_idx": float(partial_exit_idx) if np.isfinite(partial_exit_idx) else np.nan,
                "partial_exit_time": partial_exit_time,
                "partial_exit_price": float(partial_exit_price) if np.isfinite(partial_exit_price) else np.nan,
                "partial_fraction": float(partial_fraction_realized),
                "final_stop_price": float(current_sl_price),
                "final_stop_source": stop_source,
                "trailing_active": bool(trail_started),
                "event_log": "|".join(event_log),
            }
        )

        pnl_by_day[entry_day] += float(net_return_pct)
        if exit_reason == "sl":
            stopouts_by_day[entry_day] += 1

        last_exit_idx = exit_idx
        trade_id += 1

    return pd.DataFrame(trade_list)


