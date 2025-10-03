from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Literal, Optional

import numpy as np
import pandas as pd

from .indicators import (
    adx,
    atr,
    bollinger_bands,
    ema,
    keltner_channel,
    pct_rank,
    rsi,
)

Mode = Literal["production", "research"]


@dataclass
class StrategyParams:
    """Configuration for the signal engine."""

    bb_window: int = 20
    bb_k: float = 2.0
    kc_window: int = 20
    kc_mult: float = 1.5
    ema_trend_len: int = 34
    ema_long_len: int = 96
    ema_slope_lookback: int = 8
    adx_window: int = 14
    atr_window: int = 14
    squeeze_window: int = 120
    squeeze_pct: float = 0.40
    adx_threshold: float = 22.0
    adx_rr_threshold: float = 28.0
    vol_lookback: int = 20
    volume_multiplier: float = 1.2
    cooldown_bars: int = 4
    max_trades_per_day: int = 20
    enable_mean_reversion: bool = True
    rsi_window: int = 14
    rsi_buy_threshold: float = 28.0
    enable_low_risk_reversion: bool = True
    rsi_low_risk_threshold: float = 35.0
    low_risk_kc_quantile: float = 0.35
    atr_ema_window: int = 48
    fee_rate: float = 0.0005
    slippage_buffer: float = 0.0
    target_sl_low: float = 0.008
    target_sl_high: float = 0.013
    rr_low: float = 1.8
    rr_high: float = 2.4
    max_sl_pct: float = 0.03
    max_tp_pct: float = 0.06
    daily_return_lookback: int = 96
    daily_return_floor: float = -0.015
    daily_return_floor_reversion: float = -0.025
    max_stopouts_per_day: int = 2
    daily_loss_cap_pct: float = 0.02
    time_stop_bars: int = 96


@dataclass
class AutoThrottleConfig:
    target_trades_per_day_low: float
    target_trades_per_day_high: float
    window_days: int = 5
    cooldown_step: int = 1
    per_day_step: int = 2
    adx_step: float = 2.0
    squeeze_step: float = 0.05
    volume_step: float = 0.1
    min_cooldown: int = 1
    max_cooldown: int = 12
    min_per_day: int = 2
    max_per_day: int = 40
    bias_floor: int = -1
    bias_ceiling: int = 1


DEFAULT_THROTTLE: Dict[Mode, AutoThrottleConfig] = {
    "production": AutoThrottleConfig(
        target_trades_per_day_low=6.0,
        target_trades_per_day_high=10.0,
        window_days=5,
    ),
    "research": AutoThrottleConfig(
        target_trades_per_day_low=12.0,
        target_trades_per_day_high=32.0,
        window_days=7,
        max_per_day=60,
        cooldown_step=1,
        per_day_step=3,
    ),
}


@dataclass
class ThrottleLimits:
    cooldown: int
    daily_cap: int
    bias: int


class AutoThrottle:
    """Adaptive throttle that nudges trade frequency toward a target band."""

    def __init__(
        self,
        base_cooldown: int,
        base_daily_cap: int,
        config: AutoThrottleConfig,
    ) -> None:
        self.base_cooldown = max(0, base_cooldown)
        self.base_daily_cap = max(1, base_daily_cap)
        self.config = config

        self.current_cooldown = self.base_cooldown
        self.current_daily_cap = self.base_daily_cap
        self.current_bias = 0
        self._executions: list[pd.Timestamp] = []

    def limits(self, now: pd.Timestamp) -> ThrottleLimits:
        cutoff = now - pd.Timedelta(days=self.config.window_days)
        self._executions = [ts for ts in self._executions if ts >= cutoff]

        pace = 0.0
        if self._executions:
            unique_days = {ts.date() for ts in self._executions}
            days_span = max((now.date() - min(unique_days)).days + 1, 1)
            pace = len(self._executions) / days_span

        tolerance = 0.2
        if pace + tolerance < self.config.target_trades_per_day_low:
            self.current_cooldown = max(self.config.min_cooldown, self.current_cooldown - self.config.cooldown_step)
            self.current_daily_cap = min(self.config.max_per_day, self.current_daily_cap + self.config.per_day_step)
            self.current_bias = max(self.config.bias_floor, self.current_bias - 1)
        elif pace - tolerance > self.config.target_trades_per_day_high:
            self.current_cooldown = min(self.config.max_cooldown, self.current_cooldown + self.config.cooldown_step)
            self.current_daily_cap = max(self.config.min_per_day, self.current_daily_cap - self.config.per_day_step)
            self.current_bias = min(self.config.bias_ceiling, self.current_bias + 1)
        else:
            self._mean_revert_limits()
            self.current_bias = 0

        return ThrottleLimits(
            cooldown=self.current_cooldown,
            daily_cap=self.current_daily_cap,
            bias=self.current_bias,
        )

    def record(self, ts: pd.Timestamp) -> None:
        self._executions.append(ts)

    def _mean_revert_limits(self) -> None:
        if self.current_cooldown < self.base_cooldown:
            self.current_cooldown = min(self.base_cooldown, self.current_cooldown + self.config.cooldown_step)
        elif self.current_cooldown > self.base_cooldown:
            self.current_cooldown = max(self.base_cooldown, self.current_cooldown - self.config.cooldown_step)

        if self.current_daily_cap < self.base_daily_cap:
            self.current_daily_cap = min(self.base_daily_cap, self.current_daily_cap + self.config.per_day_step)
        elif self.current_daily_cap > self.base_daily_cap:
            self.current_daily_cap = max(self.base_daily_cap, self.current_daily_cap - self.config.per_day_step)


def _warmup_bars(params: StrategyParams) -> int:
    return max(
        params.bb_window,
        params.kc_window,
        params.ema_trend_len,
        params.ema_long_len,
        params.adx_window,
        params.atr_window,
        params.squeeze_window,
        params.vol_lookback,
        params.rsi_window if params.enable_mean_reversion or params.enable_low_risk_reversion else 1,
        params.atr_ema_window,
        params.daily_return_lookback,
    )


def generate_signals(
    df: pd.DataFrame,
    params: Optional[StrategyParams] = None,
    *,
    mode: Mode = "production",
    throttle_config: Optional[AutoThrottleConfig] = None,
) -> pd.DataFrame:
    """Compute strategy signals while keeping legacy signal columns."""
    if params is None:
        params = StrategyParams()
    if throttle_config is None:
        throttle_config = DEFAULT_THROTTLE[mode]

    df = df.sort_index().copy()

    ema_trend = ema(df["close"], span=params.ema_trend_len)
    atr_series = atr(df, window=params.atr_window)
    adx_series = adx(df, window=params.adx_window)
    bb_mid, bb_up, bb_lo = bollinger_bands(df["close"], window=params.bb_window, num_std=params.bb_k)
    kc_mid, kc_up, kc_lo = keltner_channel(df, window=params.kc_window, multiplier=params.kc_mult)

    width = (bb_up - bb_lo) / bb_mid.replace(0, np.nan)
    squeeze_rank = pct_rank(width, window=params.squeeze_window)
    in_kc = (bb_up < kc_up) & (bb_lo > kc_lo)

    volume_mean = df["volume"].rolling(window=params.vol_lookback, min_periods=params.vol_lookback).mean()
    atr_pct = atr_series / df["close"].replace(0, np.nan)
    atr_pct = atr_pct.fillna(method="ffill")
    atr_pct_ema = ema(atr_pct.fillna(atr_pct.expanding().mean()), span=params.atr_ema_window)

    ema_long = ema(df["close"], span=params.ema_long_len)
    ema_long_slope = ema_long - ema_long.shift(params.ema_slope_lookback)
    ema_long_slope_pct = ema_long_slope / ema_long.replace(0, np.nan)
    daily_return = df["close"] / df["close"].shift(params.daily_return_lookback) - 1
    daily_return = daily_return.fillna(0.0)

    if params.enable_mean_reversion or params.enable_low_risk_reversion:
        rsi_series = rsi(df["close"], window=params.rsi_window)
    else:
        rsi_series = pd.Series(np.nan, index=df.index)

    rr_base = np.where(adx_series >= params.adx_rr_threshold, params.rr_high, params.rr_low)
    slope_norm = ema_long_slope_pct.fillna(0.0)
    rr_dynamic = np.where(
        slope_norm >= 0.003,
        params.rr_high + 0.4,
        np.where(
            slope_norm >= 0.0015,
            params.rr_high,
            np.where(
                slope_norm <= -0.001,
                np.maximum(params.rr_low - 0.4, 1.05),
                rr_base
            )
        )
    )
    rr_dynamic = np.clip(rr_dynamic, 1.05, params.rr_high + 0.6)

    rev_rr = np.where(
        slope_norm <= -0.0015,
        0.9,
        np.where(
            slope_norm >= 0.0025,
            1.6,
            np.where(
                slope_norm >= 0.0015,
                1.4,
                1.1
            )
        )
    )

    target_mid = (params.target_sl_low + params.target_sl_high) / 2
    base_sl = atr_pct_ema.replace(0, np.nan)
    safe_base = base_sl.where(base_sl > 1e-6, target_mid)
    scale = (target_mid / safe_base).clip(lower=0.5, upper=4.0)
    adaptive_sl = (safe_base * scale).clip(lower=params.target_sl_low, upper=params.target_sl_high).fillna(params.target_sl_high)

    buffer = 2 * (params.fee_rate + params.slippage_buffer)
    sl_break = (adaptive_sl * 1.05 + buffer).clip(upper=params.max_sl_pct)
    sl_rev = (adaptive_sl * 0.90 + buffer).clip(upper=params.max_sl_pct)

    tp_break = (sl_break * rr_dynamic + buffer).clip(upper=params.max_tp_pct)
    tp_rev = (sl_rev * rev_rr + buffer).clip(upper=params.max_tp_pct)

    n = len(df)
    buy_signal = np.zeros(n, dtype=bool)
    signal_type: list[Optional[str]] = [None] * n
    tp_pct = np.full(n, np.nan, dtype=float)
    sl_pct = np.full(n, np.nan, dtype=float)

    throttle = AutoThrottle(params.cooldown_bars, params.max_trades_per_day, throttle_config)
    last_signal_idx = -10_000
    daily_counts: Dict[pd.Timestamp, int] = {}

    warmup = _warmup_bars(params)
    for i in range(warmup, n):
        ts = df.index[i]
        limits = throttle.limits(ts)

        adx_thr_eff = max(5.0, params.adx_threshold + limits.bias * throttle_config.adx_step)
        squeeze_pct_eff = float(np.clip(params.squeeze_pct + (-limits.bias) * throttle_config.squeeze_step, 0.05, 0.95))
        vol_mult_eff = max(0.1, params.volume_multiplier * (1.0 + throttle_config.volume_step * limits.bias))

        daily_floor_eff = np.clip(params.daily_return_floor + limits.bias * 0.004, -0.03, 0.04)
        rev_floor_eff = np.clip(params.daily_return_floor_reversion + limits.bias * 0.004, -0.12, -0.001)
        rev_floor_eff = min(rev_floor_eff, -0.003)

        slope_val = float(ema_long_slope_pct.iloc[i]) if pd.notna(ema_long_slope_pct.iloc[i]) else np.nan
        trend_ok = (
            df["close"].iloc[i] >= ema_trend.iloc[i]
            and df["close"].iloc[i] >= ema_long.iloc[i]
            and np.isfinite(slope_val)
            and slope_val > -0.002
        )
        reversion_trend_ok = (
            df["close"].iloc[i] >= ema_trend.iloc[i] * 0.985
            and np.isfinite(slope_val)
            and -0.03 <= slope_val < 0.012
        )

        context_ok = daily_return.iloc[i] >= max(daily_floor_eff, -0.05)
        rev_context_ok = (daily_return.iloc[i] <= (rev_floor_eff + 0.01)) or (params.enable_mean_reversion and rsi_series.iloc[i] <= params.rsi_buy_threshold - 5)

        breakout = bool(
            in_kc.iloc[i]
            and squeeze_rank.iloc[i] <= squeeze_pct_eff
            and df["close"].iloc[i] > bb_up.iloc[i]
            and df["close"].iloc[i - 1] <= bb_up.iloc[i - 1]
            and trend_ok
            and context_ok
            and adx_series.iloc[i] >= adx_thr_eff
            and df["volume"].iloc[i] >= volume_mean.iloc[i] * vol_mult_eff
        )

        candidate_type: Optional[str] = None
        if breakout:
            candidate_type = "breakout"
        else:
            if (
                params.enable_mean_reversion
                and rsi_series.iloc[i] <= params.rsi_buy_threshold
                and reversion_trend_ok
                and rev_context_ok
            ):
                candidate_type = "reversion"
            if (
                params.enable_low_risk_reversion
                and rsi_series.iloc[i] <= params.rsi_low_risk_threshold
                and reversion_trend_ok
                and rev_context_ok
                and df["close"].iloc[i] <= kc_mid.iloc[i] - params.low_risk_kc_quantile * (kc_mid.iloc[i] - kc_lo.iloc[i])
            ):
                candidate_type = "reversion_low"

        if candidate_type is None:
            continue

        if last_signal_idx >= 0 and (i - last_signal_idx) <= limits.cooldown:
            continue

        day = pd.Timestamp(ts.date())
        daily_counts.setdefault(day, 0)
        if daily_counts[day] >= limits.daily_cap:
            continue

        buy_signal[i] = True
        signal_type[i] = candidate_type
        last_signal_idx = i
        daily_counts[day] += 1
        throttle.record(ts)

        if candidate_type == "breakout":
            tp_pct[i] = tp_break.iloc[i]
            sl_pct[i] = sl_break.iloc[i]
        elif candidate_type == "reversion":
            tp_pct[i] = tp_rev.iloc[i]
            sl_pct[i] = sl_rev.iloc[i]
        else:
            tp_val = float(tp_rev.iloc[i] * 0.9)
            tp_pct[i] = max(params.target_sl_low, min(tp_val, params.max_tp_pct))
            sl_pct[i] = sl_rev.iloc[i]

    df["buy_signal"] = buy_signal
    df["signal_type"] = pd.Series(signal_type, index=df.index).replace({None: np.nan})
    df["tp_pct"] = np.where(buy_signal, tp_pct, np.nan)
    df["sl_pct"] = np.where(buy_signal, sl_pct, np.nan)

    df["EMA_trend"] = ema_trend
    df["EMA_long"] = ema_long
    df["ATR"] = atr_series
    df["ADX"] = adx_series
    df["daily_return"] = daily_return

    return df
