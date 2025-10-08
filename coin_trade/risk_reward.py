from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
import pandas as pd

__all__ = [
    "RiskRewardSettings",
    "RiskRewardTargets",
    "compute_risk_targets",
]


@dataclass(frozen=True)
class RiskRewardSettings:
    target_sl_low: float
    target_sl_high: float
    max_sl_pct: float
    max_tp_pct: float
    rr_low: float
    rr_high: float
    rr_min: float
    rr_max: float
    adx_rr_threshold: float
    breakout_slope_knots: Sequence[float]
    breakout_slope_values: Sequence[float]
    breakout_adx_knots: Sequence[float]
    breakout_adx_values: Sequence[float]
    breakout_blend: float
    reversion_slope_knots: Sequence[float]
    reversion_slope_values: Sequence[float]
    reversion_floor: float
    reversion_ceiling: float
    sl_scale_clip: tuple[float, float]
    atr_floor: float
    buffer_factor: float
    breakout_stop_multiplier: float
    reversion_stop_multiplier: float
    fee_rate: float
    slippage_buffer: float


@dataclass(frozen=True)
class RiskRewardTargets:
    rr_breakout: pd.Series
    rr_reversion: pd.Series
    sl_breakout: pd.Series
    sl_reversion: pd.Series
    tp_breakout: pd.Series
    tp_reversion: pd.Series


def _prepare_schedule(knots: Sequence[float], values: Sequence[float]) -> tuple[np.ndarray, np.ndarray]:
    if len(knots) != len(values):
        raise ValueError("Schedule knots and values must be the same length")

    pairs = sorted((float(k), float(v)) for k, v in zip(knots, values))
    if not pairs:
        raise ValueError("Schedule cannot be empty")

    dedup_knots: list[float] = []
    dedup_vals: list[float] = []
    for knot, val in pairs:
        if dedup_knots and abs(knot - dedup_knots[-1]) < 1e-12:
            dedup_knots[-1] = knot
            dedup_vals[-1] = val
        else:
            dedup_knots.append(knot)
            dedup_vals.append(val)
    return np.asarray(dedup_knots, dtype=float), np.asarray(dedup_vals, dtype=float)


def _interp(values: np.ndarray, knots: Sequence[float], targets: Sequence[float]) -> np.ndarray:
    knots_arr, targets_arr = _prepare_schedule(knots, targets)
    return np.interp(values, knots_arr, targets_arr, left=targets_arr[0], right=targets_arr[-1])


def compute_risk_targets(
    *,
    atr_pct_ema: pd.Series,
    slope: pd.Series,
    adx: pd.Series,
    settings: RiskRewardSettings,
) -> RiskRewardTargets:
    index = atr_pct_ema.index

    slope_series = slope.reindex(index).astype(float).fillna(0.0)
    adx_series = adx.reindex(index).astype(float).fillna(0.0)
    atr_series = atr_pct_ema.reindex(index).astype(float)

    slope_values = slope_series.to_numpy(dtype=float)
    adx_values = adx_series.to_numpy(dtype=float)
    atr_values = atr_series.to_numpy(dtype=float)

    breakout_slope_rr = _interp(slope_values, settings.breakout_slope_knots, settings.breakout_slope_values)
    breakout_adx_rr = _interp(adx_values, settings.breakout_adx_knots, settings.breakout_adx_values)

    blend = float(np.clip(settings.breakout_blend, 0.0, 1.0))
    rr_breakout = blend * breakout_slope_rr + (1.0 - blend) * breakout_adx_rr
    rr_breakout = np.maximum(rr_breakout, settings.rr_low)
    rr_breakout = np.where(
        adx_values >= settings.adx_rr_threshold,
        np.maximum(rr_breakout, settings.rr_high),
        rr_breakout,
    )
    rr_breakout = np.maximum(rr_breakout, breakout_adx_rr)
    rr_breakout = np.clip(rr_breakout, settings.rr_min, settings.rr_max)

    rr_reversion = _interp(slope_values, settings.reversion_slope_knots, settings.reversion_slope_values)
    rr_reversion = np.clip(rr_reversion, settings.reversion_floor, settings.reversion_ceiling)

    atr_floor = max(settings.atr_floor, 1e-9)
    mid_target = 0.5 * (settings.target_sl_low + settings.target_sl_high)
    atr_clean = np.where(
        np.isfinite(atr_values) & (atr_values > atr_floor),
        atr_values,
        mid_target,
    )

    scale_min, scale_max = settings.sl_scale_clip
    if scale_min <= 0 or scale_min > scale_max:
        raise ValueError("Invalid sl_scale_clip bounds")
    scale = np.clip(mid_target / atr_clean, scale_min, scale_max)
    adaptive_sl = np.clip(atr_clean * scale, settings.target_sl_low, settings.target_sl_high)

    buffer = max(settings.buffer_factor, 0.0) * max(settings.fee_rate + settings.slippage_buffer, 0.0)

    sl_breakout = np.clip(
        adaptive_sl * settings.breakout_stop_multiplier + buffer,
        settings.target_sl_low + buffer,
        settings.max_sl_pct,
    )
    sl_reversion = np.clip(
        adaptive_sl * settings.reversion_stop_multiplier + buffer,
        settings.target_sl_low * 0.5 + buffer,
        settings.max_sl_pct,
    )

    tp_breakout = np.clip(sl_breakout * rr_breakout + buffer, 0.0, settings.max_tp_pct)
    tp_reversion = np.clip(sl_reversion * rr_reversion + buffer, 0.0, settings.max_tp_pct)

    return RiskRewardTargets(
        rr_breakout=pd.Series(rr_breakout, index=index, name="rr_breakout"),
        rr_reversion=pd.Series(rr_reversion, index=index, name="rr_reversion"),
        sl_breakout=pd.Series(sl_breakout, index=index, name="sl_breakout"),
        sl_reversion=pd.Series(sl_reversion, index=index, name="sl_reversion"),
        tp_breakout=pd.Series(tp_breakout, index=index, name="tp_breakout"),
        tp_reversion=pd.Series(tp_reversion, index=index, name="tp_reversion"),
    )
