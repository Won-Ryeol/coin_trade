from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Optional

import numpy as np
import pandas as pd

from .indicators import adx, atr, ema

RegimeLabel = Literal["trend", "range"]


@dataclass
class RegimeConfig:
    """Parameters controlling regime classification."""

    enable: bool = True
    default_regime: RegimeLabel = "trend"
    adx_window: int = 14
    adx_trend_threshold: float = 27.0
    adx_range_threshold: float = 20.0
    slope_ema_span: int = 96
    slope_lookback: int = 8
    slope_trend_threshold: float = 0.0015
    slope_range_threshold: float = 0.0006
    vol_fast_window: int = 20
    vol_slow_window: int = 60
    vol_ratio_trend: float = 1.10
    vol_ratio_range: float = 0.95
    min_trend_votes: int = 2
    min_range_votes: int = 2
    switch_cooldown: int = 12


@dataclass
class RegimeResult:
    labels: pd.Series
    trend_votes: pd.Series
    range_votes: pd.Series
    features: pd.DataFrame


def _normalize_default(label: str) -> RegimeLabel:
    return label if label in ("trend", "range") else "trend"


def detect_regime(
    df: pd.DataFrame,
    config: RegimeConfig,
    *,
    adx_series: Optional[pd.Series] = None,
    ema_long: Optional[pd.Series] = None,
) -> RegimeResult:
    """Classify each bar as trend or range regime using hysteresis."""
    index = df.index
    if len(index) == 0:
        empty = pd.Series(dtype="object", index=index)
        zeros = pd.Series(data=[], index=index, dtype=int)
        features = pd.DataFrame(
            {
                "adx": pd.Series(dtype=float, index=index),
                "ema_slope": pd.Series(dtype=float, index=index),
                "abs_ema_slope": pd.Series(dtype=float, index=index),
                "vol_ratio": pd.Series(dtype=float, index=index),
            }
        )
        return RegimeResult(empty, zeros, zeros, features)

    default_label = _normalize_default(config.default_regime)

    adx_src = adx_series if adx_series is not None else adx(df, window=config.adx_window)
    adx_src = adx_src.reindex(index).fillna(method="ffill").fillna(method="bfill").fillna(0.0)

    ema_base = ema_long if ema_long is not None else ema(df["close"], span=config.slope_ema_span)
    ema_base = ema_base.reindex(index).fillna(method="ffill").fillna(method="bfill").fillna(0.0)
    slope = ema_base.pct_change(config.slope_lookback).replace([np.inf, -np.inf], np.nan).fillna(0.0)
    abs_slope = slope.abs()

    atr_fast = atr(df, window=config.vol_fast_window).reindex(index)
    atr_slow = atr(df, window=config.vol_slow_window).reindex(index)
    vol_ratio = (atr_fast / (atr_slow + 1e-12)).replace([np.inf, -np.inf], np.nan).fillna(1.0)

    trend_votes = (
        (adx_src >= config.adx_trend_threshold).astype(int)
        + (abs_slope >= config.slope_trend_threshold).astype(int)
        + (vol_ratio >= config.vol_ratio_trend).astype(int)
    ).astype(int)
    range_votes = (
        (adx_src <= config.adx_range_threshold).astype(int)
        + (abs_slope <= config.slope_range_threshold).astype(int)
        + (vol_ratio <= config.vol_ratio_range).astype(int)
    ).astype(int)

    features = pd.DataFrame(
        {
            "adx": adx_src,
            "ema_slope": slope,
            "abs_ema_slope": abs_slope,
            "vol_ratio": vol_ratio,
        },
        index=index,
    )

    if not config.enable:
        labels = pd.Series(default_label, index=index, dtype="object")
        zeros = pd.Series(0, index=index, dtype=int)
        return RegimeResult(labels, zeros, zeros, features)

    current = default_label
    if trend_votes.iloc[0] > range_votes.iloc[0]:
        current = "trend"
    elif range_votes.iloc[0] > trend_votes.iloc[0]:
        current = "range"

    cooldown_max = max(int(config.switch_cooldown), 0)
    trend_gate = max(int(config.min_trend_votes), 1)
    range_gate = max(int(config.min_range_votes), 1)
    cooldown = 0
    labels_list: list[str] = []

    for tv, rv in zip(trend_votes, range_votes):
        if cooldown > 0:
            cooldown -= 1
        if current == "trend":
            if rv >= range_gate and cooldown == 0:
                current = "range"
                cooldown = cooldown_max
        else:
            if tv >= trend_gate and cooldown == 0:
                current = "trend"
                cooldown = cooldown_max
        labels_list.append(current)

    labels = pd.Series(labels_list, index=index, dtype="object")
    return RegimeResult(labels, trend_votes, range_votes, features)
