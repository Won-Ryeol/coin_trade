from __future__ import annotations

from dataclasses import dataclass

PRICE_COLUMNS: tuple[str, ...] = ("open", "high", "low", "close", "volume")
BAR_INTERVAL_MINUTES = 15
MAX_FILE_GAP_DAYS = 3


@dataclass(frozen=True)
class ExecutionConfig:
    fee_rate: float = 0.0005
    slippage_rate: float = 0.0
    breakeven_buffer_pct: float = 0.0005


@dataclass(frozen=True)
class SignalWindowConfig:
    atr_window: int = 14
    adx_window: int = 14


DEFAULT_EXECUTION = ExecutionConfig()
DEFAULT_SIGNAL_WINDOWS = SignalWindowConfig()
