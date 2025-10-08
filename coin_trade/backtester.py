from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from .config import DEFAULT_EXECUTION
from .metrics import MetricResult, compute_equity_curve, compute_metrics
from .signals import AutoThrottleConfig, Mode, StrategyParams, generate_signals
from .trades import TradeExitPriority, build_trades


@dataclass
class BacktestResult:
    prices: pd.DataFrame
    signals: pd.DataFrame
    trades: pd.DataFrame
    equity: pd.DataFrame
    metrics: MetricResult


def run_backtest(
    price_df: pd.DataFrame,
    *,
    params: StrategyParams | None = None,
    mode: Mode = "production",
    throttle_config: AutoThrottleConfig | None = None,
    fee_rate: float = DEFAULT_EXECUTION.fee_rate,
    slippage_rate: float = DEFAULT_EXECUTION.slippage_rate,
    exit_priority: TradeExitPriority = "tp",
    entry_df: pd.DataFrame | None = None,
    min_trades_for_stats: int = 200,
) -> BacktestResult:
    """Run the full backtest pipeline from signals to metrics."""
    if not isinstance(price_df.index, pd.DatetimeIndex):
        raise ValueError("price_df must have a DatetimeIndex")

    params = params or StrategyParams()
    params.fee_rate = fee_rate
    params.slippage_buffer = max(params.slippage_buffer, slippage_rate)

    signals_df = generate_signals(price_df, params, mode=mode, throttle_config=throttle_config)

    trades = build_trades(
        signals_df,
        signal_col="buy_signal",
        tp_col="tp_pct",
        sl_col="sl_pct",
        priority=exit_priority,
        fee_rate=fee_rate,
        slippage_rate=slippage_rate,
        entry_df=entry_df,
        max_stopouts_per_day=params.max_stopouts_per_day,
        daily_loss_cap_pct=params.daily_loss_cap_pct,
        time_stop_bars=params.time_stop_bars,
        enable_partial_take_profit=params.enable_partial_take_profit,
        partial_tp_fraction=params.partial_tp_fraction,
        partial_tp_rr=params.partial_tp_rr,
        breakeven_buffer_pct=params.breakeven_buffer_pct,
        enable_trailing_stop=params.enable_trailing_stop,
        trailing_stop_activation_rr=params.trailing_stop_activation_rr,
        trailing_stop_atr_multiple=params.trailing_stop_atr_multiple,
    )

    equity_curve = compute_equity_curve(signals_df, trades)
    metrics = compute_metrics(
        signals_df,
        trades,
        equity_curve,
        min_trades_for_stats=min_trades_for_stats,
    )

    equity_df = pd.DataFrame(
        {
            "equity": equity_curve.equity,
            "drawdown": equity_curve.drawdown,
            "high_water": equity_curve.high_water,
        }
    )

    return BacktestResult(
        prices=price_df,
        signals=signals_df,
        trades=trades,
        equity=equity_df,
        metrics=metrics,
    )

