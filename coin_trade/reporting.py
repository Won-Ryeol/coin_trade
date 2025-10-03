from __future__ import annotations

from typing import Any, Dict

from coin_trade.backtester import BacktestResult


def summarize_metrics(result: BacktestResult) -> Dict[str, Any]:
    metrics = result.metrics.metrics
    return {
        "trade_count": metrics.get("trade_count"),
        "trades_per_day": metrics.get("trades_per_day"),
        "win_rate": metrics.get("win_rate"),
        "profit_factor": metrics.get("profit_factor"),
        "net_return_pct": metrics.get("net_return_pct"),
        "max_drawdown_pct": metrics.get("max_drawdown_pct"),
        "sharpe": metrics.get("sharpe"),
        "exposure": metrics.get("exposure"),
    }


def format_metric(value: Any) -> str:
    if value is None:
        return "N/A"
    if isinstance(value, float):
        if value != value:  # NaN check
            return "N/A"
        if abs(value) >= 1000:
            return f"{value:.2e}"
        return f"{value:.4f}"
    return str(value)
