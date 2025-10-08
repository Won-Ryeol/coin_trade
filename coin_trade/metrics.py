from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass
class EquityCurve:
    index: pd.DatetimeIndex
    equity: pd.Series
    drawdown: pd.Series
    high_water: pd.Series


@dataclass
class MetricResult:
    metrics: dict[str, float]
    min_trades_threshold: int


def compute_equity_curve(
    df: pd.DataFrame,
    trades: pd.DataFrame,
    initial_equity: float = 1.0,
) -> EquityCurve:
    if df.empty:
        raise ValueError("Price DataFrame is empty")

    equity_values = np.full(len(df), np.nan, dtype=float)
    equity_values[0] = initial_equity
    running_equity = initial_equity

    if not trades.empty:
        sorted_trades = trades.sort_values("exit_idx")
        for _, trade in sorted_trades.iterrows():
            running_equity *= 1.0 + trade["net_return_pct"]
            exit_idx = int(trade["exit_idx"])
            if exit_idx < len(df):
                equity_values[exit_idx] = running_equity

    equity_series = pd.Series(equity_values, index=df.index).ffill().fillna(initial_equity)
    high_water = equity_series.cummax()
    drawdown = equity_series / high_water - 1.0

    return EquityCurve(
        index=df.index,
        equity=equity_series,
        drawdown=drawdown,
        high_water=high_water,
    )


def _annualization_factor(df: pd.DataFrame) -> float:
    if len(df.index) < 2:
        return np.nan
    freq_seconds = (df.index[1] - df.index[0]).total_seconds()
    if freq_seconds <= 0:
        return np.nan
    bars_per_day = 86400 / freq_seconds
    return np.sqrt(bars_per_day * 252)


def compute_metrics(
    df: pd.DataFrame,
    trades: pd.DataFrame,
    equity: EquityCurve,
    *,
    min_trades_for_stats: int = 200,
) -> MetricResult:
    metrics: dict[str, float] = {}

    trade_count = int(len(trades))
    metrics["trade_count"] = float(trade_count)

    unique_days = df.index.normalize().unique()
    metrics["trades_per_day"] = float(trade_count / max(len(unique_days), 1))

    metrics["exposure"] = float(trades["holding_bars"].sum() / max(len(df), 1)) if trade_count else 0.0

    metrics["net_return_pct"] = float(equity.equity.iloc[-1] - 1.0)
    metrics["max_drawdown_pct"] = float(equity.drawdown.min())

    returns = equity.equity.pct_change().dropna()
    annual_factor = _annualization_factor(df)
    if returns.empty or np.isnan(annual_factor):
        metrics["sharpe"] = np.nan
    else:
        std = returns.std()
        if std == 0:
            metrics["sharpe"] = np.nan
        else:
            metrics["sharpe"] = float(returns.mean() / std * annual_factor)

    if trade_count >= min_trades_for_stats and trade_count > 0:
        wins = trades[trades["net_return_pct"] > 0]["net_return_pct"].sum()
        losses = trades[trades["net_return_pct"] < 0]["net_return_pct"].sum()
        metrics["win_rate"] = float((trades["net_return_pct"] > 0).mean())
        if losses != 0:
            metrics["profit_factor"] = float(wins / abs(losses))
        else:
            metrics["profit_factor"] = np.inf
    else:
        metrics["win_rate"] = np.nan
        metrics["profit_factor"] = np.nan

    return MetricResult(metrics=metrics, min_trades_threshold=min_trades_for_stats)
