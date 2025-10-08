from __future__ import annotations

from collections.abc import Iterable

from coin_trade.config import DEFAULT_EXECUTION

PriceLeg = tuple[float, float]


def entry_cost(
    price: float,
    *,
    fee_rate: float = DEFAULT_EXECUTION.fee_rate,
    slippage_rate: float = DEFAULT_EXECUTION.slippage_rate,
) -> float:
    return price * (1.0 + fee_rate + slippage_rate)


def exit_cost(
    price: float,
    *,
    fee_rate: float = DEFAULT_EXECUTION.fee_rate,
    slippage_rate: float = DEFAULT_EXECUTION.slippage_rate,
) -> float:
    return price * (1.0 - fee_rate - slippage_rate)


def risk_pct_from_stop(stop_pct: float) -> float:
    return float(stop_pct)


def price_at_rr(entry_price: float, stop_pct: float, rr: float) -> float:
    risk = risk_pct_from_stop(stop_pct)
    return entry_price * (1.0 + risk * rr)


def weighted_returns(
    entry_price: float,
    legs: Iterable[PriceLeg],
    *,
    fee_rate: float,
    slippage_rate: float,
) -> tuple[float, float, float]:
    entry_effective = entry_cost(
        entry_price,
        fee_rate=fee_rate,
        slippage_rate=slippage_rate,
    )
    gross_total = 0.0
    net_total = 0.0
    for fraction, exit_price in legs:
        gross_leg = exit_price / entry_price - 1.0
        net_leg = (
            exit_cost(exit_price, fee_rate=fee_rate, slippage_rate=slippage_rate)
            / entry_effective
            - 1.0
        )
        gross_total += fraction * gross_leg
        net_total += fraction * net_leg
    return gross_total, net_total, gross_total - net_total
