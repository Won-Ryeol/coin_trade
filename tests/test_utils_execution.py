import math

from coin_trade.utils.execution import price_at_rr, weighted_returns


def test_price_at_rr_scales_with_r_multiple():
    entry = 100.0
    stop_pct = 0.02
    rr = 1.5
    expected = entry * (1 + stop_pct * rr)
    assert price_at_rr(entry, stop_pct, rr) == expected


def test_weighted_returns_handles_multiple_legs():
    entry = 100.0
    legs = [(0.5, 105.0), (0.5, 95.0)]
    gross, net, cost = weighted_returns(entry, legs, fee_rate=0.001, slippage_rate=0.001)
    assert math.isfinite(gross)
    assert math.isfinite(net)
    assert math.isfinite(cost)
    assert gross != net
