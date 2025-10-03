import pytest
import pandas as pd
import numpy as np

from coin_trade.trades import build_trades


def _make_df(index, prices):
    df = pd.DataFrame(
        {
            "open": prices,
            "high": prices + 0.5,
            "low": prices - 0.5,
            "close": prices + 1,
            "volume": 1000,
        },
        index=index,
    )
    df["buy_signal"] = False
    df["tp_pct"] = np.nan
    df["sl_pct"] = np.nan
    return df


def test_next_bar_open_entry():
    index = pd.date_range("2021-01-01", periods=6, freq="15T")
    prices = np.array([100, 101, 102, 103, 104, 105], dtype=float)
    df = _make_df(index, prices)

    df.loc[index[0], "buy_signal"] = True
    df.loc[index[0], "tp_pct"] = 0.02
    df.loc[index[0], "sl_pct"] = 0.01
    df.loc[index[2], "high"] = df.loc[index[2], "open"] * 1.02 + 0.1

    trades = build_trades(df)
    assert len(trades) == 1
    trade = trades.iloc[0]
    assert trade["entry_idx"] == 1
    assert trade["entry_price"] == df.loc[index[1], "open"]
    assert trade["signal_idx"] == 0


def test_same_bar_priority_resolution():
    index = pd.date_range("2021-01-01", periods=5, freq="15T")
    prices = np.array([100, 100, 100, 100, 100], dtype=float)
    df = _make_df(index, prices)

    df.loc[index[0], "buy_signal"] = True
    df.loc[index[0], "tp_pct"] = 0.01
    df.loc[index[0], "sl_pct"] = 0.01

    df.loc[index[1], "high"] = df.loc[index[1], "open"] * 1.02
    df.loc[index[1], "low"] = df.loc[index[1], "open"] * 0.98

    trades_tp = build_trades(df, priority="tp")
    trades_sl = build_trades(df, priority="sl")

    assert trades_tp.iloc[0]["outcome"] == "tp"
    assert trades_sl.iloc[0]["outcome"] == "sl"


def test_ignore_signals_during_open_trade():
    index = pd.date_range("2021-01-01", periods=8, freq="15T")
    prices = np.linspace(100, 110, len(index))
    df = _make_df(index, prices)

    df.loc[index[0], ["buy_signal", "tp_pct", "sl_pct"]] = [True, 0.2, 0.01]
    df.loc[index[1], ["buy_signal", "tp_pct", "sl_pct"]] = [True, 0.2, 0.01]
    entry_price = df.loc[index[1], "open"]
    stop_price = entry_price * (1 - 0.01)
    df.loc[index[5], "low"] = stop_price - 0.1

    trades = build_trades(df)
    assert len(trades) == 1
    assert trades.iloc[0]["entry_idx"] == 1
    assert trades.iloc[0]["exit_idx"] == 5



def test_fee_and_slippage_are_applied():
    index = pd.date_range("2021-01-01", periods=4, freq="15T")
    prices = np.array([100, 100, 110, 110], dtype=float)
    df = _make_df(index, prices)

    df.loc[index[0], ["buy_signal", "tp_pct", "sl_pct"]] = [True, 0.1, 0.05]
    df.loc[index[2], "high"] = 120

    trades = build_trades(df, fee_rate=0.001, slippage_rate=0.001)
    trade = trades.iloc[0]

    assert trade["gross_return_pct"] == pytest.approx(0.10, rel=1e-6)
    expected_net = (110 * (1 - 0.002)) / (100 * (1 + 0.002)) - 1
    assert trade["net_return_pct"] == pytest.approx(expected_net, rel=1e-6)
    assert trade["cost_pct"] == pytest.approx(trade["gross_return_pct"] - expected_net, rel=1e-6)

def test_max_stopouts_per_day_limits_additional_signals():
    index = pd.date_range("2021-01-01", periods=12, freq="15T")
    prices = np.linspace(100, 111, len(index))
    df = _make_df(index, prices)

    for i in range(3):
        df.loc[index[i], ["buy_signal", "tp_pct", "sl_pct"]] = [True, 0.05, 0.01]
        df.loc[index[i + 1], "low"] = df.loc[index[i + 1], "open"] * 0.99 - 0.1

    trades = build_trades(df, max_stopouts_per_day=1)
    assert len(trades) == 1
    assert (trades["outcome"] == "sl").all()


def test_time_stop_exit_marks_time_outcome():
    index = pd.date_range("2021-01-01", periods=10, freq="15T")
    prices = np.linspace(100, 101, len(index))
    df = _make_df(index, prices)

    df.loc[index[0], ["buy_signal", "tp_pct", "sl_pct"]] = [True, 0.05, 0.05]
    trades = build_trades(df, time_stop_bars=2)

    assert len(trades) == 1
    trade = trades.iloc[0]
    assert trade["outcome"] == "time"
    assert trade["holding_bars"] == 3
