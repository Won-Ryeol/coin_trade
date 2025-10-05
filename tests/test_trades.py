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

def test_partial_tp_promotes_stop_to_breakeven():
    index = pd.date_range("2021-01-01", periods=8, freq="15T")
    prices = np.full(len(index), 100.0)
    df = _make_df(index, prices)
    df["ATR"] = 0.5

    df.loc[index[0], ["buy_signal", "tp_pct", "sl_pct"]] = [True, 0.04, 0.02]
    partial_bar = index[2]
    df.loc[partial_bar, "high"] = 102.2
    stop_hit_bar = index[3]
    df.loc[stop_hit_bar, "low"] = 99.9

    trades = build_trades(
        df,
        enable_partial_take_profit=True,
        partial_tp_fraction=0.5,
        partial_tp_rr=1.0,
        breakeven_buffer_pct=0.0005,
    )

    assert len(trades) == 1
    trade = trades.iloc[0]
    assert trade["partial_fraction"] == pytest.approx(0.5)
    assert trade["outcome"] == "breakeven"
    assert trade["final_stop_source"] == "breakeven"
    assert trade["partial_exit_price"] == pytest.approx(102.0, rel=1e-6, abs=1e-6)
    assert trade["gross_return_pct"] > 0
    assert "partial" in trade["event_log"]
    assert "breakeven" in trade["event_log"]


def test_trailing_stop_tightens_and_exits():
    index = pd.date_range("2021-01-02", periods=8, freq="15T")
    prices = np.full(len(index), 100.0)
    df = _make_df(index, prices)
    df["ATR"] = 0.5

    df.loc[index[0], ["buy_signal", "tp_pct", "sl_pct"]] = [True, 0.10, 0.02]
    df.loc[index[1], "high"] = 102.5
    df.loc[index[2], "high"] = 103.0
    df.loc[index[3], "high"] = 104.0
    df.loc[index[4], "low"] = 103.0

    trades = build_trades(
        df,
        enable_trailing_stop=True,
        trailing_stop_activation_rr=0.5,
        trailing_stop_atr_multiple=1.0,
    )

    assert len(trades) == 1
    trade = trades.iloc[0]
    assert trade["outcome"] == "trail"
    assert trade["final_stop_source"] == "trailing"
    assert trade["trailing_active"] is True
    assert trade["partial_fraction"] == pytest.approx(0.0)
    assert trade["exit_price"] > trade["entry_price"]
    assert "trail@" in trade["event_log"]
