import numpy as np
import pandas as pd

from coin_trade.signals import AutoThrottleConfig, StrategyParams, generate_signals, _warmup_bars


def _frozen_throttle() -> AutoThrottleConfig:
    return AutoThrottleConfig(
        target_trades_per_day_low=0.0,
        target_trades_per_day_high=1e9,
        window_days=5,
        cooldown_step=0,
        per_day_step=0,
        adx_step=0.0,
        squeeze_step=0.0,
        volume_step=0.0,
        min_cooldown=0,
        max_cooldown=10_000,
        min_per_day=1,
        max_per_day=10_000,
        bias_floor=0,
        bias_ceiling=0,
    )


def _base_params(**kwargs) -> StrategyParams:
    params = StrategyParams(
        bb_window=2,
        bb_k=0.1,
        kc_window=2,
        kc_mult=3.0,
        ema_trend_len=2,
        adx_window=2,
        atr_window=2,
        squeeze_window=3,
        squeeze_pct=1.0,
        adx_threshold=0.0,
        adx_rr_threshold=0.0,
        vol_lookback=1,
        volume_multiplier=0.0,
        enable_low_risk_reversion=False,
        enable_mean_reversion=True,
        rsi_window=2,
        rsi_buy_threshold=110.0,
        atr_ema_window=2,
        target_sl_low=0.005,
        target_sl_high=0.01,
        max_sl_pct=0.02,
        max_tp_pct=0.04,
    )
    for key, value in kwargs.items():
        setattr(params, key, value)
    return params


def _make_price_df(start: str, periods: int) -> pd.DataFrame:
    index = pd.date_range(start, periods=periods, freq="15T")
    prices = np.linspace(100, 105, periods)
    df = pd.DataFrame(
        {
            "open": prices,
            "high": prices + 0.5,
            "low": prices - 0.5,
            "close": prices,
            "volume": 1_000,
        },
        index=index,
    )
    return df


def test_cooldown_enforces_spacing():
    periods = 40
    df = _make_price_df("2021-01-01", periods)
    params = _base_params(cooldown_bars=3, max_trades_per_day=100)
    throttle = _frozen_throttle()

    signals = generate_signals(df, params, mode="production", throttle_config=throttle)
    assert "regime" in signals.columns
    idx = np.flatnonzero(signals["buy_signal"].to_numpy())
    if len(idx) > 1:
        diffs = np.diff(idx)
        assert (diffs >= params.cooldown_bars + 1).all()


def test_per_day_cap_limits_signals():
    periods = 96 * 2  # two days of 15m data
    df = _make_price_df("2021-01-01", periods)
    params = _base_params(cooldown_bars=0, max_trades_per_day=1)
    throttle = _frozen_throttle()

    signals = generate_signals(df, params, mode="production", throttle_config=throttle)
    signal_series = signals["buy_signal"]
    per_day = signal_series.groupby(signal_series.index.normalize()).sum()
    assert (per_day <= 1).all()


def test_no_signals_before_warmup():
    periods = 100
    df = _make_price_df("2021-01-01", periods)
    params = _base_params(cooldown_bars=0, max_trades_per_day=10)
    throttle = _frozen_throttle()

    warmup = _warmup_bars(params)
    signals = generate_signals(df, params, mode="production", throttle_config=throttle)
    signal_idx = np.flatnonzero(signals["buy_signal"].to_numpy())
    if len(signal_idx):
        assert signal_idx[0] >= warmup

