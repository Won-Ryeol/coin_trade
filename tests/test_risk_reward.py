import numpy as np
import pandas as pd

from coin_trade.risk_reward import RiskRewardSettings, compute_risk_targets


def _default_settings() -> RiskRewardSettings:
    return RiskRewardSettings(
        target_sl_low=0.008,
        target_sl_high=0.013,
        max_sl_pct=0.03,
        max_tp_pct=0.06,
        rr_low=1.8,
        rr_high=2.4,
        rr_min=1.05,
        rr_max=2.8,
        adx_rr_threshold=28.0,
        breakout_slope_knots=(-0.004, -0.001, 0.0015, 0.0035),
        breakout_slope_values=(1.4, 1.7, 2.1, 2.6),
        breakout_adx_knots=(15.0, 22.0, 28.0, 40.0),
        breakout_adx_values=(1.5, 1.8, 2.2, 2.6),
        breakout_blend=0.55,
        reversion_slope_knots=(-0.004, -0.0015, 0.0, 0.0015, 0.003),
        reversion_slope_values=(0.9, 1.05, 1.2, 1.45, 1.6),
        reversion_floor=0.9,
        reversion_ceiling=1.7,
        sl_scale_clip=(0.5, 4.0),
        atr_floor=1e-6,
        buffer_factor=2.0,
        breakout_stop_multiplier=1.05,
        reversion_stop_multiplier=0.9,
        fee_rate=0.0005,
        slippage_buffer=0.0,
    )


def test_compute_risk_targets_bounds() -> None:
    index = pd.date_range("2024-01-01", periods=6, freq="15min")
    atr = pd.Series(np.linspace(0.007, 0.014, len(index)), index=index)
    slope = pd.Series(np.linspace(-0.0025, 0.003, len(index)), index=index)
    adx = pd.Series(np.linspace(18.0, 42.0, len(index)), index=index)

    targets = compute_risk_targets(
        atr_pct_ema=atr,
        slope=slope,
        adx=adx,
        settings=_default_settings(),
    )

    assert targets.rr_breakout.between(1.05, 2.8).all()
    assert targets.rr_reversion.between(0.9, 1.7).all()
    assert (targets.tp_breakout >= targets.sl_breakout).all()
    assert (targets.tp_reversion >= targets.sl_reversion).all()


def test_compute_risk_targets_sensitivity() -> None:
    index = pd.date_range("2024-02-01", periods=4, freq="15min")
    atr = pd.Series([0.01, 0.011, 0.012, 0.013], index=index)
    slope_low = pd.Series([-0.002, -0.0015, -0.001, -0.0005], index=index)
    slope_high = pd.Series([0.001, 0.0015, 0.002, 0.003], index=index)
    adx = pd.Series(24.0, index=index)

    settings = _default_settings()

    low_targets = compute_risk_targets(atr_pct_ema=atr, slope=slope_low, adx=adx, settings=settings)
    high_targets = compute_risk_targets(atr_pct_ema=atr, slope=slope_high, adx=adx, settings=settings)

    assert high_targets.rr_breakout.mean() > low_targets.rr_breakout.mean()
    assert high_targets.tp_breakout.mean() > low_targets.tp_breakout.mean()