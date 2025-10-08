import numpy as np
import pandas as pd

from coin_trade.regime import RegimeConfig, detect_regime


def _make_price_frame(close_values: np.ndarray, start: str) -> pd.DataFrame:
    index = pd.date_range(start, periods=len(close_values), freq="15min")
    df = pd.DataFrame(
        {
            "open": close_values,
            "high": close_values + 0.3,
            "low": close_values - 0.3,
            "close": close_values,
            "volume": np.full(len(close_values), 1_000.0),
        },
        index=index,
    )
    return df


def test_detect_regime_trend_and_range_segments():
    trend_part = np.linspace(100, 112, 200)
    range_part = 106 + 0.8 * np.sin(np.linspace(0, 8 * np.pi, 200))
    close = np.concatenate([trend_part, range_part])
    df = _make_price_frame(close, "2024-01-01")

    config = RegimeConfig(
        adx_trend_threshold=15.0,
        adx_range_threshold=18.0,
        slope_trend_threshold=0.0007,
        slope_range_threshold=0.00015,
        vol_ratio_trend=1.05,
        vol_ratio_range=1.0,
        switch_cooldown=6,
        min_trend_votes=2,
        min_range_votes=1,
    )

    result = detect_regime(df, config)
    labels = result.labels
    counts_start = labels.iloc[:200].value_counts()
    counts_end = labels.iloc[200:].value_counts()

    assert {"trend", "range"}.issubset(set(labels.unique()))
    assert counts_start.get("range", 0) > 0
    assert counts_end.get("trend", 0) > 0


def test_regime_hysteresis_limits_flapping():
    steps = 160
    move = np.where(np.arange(steps) % 2 == 0, 0.4, -0.35)
    close = 100 + np.cumsum(move)
    df = _make_price_frame(close, "2024-02-01")

    config = RegimeConfig(
        adx_trend_threshold=12.0,
        adx_range_threshold=10.0,
        slope_trend_threshold=0.0005,
        slope_range_threshold=0.0002,
        vol_ratio_trend=1.02,
        vol_ratio_range=0.97,
        switch_cooldown=10,
        min_trend_votes=2,
        min_range_votes=2,
    )

    result = detect_regime(df, config)
    labels = result.labels
    changes = labels.ne(labels.shift()).sum()
    switches = max(int(changes) - 1, 0)

    unique = set(labels.unique())
    assert unique.issubset({"trend", "range"})
    assert switches <= steps // config.switch_cooldown + 3
