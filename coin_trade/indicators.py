from __future__ import annotations

import numpy as np
import pandas as pd


def ema(series: pd.Series, span: int) -> pd.Series:
    """Exponential moving average wrapper with adjust=False."""
    return series.ewm(span=span, adjust=False, min_periods=span).mean()


def true_range(df: pd.DataFrame) -> pd.Series:
    """Compute the true range for each bar."""
    previous_close = df["close"].shift(1)
    high_low = df["high"] - df["low"]
    high_pc = (df["high"] - previous_close).abs()
    low_pc = (df["low"] - previous_close).abs()
    tr = pd.concat([high_low, high_pc, low_pc], axis=1)
    return tr.max(axis=1)


def atr(df: pd.DataFrame, window: int = 14) -> pd.Series:
    """Average true range using a simple moving average."""
    return true_range(df).rolling(window=window, min_periods=window).mean()


def adx(df: pd.DataFrame, window: int = 14) -> pd.Series:
    """Average Directional Index following Wilder's smoothing."""
    up_move = df["high"].diff()
    down_move = -df["low"].diff()

    plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0.0)
    minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0.0)

    tr = true_range(df)
    atr_window = tr.rolling(window=window, min_periods=window).mean()

    plus_di = (
        100
        * pd.Series(plus_dm, index=df.index).rolling(window=window, min_periods=window).mean()
        / (atr_window + 1e-12)
    )
    minus_di = (
        100
        * pd.Series(minus_dm, index=df.index).rolling(window=window, min_periods=window).mean()
        / (atr_window + 1e-12)
    )

    dx = (plus_di.subtract(minus_di).abs() / (plus_di.add(minus_di) + 1e-12)) * 100
    return dx.rolling(window=window, min_periods=window).mean()


def bollinger_bands(
    series: pd.Series,
    window: int = 20,
    num_std: float = 2.0,
) -> tuple[pd.Series, pd.Series, pd.Series]:
    """Return Bollinger Band mid, upper, and lower lines."""
    mid = series.rolling(window=window, min_periods=window).mean()
    std = series.rolling(window=window, min_periods=window).std(ddof=0)
    upper = mid + num_std * std
    lower = mid - num_std * std
    return mid, upper, lower


def keltner_channel(
    df: pd.DataFrame,
    window: int = 20,
    multiplier: float = 1.5,
) -> tuple[pd.Series, pd.Series, pd.Series]:
    """Return Keltner Channel mid, upper, and lower lines."""
    mid = ema(df["close"], span=window)
    rng = atr(df, window)
    upper = mid + multiplier * rng
    lower = mid - multiplier * rng
    return mid, upper, lower


def rsi(series: pd.Series, window: int = 14) -> pd.Series:
    """Classic RSI using smoothed moving averages."""
    delta = series.diff()
    up = delta.clip(lower=0)
    down = -delta.clip(upper=0)

    roll_up = up.rolling(window=window, min_periods=window).mean()
    roll_down = down.rolling(window=window, min_periods=window).mean()

    rs = roll_up / (roll_down + 1e-12)
    return 100 - (100 / (1 + rs))


def pct_rank(series: pd.Series, window: int) -> pd.Series:
    """Rolling percentile rank of the most recent value."""
    def _rank(values: pd.Series) -> float:
        if values.isna().any():
            values = values.dropna()
        if not len(values):
            return np.nan
        return (values.rank(pct=True).iloc[-1]).item()

    return series.rolling(window=window, min_periods=window).apply(_rank, raw=False)
