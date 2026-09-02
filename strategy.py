"""
BULL. BEAR AND BROKE - V14 STRATEGY
===================================

V14 DATA / TIMEFRAME SAFE VERSION

Architecture:
    4H regime
        ↓
    15M momentum
        ↓
    1M pullback/reclaim
        ↓
    next 1M bar execution
        ↓
    ATR stop / target

IMPORTANT:
    Higher-timeframe candles are shifted so the signal only sees
    COMPLETED 15M and 4H candles.

    All timestamps are normalized to:
        datetime64[ns, UTC]

    This prevents pandas merge_asof failures caused by Yahoo returning
    different datetime resolutions such as datetime64[s] and datetime64[us].
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd


# ============================================================================
# CONFIG
# ============================================================================

ATR_PERIOD = 14
RSI_PERIOD = 14

MACD_FAST = 12
MACD_SLOW = 26
MACD_SIGNAL = 9

EMA_4H_PERIOD = 20

# Risk / reward
ATR_STOP_MULT = 1.25
ATR_TARGET_MULT = 2.00

# Momentum
RSI_MIN = 52.0
RSI_MAX = 72.0

# 1M execution quality
EMA_FAST_1M = 9
EMA_SLOW_1M = 21

# Pullback/reclaim
MAX_DISTANCE_FROM_EMA_ATR = 1.50


# ============================================================================
# SIGNAL
# ============================================================================

@dataclass
class Signal:
    valid: bool
    reason: str = ""


# ============================================================================
# TIME INDEX NORMALIZATION
# ============================================================================

def normalize_index(df: pd.DataFrame) -> pd.DataFrame:
    """
    Normalize every dataframe index to exactly:

        datetime64[ns, UTC]

    Yahoo Finance can return different datetime resolutions depending
    on interval/source. pandas merge_asof requires compatible dtypes.
    """

    if df is None or df.empty:
        return pd.DataFrame()

    out = df.copy()

    # Convert index safely.
    index = pd.to_datetime(
        out.index,
        errors="coerce",
        utc=True,
    )

    valid = ~index.isna()

    out = out.loc[valid].copy()

    out.index = index[valid]

    # Force nanosecond resolution.
    out.index = pd.DatetimeIndex(
        out.index
    ).as_unit("ns")

    # Sort.
    out = out.sort_index()

    # Remove duplicate timestamps.
    out = out[
        ~out.index.duplicated(
            keep="last"
        )
    ]

    return out


# ============================================================================
# OHLCV CLEANUP
# ============================================================================

def clean_ohlcv(
    df: pd.DataFrame,
) -> pd.DataFrame:

    if df is None or df.empty:
        return pd.DataFrame()

    out = normalize_index(df)

    required = [
        "Open",
        "High",
        "Low",
        "Close",
        "Volume",
    ]

    if not all(
        column in out.columns
        for column in required
    ):
        return pd.DataFrame()

    out = out[required].copy()

    for column in required:

        out[column] = pd.to_numeric(
            out[column],
            errors="coerce",
        )

    out = out.dropna(
        subset=[
            "Open",
            "High",
            "Low",
            "Close",
        ]
    )

    return out


# ============================================================================
# EMA
# ============================================================================

def _ema(
    series: pd.Series,
    period: int,
) -> pd.Series:

    return series.ewm(
        span=period,
        adjust=False,
        min_periods=period,
    ).mean()


# ============================================================================
# RSI
# ============================================================================

def _rsi(
    close: pd.Series,
    period: int = RSI_PERIOD,
) -> pd.Series:

    delta = close.diff()

    gain = delta.clip(
        lower=0
    )

    loss = -delta.clip(
        upper=0
    )

    avg_gain = gain.ewm(
        alpha=1 / period,
        adjust=False,
        min_periods=period,
    ).mean()

    avg_loss = loss.ewm(
        alpha=1 / period,
        adjust=False,
        min_periods=period,
    ).mean()

    rs = (
        avg_gain
        / avg_loss.replace(
            0,
            np.nan,
        )
    )

    return 100 - (
        100 / (1 + rs)
    )


# ============================================================================
# ATR
# ============================================================================

def _atr(
    df: pd.DataFrame,
    period: int = ATR_PERIOD,
) -> pd.Series:

    previous_close = df["Close"].shift(1)

    tr = pd.concat(
        [
            df["High"] - df["Low"],
            (
                df["High"]
                - previous_close
            ).abs(),
            (
                df["Low"]
                - previous_close
            ).abs(),
        ],
        axis=1,
    ).max(axis=1)

    return tr.ewm(
        alpha=1 / period,
        adjust=False,
        min_periods=period,
    ).mean()


# ============================================================================
# 1M INDICATORS
# ============================================================================

def calculate_1m_indicators(
    df: pd.DataFrame,
) -> pd.DataFrame:

    out = clean_ohlcv(df)

    if out.empty:
        return out

    out["ATR_1M"] = _atr(
        out,
        ATR_PERIOD,
    )

    out["RSI_1M"] = _rsi(
        out["Close"],
        RSI_PERIOD,
    )

    out["EMA9_1M"] = _ema(
        out["Close"],
        EMA_FAST_1M,
    )

    out["EMA21_1M"] = _ema(
        out["Close"],
        EMA_SLOW_1M,
    )

    return out


# ============================================================================
# 15M INDICATORS
# ============================================================================

def calculate_15m_indicators(
    df: pd.DataFrame,
) -> pd.DataFrame:

    out = clean_ohlcv(df)

    if out.empty:
        return out

    fast = _ema(
        out["Close"],
        MACD_FAST,
    )

    slow = _ema(
        out["Close"],
        MACD_SLOW,
    )

    macd = fast - slow

    signal = _ema(
        macd,
        MACD_SIGNAL,
    )

    out["MACD_15M"] = macd

    out["MACD_SIGNAL_15M"] = signal

    out["MACD_DIFF_15M"] = (
        macd - signal
    )

    return out


# ============================================================================
# 4H INDICATORS
# ============================================================================

def calculate_4h_indicators(
    df: pd.DataFrame,
) -> pd.DataFrame:

    out = clean_ohlcv(df)

    if out.empty:
        return out

    out["EMA20_4H"] = _ema(
        out["Close"],
        EMA_4H_PERIOD,
    )

    return out


# ============================================================================
# ALIGN TIMEFRAMES
# ============================================================================

def align_timeframes(
    df_1m: pd.DataFrame,
    df_15m: pd.DataFrame,
    df_4h: pd.DataFrame,
) -> pd.DataFrame:
    """
    Align real 1M, 15M and 4H data.

    IMPORTANT:

    The higher-timeframe series are shifted by one candle BEFORE
    merge_asof().

    Therefore a 1M candle can only see the previous completed
    15M and 4H candles.
    """

    one = calculate_1m_indicators(
        df_1m
    )

    fifteen = calculate_15m_indicators(
        df_15m
    )

    four = calculate_4h_indicators(
        df_4h
    )

    if one.empty:
        raise ValueError(
            "1M dataframe is empty."
        )

    if fifteen.empty:
        raise ValueError(
            "15M dataframe is empty."
        )

    if four.empty:
        raise ValueError(
            "4H dataframe is empty."
        )

    # ------------------------------------------------------------------------
    # Ensure exact timestamp dtype one more time.
    # ------------------------------------------------------------------------

    one = normalize_index(one)
    fifteen = normalize_index(fifteen)
    four = normalize_index(four)

    # ------------------------------------------------------------------------
    # Keep only required higher-timeframe columns.
    # ------------------------------------------------------------------------

    fifteen = fifteen[
        [
            "MACD_15M",
            "MACD_SIGNAL_15M",
            "MACD_DIFF_15M",
        ]
    ].copy()

    four = four[
        [
            "EMA20_4H",
        ]
    ].copy()

    # ------------------------------------------------------------------------
    # CRITICAL:
    #
    # Shift higher timeframe indicators one full candle.
    #
    # Example:
    # 10:15 1M bar cannot use the 10:00-10:15 candle while that candle
    # is still forming.
    #
    # After shift, the value available at the next timeframe timestamp
    # represents the completed prior candle.
    # ------------------------------------------------------------------------

    fifteen = fifteen.shift(1)

    four = four.shift(1)

    # ------------------------------------------------------------------------
    # Remove rows where indicators aren't ready.
    # ------------------------------------------------------------------------

    fifteen = fifteen.dropna(
        subset=[
            "MACD_DIFF_15M",
        ]
    )

    four = four.dropna(
        subset=[
            "EMA20_4H",
        ]
    )

    if fifteen.empty:
        raise ValueError(
            "15M indicators contain no valid completed candles."
        )

    if four.empty:
        raise ValueError(
            "4H EMA20 contains no valid completed candles."
        )

    # ------------------------------------------------------------------------
    # Verify timestamp dtypes.
    # ------------------------------------------------------------------------

    if not isinstance(
        one.index,
        pd.DatetimeIndex,
    ):
        raise TypeError(
            "1M index is not DatetimeIndex."
        )

    if not isinstance(
        fifteen.index,
        pd.DatetimeIndex,
    ):
        raise TypeError(
            "15M index is not DatetimeIndex."
        )

    if not isinstance(
        four.index,
        pd.DatetimeIndex,
    ):
        raise TypeError(
            "4H index is not DatetimeIndex."
        )

    # ------------------------------------------------------------------------
    # Final hard normalization.
    # ------------------------------------------------------------------------

    one.index = pd.DatetimeIndex(
        one.index
    ).as_unit("ns")

    fifteen.index = pd.DatetimeIndex(
        fifteen.index
    ).as_unit("ns")

    four.index = pd.DatetimeIndex(
        four.index
    ).as_unit("ns")

    # ------------------------------------------------------------------------
    # merge_asof 1M <- 15M
    # ------------------------------------------------------------------------

    one = pd.merge_asof(
        one.sort_index(),
        fifteen.sort_index(),
        left_index=True,
        right_index=True,
        direction="backward",
    )

    # ------------------------------------------------------------------------
    # merge_asof result <- 4H
    # ------------------------------------------------------------------------

    one = pd.merge_asof(
        one.sort_index(),
        four.sort_index(),
        left_index=True,
        right_index=True,
        direction="backward",
    )

    return one


# ============================================================================
# BACKWARD COMPATIBILITY
# ============================================================================

def calculate_indicators(
    df: pd.DataFrame,
) -> pd.DataFrame:

    """
    Older code can still call this.

    For V14 multi-timeframe backtesting, use align_timeframes().
    """

    return calculate_1m_indicators(df)


# ============================================================================
# SIGNAL ENGINE
# ============================================================================

def evaluate_signal(
    row: pd.Series,
) -> Signal:

    required = [
        "Close",
        "High",
        "Low",
        "ATR_1M",
        "RSI_1M",
        "MACD_DIFF_15M",
        "EMA20_4H",
        "EMA9_1M",
        "EMA21_1M",
    ]

    for column in required:

        value = row.get(
            column,
            np.nan,
        )

        if not np.isfinite(value):

            return Signal(
                False,
                f"missing_{column}",
            )

    close = float(
        row["Close"]
    )

    high = float(
        row["High"]
    )

    low = float(
        row["Low"]
    )

    atr = float(
        row["ATR_1M"]
    )

    rsi = float(
        row["RSI_1M"]
    )

    macd_diff = float(
        row["MACD_DIFF_15M"]
    )

    ema20_4h = float(
        row["EMA20_4H"]
    )

    ema9 = float(
        row["EMA9_1M"]
    )

    ema21 = float(
        row["EMA21_1M"]
    )

    # ------------------------------------------------------------------------
    # Basic validity
    # ------------------------------------------------------------------------

    if close <= 0:
        return Signal(
            False,
            "invalid_close",
        )

    if atr <= 0:
        return Signal(
            False,
            "invalid_atr",
        )

    # ------------------------------------------------------------------------
    # 4H REGIME
    # ------------------------------------------------------------------------

    if close <= ema20_4h:

        return Signal(
            False,
            "below_4h_ema",
        )

    # ------------------------------------------------------------------------
    # 15M MOMENTUM
    # ------------------------------------------------------------------------

    if macd_diff <= 0:

        return Signal(
            False,
            "negative_15m_macd",
        )

    # ------------------------------------------------------------------------
    # 1M RSI
    # ------------------------------------------------------------------------

    if not (
        RSI_MIN
        <= rsi
        <= RSI_MAX
    ):

        return Signal(
            False,
            "rsi_filter",
        )

    # ------------------------------------------------------------------------
    # 1M EMA STRUCTURE
    #
    # Price must reclaim the fast EMA while fast EMA is above slow EMA.
    # ------------------------------------------------------------------------

    if not (
        close > ema9
        and ema9 > ema21
    ):

        return Signal(
            False,
            "no_1m_reclaim",
        )

    # ------------------------------------------------------------------------
    # Don't buy extremely stretched price.
    # ------------------------------------------------------------------------

    distance_from_ema = (
        close - ema21
    ) / atr

    if (
        distance_from_ema
        > MAX_DISTANCE_FROM_EMA_ATR
    ):

        return Signal(
            False,
            "overextended",
        )

    # ------------------------------------------------------------------------
    # Don't allow absurd single-bar ranges.
    # ------------------------------------------------------------------------

    candle_range = (
        high - low
    )

    if candle_range > 3.0 * atr:

        return Signal(
            False,
            "oversized_candle",
        )

    return Signal(
        True,
        "V14_LONG",
    )


# ============================================================================
# EXIT PRICES
# ============================================================================

def calculate_exit_prices(
    entry_price: float,
    atr: float,
) -> dict:

    if (
        not np.isfinite(entry_price)
        or not np.isfinite(atr)
        or entry_price <= 0
        or atr <= 0
    ):

        raise ValueError(
            "Invalid entry price or ATR."
        )

    stop = (
        entry_price
        - ATR_STOP_MULT * atr
    )

    target = (
        entry_price
        + ATR_TARGET_MULT * atr
    )

    if stop <= 0:

        raise ValueError(
            "Calculated stop is not positive."
        )

    return {
        "stop": float(stop),
        "target": float(target),
    }
