"""
BULL. BEAR AND BROKE - V14 STRATEGY
===================================

V14:
    - Real 1M execution
    - Real 15M confirmation
    - Real 4H regime
    - No fake 4H EMA from 1M data
    - No lookahead from higher timeframes
    - Long only
    - Pullback + reclaim entry
    - 4H trend + slope regime
    - 15M MACD momentum
    - 1M RSI confirmation
    - ATR risk management

IMPORTANT:
    Experimental strategy.
    Historical performance is not a guarantee of future results.
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

EMA_FAST_1M = 9
EMA_SLOW_1M = 21

ATR_STOP_MULT = 1.25
ATR_TARGET_MULT = 2.00

RSI_MIN = 52.0
RSI_MAX = 72.0

MAX_CANDLE_ATR = 3.0

# Pullback/reclaim parameters.
PULLBACK_LOOKBACK = 5
RECLAIM_LOOKBACK = 3

# Do not enter when price is excessively extended from 1M EMA21.
MAX_EXTENSION_ATR = 1.75


# ============================================================================
# SIGNAL
# ============================================================================

@dataclass
class Signal:
    valid: bool
    reason: str = ""


# ============================================================================
# HELPERS
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


def _rsi(
    close: pd.Series,
    period: int = RSI_PERIOD,
) -> pd.Series:

    delta = close.diff()

    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)

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
        / avg_loss.replace(0, np.nan)
    )

    return 100 - (
        100 / (1 + rs)
    )


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
# 1M
# ============================================================================

def calculate_1m_indicators(
    df: pd.DataFrame,
) -> pd.DataFrame:

    out = df.copy()

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

    # Previous-bar values used by the signal.
    out["PREV_CLOSE_1M"] = (
        out["Close"].shift(1)
    )

    out["PREV_EMA9_1M"] = (
        out["EMA9_1M"].shift(1)
    )

    out["PREV_EMA21_1M"] = (
        out["EMA21_1M"].shift(1)
    )

    out["PREV_RSI_1M"] = (
        out["RSI_1M"].shift(1)
    )

    # Recent high excluding current candle.
    out["PREV_5_HIGH"] = (
        out["High"]
        .shift(1)
        .rolling(5)
        .max()
    )

    return out


# ============================================================================
# 15M
# ============================================================================

def calculate_15m_indicators(
    df: pd.DataFrame,
) -> pd.DataFrame:

    out = df.copy()

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

    out["PREV_MACD_DIFF_15M"] = (
        out["MACD_DIFF_15M"].shift(1)
    )

    return out


# ============================================================================
# 4H
# ============================================================================

def calculate_4h_indicators(
    df: pd.DataFrame,
) -> pd.DataFrame:

    out = df.copy()

    out["EMA20_4H"] = _ema(
        out["Close"],
        EMA_4H_PERIOD,
    )

    out["PREV_EMA20_4H"] = (
        out["EMA20_4H"].shift(1)
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

    one = calculate_1m_indicators(
        df_1m
    )

    fifteen = calculate_15m_indicators(
        df_15m
    )

    four = calculate_4h_indicators(
        df_4h
    )

    # ------------------------------------------------------------------------
    # IMPORTANT:
    #
    # Yahoo timestamps normally identify the START of the candle.
    #
    # A 15M candle beginning at 10:00 is not completed until 10:15.
    # A 4H candle beginning at 09:30 is not completed until 13:30.
    #
    # Move the higher timeframe timestamp forward by its duration.
    #
    # Then merge_asof backward.
    #
    # This is much safer than simply shift(1), which can introduce
    # unnecessary delays and confusing alignment.
    # ------------------------------------------------------------------------

    fifteen = fifteen[
        [
            "MACD_15M",
            "MACD_SIGNAL_15M",
            "MACD_DIFF_15M",
            "PREV_MACD_DIFF_15M",
        ]
    ].copy()

    four = four[
        [
            "EMA20_4H",
            "PREV_EMA20_4H",
        ]
    ].copy()

    fifteen.index = (
        fifteen.index
        + pd.Timedelta(minutes=15)
    )

    four.index = (
        four.index
        + pd.Timedelta(hours=4)
    )

    one = one.sort_index()

    fifteen = fifteen.sort_index()
    four = four.sort_index()

    one = pd.merge_asof(
        one,
        fifteen,
        left_index=True,
        right_index=True,
        direction="backward",
        allow_exact_matches=True,
    )

    one = pd.merge_asof(
        one,
        four,
        left_index=True,
        right_index=True,
        direction="backward",
        allow_exact_matches=True,
    )

    return one


# ============================================================================
# BACKWARD COMPATIBILITY
# ============================================================================

def calculate_indicators(
    df: pd.DataFrame,
) -> pd.DataFrame:

    return calculate_1m_indicators(
        df
    )


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
        "EMA9_1M",
        "EMA21_1M",
        "PREV_CLOSE_1M",
        "PREV_EMA9_1M",
        "PREV_EMA21_1M",
        "PREV_RSI_1M",
        "PREV_5_HIGH",
        "MACD_DIFF_15M",
        "PREV_MACD_DIFF_15M",
        "EMA20_4H",
        "PREV_EMA20_4H",
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

    close = float(row["Close"])
    high = float(row["High"])
    low = float(row["Low"])

    atr = float(
        row["ATR_1M"]
    )

    rsi = float(
        row["RSI_1M"]
    )

    ema9 = float(
        row["EMA9_1M"]
    )

    ema21 = float(
        row["EMA21_1M"]
    )

    prev_close = float(
        row["PREV_CLOSE_1M"]
    )

    prev_ema9 = float(
        row["PREV_EMA9_1M"]
    )

    prev_ema21 = float(
        row["PREV_EMA21_1M"]
    )

    prev_rsi = float(
        row["PREV_RSI_1M"]
    )

    previous_high = float(
        row["PREV_5_HIGH"]
    )

    macd_diff = float(
        row["MACD_DIFF_15M"]
    )

    previous_macd = float(
        row["PREV_MACD_DIFF_15M"]
    )

    ema20_4h = float(
        row["EMA20_4H"]
    )

    previous_ema20_4h = float(
        row["PREV_EMA20_4H"]
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

    if ema20_4h <= previous_ema20_4h:
        return Signal(
            False,
            "4h_ema_not_rising",
        )

    # ------------------------------------------------------------------------
    # 15M MOMENTUM
    # ------------------------------------------------------------------------

    if macd_diff <= 0:
        return Signal(
            False,
            "negative_15m_macd",
        )

    # Prefer improving momentum.
    if macd_diff < previous_macd:
        return Signal(
            False,
            "15m_macd_fading",
        )

    # ------------------------------------------------------------------------
    # 1M RSI
    # ------------------------------------------------------------------------

    if not (
        RSI_MIN <= rsi <= RSI_MAX
    ):
        return Signal(
            False,
            "rsi_filter",
        )

    # RSI should not be collapsing.
    if rsi < prev_rsi - 8:
        return Signal(
            False,
            "rsi_falling",
        )

    # ------------------------------------------------------------------------
    # CANDLE SIZE
    # ------------------------------------------------------------------------

    candle_range = high - low

    if candle_range > (
        MAX_CANDLE_ATR * atr
    ):
        return Signal(
            False,
            "oversized_candle",
        )

    # ------------------------------------------------------------------------
    # AVOID EXTREME EXTENSION
    # ------------------------------------------------------------------------

    extension = (
        close - ema21
    ) / atr

    if extension > MAX_EXTENSION_ATR:
        return Signal(
            False,
            "too_extended",
        )

    # ------------------------------------------------------------------------
    # PULLBACK / RECLAIM
    #
    # We want the previous candle to have been weaker,
    # while the current completed candle reclaims EMA9.
    # ------------------------------------------------------------------------

    previous_below_or_touch = (
        prev_close <= prev_ema9
        or prev_close <= prev_ema21
    )

    reclaim_now = (
        close > ema9
        and close > prev_close
    )

    if not previous_below_or_touch:
        return Signal(
            False,
            "no_pullback",
        )

    if not reclaim_now:
        return Signal(
            False,
            "no_reclaim",
        )

    # ------------------------------------------------------------------------
    # MICRO BREAKOUT
    #
    # Current close must reclaim recent structure.
    # ------------------------------------------------------------------------

    if close < previous_high:
        return Signal(
            False,
            "no_micro_breakout",
        )

    # ------------------------------------------------------------------------
    # FINAL SIGNAL
    # ------------------------------------------------------------------------

    return Signal(
        True,
        "V14_LONG_PULLBACK_RECLAIM",
    )


# ============================================================================
# EXITS
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
