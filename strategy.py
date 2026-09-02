"""
BULL. BEAR AND BROKE - V14 STRATEGY
===================================

Regime -> Pullback -> Reclaim -> 1M Trigger

Design goals:
    - Real 4H regime data
    - Real 15M confirmation data
    - 1M execution trigger
    - Long-only
    - No lookahead
    - Pullback/reclaim structure
    - ATR-based risk
    - Conservative signal quality

The strategy is intentionally NOT optimized to a specific historical period.
The backtester should determine whether the structure has edge.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict

import numpy as np
import pandas as pd


# ============================================================================
# CONFIG
# ============================================================================

ATR_PERIOD = 14
RSI_PERIOD = 14

EMA_FAST_1M = 9
EMA_SLOW_1M = 21

EMA_FAST_15M = 9
EMA_SLOW_15M = 21

EMA_FAST_4H = 20
EMA_SLOW_4H = 50

MACD_FAST = 12
MACD_SLOW = 26
MACD_SIGNAL = 9

RVOL_PERIOD = 20

MIN_SCORE = 7

RSI_MIN = 45.0
RSI_MAX = 68.0

MAX_EXTENSION_ATR = 1.75
MAX_CANDLE_ATR = 2.50

PULLBACK_ATR_MIN = 0.15
PULLBACK_ATR_MAX = 1.50

STOP_ATR = 1.25
TARGET_ATR = 2.50

MIN_REWARD_RISK = 1.80


# ============================================================================
# RESULT
# ============================================================================

@dataclass
class Signal:
    valid: bool
    score: int = 0
    reason: str = ""


# ============================================================================
# INDICATORS
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

    gain = delta.clip(lower=0.0)
    loss = -delta.clip(upper=0.0)

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

    rs = avg_gain / avg_loss.replace(0, np.nan)

    rsi = 100 - (
        100 / (1 + rs)
    )

    # If losses are zero, RSI should be 100.
    rsi = rsi.mask(
        (avg_loss == 0) & (avg_gain > 0),
        100.0,
    )

    # If both are zero, price is flat.
    rsi = rsi.mask(
        (avg_gain == 0) & (avg_loss == 0),
        50.0,
    )

    return rsi


def _atr(
    df: pd.DataFrame,
    period: int = ATR_PERIOD,
) -> pd.Series:

    previous_close = df["Close"].shift(1)

    tr = pd.concat(
        [
            df["High"] - df["Low"],
            (df["High"] - previous_close).abs(),
            (df["Low"] - previous_close).abs(),
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

    out["ATR_1M"] = _atr(out)

    out["RSI_1M"] = _rsi(
        out["Close"]
    )

    out["EMA9_1M"] = _ema(
        out["Close"],
        EMA_FAST_1M,
    )

    out["EMA21_1M"] = _ema(
        out["Close"],
        EMA_SLOW_1M,
    )

    out["RVOL_1M"] = (
        out["Volume"]
        /
        out["Volume"]
        .shift(1)
        .rolling(RVOL_PERIOD)
        .mean()
    )

    out["PREV_HIGH_5"] = (
        out["High"]
        .shift(1)
        .rolling(5)
        .max()
    )

    out["PREV_LOW_5"] = (
        out["Low"]
        .shift(1)
        .rolling(5)
        .min()
    )

    out["PREV_RSI_1M"] = (
        out["RSI_1M"].shift(1)
    )

    return out


# ============================================================================
# 15M
# ============================================================================

def calculate_15m_indicators(
    df: pd.DataFrame,
) -> pd.DataFrame:

    out = df.copy()

    out["EMA9_15M"] = _ema(
        out["Close"],
        EMA_FAST_15M,
    )

    out["EMA21_15M"] = _ema(
        out["Close"],
        EMA_SLOW_15M,
    )

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
        EMA_FAST_4H,
    )

    out["EMA50_4H"] = _ema(
        out["Close"],
        EMA_SLOW_4H,
    )

    out["PREV_EMA20_4H"] = (
        out["EMA20_4H"].shift(1)
    )

    out["PREV_EMA50_4H"] = (
        out["EMA50_4H"].shift(1)
    )

    return out


# ============================================================================
# ALIGN
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

    # A higher-timeframe candle becomes
    # usable only AFTER its timestamped candle
    # has completed.
    #
    # We shift the indicator values forward
    # one candle so merge_asof cannot expose
    # the currently forming candle.

    fifteen = fifteen[
        [
            "EMA9_15M",
            "EMA21_15M",
            "MACD_15M",
            "MACD_SIGNAL_15M",
            "MACD_DIFF_15M",
            "PREV_MACD_DIFF_15M",
        ]
    ].shift(1)

    four = four[
        [
            "EMA20_4H",
            "EMA50_4H",
            "PREV_EMA20_4H",
            "PREV_EMA50_4H",
        ]
    ].shift(1)

    one = pd.merge_asof(
        one.sort_index(),
        fifteen.sort_index(),
        left_index=True,
        right_index=True,
        direction="backward",
    )

    one = pd.merge_asof(
        one.sort_index(),
        four.sort_index(),
        left_index=True,
        right_index=True,
        direction="backward",
    )

    return one


# ============================================================================
# SIGNAL
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
        "RVOL_1M",
        "PREV_HIGH_5",
        "EMA9_15M",
        "EMA21_15M",
        "MACD_DIFF_15M",
        "PREV_MACD_DIFF_15M",
        "EMA20_4H",
        "EMA50_4H",
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
                0,
                f"missing_{column}",
            )

    close = float(row["Close"])
    high = float(row["High"])
    low = float(row["Low"])

    atr = float(row["ATR_1M"])

    if close <= 0:
        return Signal(
            False,
            0,
            "invalid_close",
        )

    if atr <= 0:
        return Signal(
            False,
            0,
            "invalid_atr",
        )

    # ========================================================================
    # HARD REGIME
    # ========================================================================

    ema20_4h = float(
        row["EMA20_4H"]
    )

    ema50_4h = float(
        row["EMA50_4H"]
    )

    prev_ema20 = float(
        row["PREV_EMA20_4H"]
    )

    if ema20_4h <= ema50_4h:
        return Signal(
            False,
            0,
            "4h_bearish",
        )

    if ema20_4h <= prev_ema20:
        return Signal(
            False,
            0,
            "4h_slope_negative",
        )

    if close <= ema20_4h:
        return Signal(
            False,
            0,
            "below_4h_ema",
        )

    # ========================================================================
    # EXTENSION
    # ========================================================================

    extension = (
        close - ema20_4h
    ) / atr

    if extension > MAX_EXTENSION_ATR:
        return Signal(
            False,
            0,
            "too_extended",
        )

    # ========================================================================
    # 15M STRUCTURE
    # ========================================================================

    ema9_15 = float(
        row["EMA9_15M"]
    )

    ema21_15 = float(
        row["EMA21_15M"]
    )

    macd = float(
        row["MACD_DIFF_15M"]
    )

    previous_macd = float(
        row["PREV_MACD_DIFF_15M"]
    )

    if ema9_15 <= ema21_15:
        return Signal(
            False,
            0,
            "15m_trend_bearish",
        )

    if macd <= 0:
        return Signal(
            False,
            0,
            "15m_macd_negative",
        )

    # ========================================================================
    # 1M CANDLE QUALITY
    # ========================================================================

    candle_range = high - low

    if candle_range > MAX_CANDLE_ATR * atr:
        return Signal(
            False,
            0,
            "oversized_1m_candle",
        )

    # ========================================================================
    # PULLBACK
    # ========================================================================

    ema21 = float(
        row["EMA21_1M"]
    )

    pullback_distance = (
        abs(close - ema21)
        / atr
    )

    pullback_ok = (
        PULLBACK_ATR_MIN
        <= pullback_distance
        <= PULLBACK_ATR_MAX
    )

    # ========================================================================
    # SCORE
    # ========================================================================

    score = 0

    # 4H regime quality
    score += 2

    # 15M trend
    score += 2

    # MACD positive
    score += 1

    # MACD improving
    if macd > previous_macd:
        score += 1

    # 1M trend
    ema9 = float(
        row["EMA9_1M"]
    )

    if close > ema9 > ema21:
        score += 2

    # RSI
    rsi = float(
        row["RSI_1M"]
    )

    if RSI_MIN <= rsi <= RSI_MAX:
        score += 1

    # Volume
    rvol = float(
        row["RVOL_1M"]
    )

    if rvol >= 1.20:
        score += 2

    elif rvol >= 0.75:
        score += 1

    # Pullback
    if pullback_ok:
        score += 1

    # ========================================================================
    # ACTUAL RECLAIM / TRIGGER
    # ========================================================================

    previous_high = float(
        row["PREV_HIGH_5"]
    )

    reclaim = (
        close > ema9
        and close >= previous_high
    )

    if not reclaim:
        return Signal(
            False,
            score,
            "no_1m_reclaim",
        )

    score += 2

    # ========================================================================
    # FINAL SCORE
    # ========================================================================

    if score < MIN_SCORE:
        return Signal(
            False,
            score,
            f"score_{score}_below_{MIN_SCORE}",
        )

    # ========================================================================
    # RISK/REWARD
    # ========================================================================

    reward_risk = (
        TARGET_ATR
        / STOP_ATR
    )

    if reward_risk < MIN_REWARD_RISK:
        return Signal(
            False,
            score,
            "rr_below_minimum",
        )

    return Signal(
        True,
        score,
        "V14_LONG",
    )


# ============================================================================
# EXITS
# ============================================================================

def calculate_exit_prices(
    entry_price: float,
    atr: float,
) -> Dict[str, float]:

    if (
        not np.isfinite(entry_price)
        or not np.isfinite(atr)
        or entry_price <= 0
        or atr <= 0
    ):
        raise ValueError(
            "Invalid entry or ATR."
        )

    stop = (
        entry_price
        - STOP_ATR * atr
    )

    target = (
        entry_price
        + TARGET_ATR * atr
    )

    if stop <= 0:
        raise ValueError(
            "Calculated stop <= 0."
        )

    return {
        "stop": float(stop),
        "target": float(target),
    }


# ============================================================================
# COMPATIBILITY
# ============================================================================

def calculate_indicators(
    df: pd.DataFrame,
) -> pd.DataFrame:

    return calculate_1m_indicators(
        df
    )


def check_signal(
    row: pd.Series,
) -> bool:

    return evaluate_signal(
        row
    ).valid
