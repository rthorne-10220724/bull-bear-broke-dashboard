"""
BULL. BEAR AND BROKE - V13.2 STRATEGY
=====================================

V13.2 FIX:
    - 1M execution
    - 15M confirmation calculated from real 15M data
    - 4H regime calculated from real 4H data
    - No fake 4H EMA built from a 7-day 1M window
    - Long-only
    - ATR-based stop/target

The signal engine deliberately stays conservative.
V13.2 is primarily a DATA / TIMEFRAME integrity correction.
"""

from __future__ import annotations

from dataclasses import dataclass
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

ATR_STOP_MULT = 1.25
ATR_TARGET_MULT = 2.00

RSI_MIN = 52.0
RSI_MAX = 72.0


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

def _ema(series: pd.Series, period: int) -> pd.Series:
    return series.ewm(
        span=period,
        adjust=False,
        min_periods=period,
    ).mean()


def _rsi(close: pd.Series, period: int = 14) -> pd.Series:

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

    rs = avg_gain / avg_loss.replace(0, np.nan)

    return 100 - (100 / (1 + rs))


def _atr(df: pd.DataFrame, period: int = 14) -> pd.Series:

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
# 1-MINUTE INDICATORS
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

    return out


# ============================================================================
# 15-MINUTE INDICATORS
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
    out["MACD_DIFF_15M"] = macd - signal

    return out


# ============================================================================
# 4-HOUR INDICATORS
# ============================================================================

def calculate_4h_indicators(
    df: pd.DataFrame,
) -> pd.DataFrame:

    out = df.copy()

    out["EMA20_4H"] = _ema(
        out["Close"],
        EMA_4H_PERIOD,
    )

    return out


# ============================================================================
# ALIGN MULTI-TIMEFRAME DATA
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

    # Only use completed higher-timeframe candles.
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

    # Shift higher timeframe values one candle.
    # This prevents using a currently-forming 15M/4H candle.
    fifteen = fifteen.shift(1)
    four = four.shift(1)

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
# BACKWARD-COMPATIBILITY WRAPPER
# ============================================================================

def calculate_indicators(
    df: pd.DataFrame,
) -> pd.DataFrame:

    """
    Compatibility helper.

    This function can still be called by older code, but it only calculates
    native 1M indicators. V13.2 backtest should use align_timeframes().
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
        "ATR_1M",
        "RSI_1M",
        "MACD_DIFF_15M",
        "EMA20_4H",
    ]

    for column in required:

        value = row.get(column, np.nan)

        if not np.isfinite(value):

            return Signal(
                False,
                f"missing_{column}",
            )

    close = float(row["Close"])
    atr = float(row["ATR_1M"])
    rsi = float(row["RSI_1M"])
    macd_diff = float(row["MACD_DIFF_15M"])
    ema20_4h = float(row["EMA20_4H"])

    if close <= 0:
        return Signal(False, "invalid_close")

    if atr <= 0:
        return Signal(False, "invalid_atr")

    # ------------------------------------------------------------------------
    # V13 regime:
    # Price must be above completed 4H EMA20.
    # ------------------------------------------------------------------------

    regime_ok = (
        close > ema20_4h
    )

    if not regime_ok:
        return Signal(False, "below_4h_ema")

    # ------------------------------------------------------------------------
    # 15M momentum confirmation.
    # ------------------------------------------------------------------------

    momentum_ok = (
        macd_diff > 0
    )

    if not momentum_ok:
        return Signal(False, "negative_15m_macd")

    # ------------------------------------------------------------------------
    # 1M RSI confirmation.
    # ------------------------------------------------------------------------

    rsi_ok = (
        RSI_MIN <= rsi <= RSI_MAX
    )

    if not rsi_ok:
        return Signal(False, "rsi_filter")

    # ------------------------------------------------------------------------
    # Avoid extremely stretched candles.
    # ------------------------------------------------------------------------

    candle_range = (
        float(row["High"])
        - float(row["Low"])
    )

    if candle_range > 3.0 * atr:
        return Signal(
            False,
            "oversized_candle",
        )

    return Signal(
        True,
        "V13.2_LONG",
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
