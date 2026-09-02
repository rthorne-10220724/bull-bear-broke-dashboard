"""
BULL. BEAR AND BROKE - STRATEGY v13
====================================

Regime-adaptive pullback/reclaim strategy.

Architecture:
    1. Data quality
    2. 4H regime gate
    3. 15M momentum/setup confirmation
    4. 1M execution trigger
    5. Setup score
    6. ATR-based risk levels

IMPORTANT:
    This is an experimental strategy.
    It is NOT guaranteed profitable.

The backtester should be the judge.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
import ta


# ============================================================================
# CONFIG
# ============================================================================

RSI_WINDOW = 14

EMA_FAST_1M = 9
EMA_SLOW_1M = 21

EMA_FAST_15M = 9
EMA_SLOW_15M = 21

EMA_FAST_4H = 20
EMA_SLOW_4H = 50

MACD_FAST = 12
MACD_SLOW = 26
MACD_SIGNAL = 9

ATR_WINDOW = 14

MIN_RVOL = 0.75

# Signal quality
MIN_SIGNAL_SCORE = 7

# Do not chase extremely extended price.
MAX_EXTENSION_ATR = 1.75

# Pullback must remain structurally healthy.
MAX_PULLBACK_ATR = 1.25

# Minimum reward/risk before entering.
MIN_REWARD_RISK = 1.80

# ATR exits
STOP_ATR = 1.50
TARGET_ATR = 3.00


# ============================================================================
# HELPERS
# ============================================================================

def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        value = float(value)
        return value if np.isfinite(value) else default
    except (TypeError, ValueError):
        return default


def _rvol(volume: pd.Series, window: int = 20) -> float:
    """
    Relative volume using ONLY completed prior bars as baseline.
    """
    if len(volume) < window + 1:
        return 0.0

    baseline = volume.iloc[-window - 1:-1].mean()

    if baseline <= 0 or not np.isfinite(baseline):
        return 0.0

    return _safe_float(volume.iloc[-1] / baseline)


def _atr(df: pd.DataFrame, window: int = ATR_WINDOW) -> float:
    """
    Wilder-style ATR through the ta library.
    """
    if len(df) < window + 2:
        return 0.0

    atr = ta.volatility.average_true_range(
        high=df["High"],
        low=df["Low"],
        close=df["Close"],
        window=window,
    )

    return _safe_float(atr.iloc[-1])


def _resample_ohlcv(
    df: pd.DataFrame,
    rule: str,
) -> pd.DataFrame:

    result = (
        df.resample(rule)
        .agg(
            {
                "Open": "first",
                "High": "max",
                "Low": "min",
                "Close": "last",
                "Volume": "sum",
            }
        )
        .dropna()
    )

    return result


# ============================================================================
# INDICATORS
# ============================================================================

def calculate_indicators(
    df: pd.DataFrame,
    symbol: str = "",
) -> pd.DataFrame:
    """
    Shared indicator calculation.

    Expected input:
        Open High Low Close Volume

    Input timeframe:
        1-minute bars.

    Higher timeframes are derived internally.
    """

    if df is None or df.empty:
        return pd.DataFrame()

    df = df.copy()

    # Handle yfinance MultiIndex.
    if isinstance(df.columns, pd.MultiIndex) and symbol:

        try:
            if symbol in df.columns.get_level_values(-1):
                df = df.xs(
                    symbol,
                    level=-1,
                    axis=1,
                )
            elif symbol in df.columns.get_level_values(0):
                df = df.xs(
                    symbol,
                    level=0,
                    axis=1,
                )
        except Exception:
            pass

    required = [
        "Open",
        "High",
        "Low",
        "Close",
        "Volume",
    ]

    if not all(column in df.columns for column in required):
        return pd.DataFrame()

    df = df[required].copy()

    for column in required:
        df[column] = pd.to_numeric(
            df[column],
            errors="coerce",
        )

    df = df.dropna()

    if len(df) < 100:
        return pd.DataFrame()

    # ------------------------------------------------------------------
    # 1M indicators
    # ------------------------------------------------------------------

    df["RSI_1M"] = ta.momentum.rsi(
        df["Close"],
        window=RSI_WINDOW,
    )

    df["EMA9_1M"] = ta.trend.ema_indicator(
        df["Close"],
        window=EMA_FAST_1M,
    )

    df["EMA21_1M"] = ta.trend.ema_indicator(
        df["Close"],
        window=EMA_SLOW_1M,
    )

    df["ATR_1M"] = ta.volatility.average_true_range(
        df["High"],
        df["Low"],
        df["Close"],
        window=ATR_WINDOW,
    )

    # RVOL against prior completed bars.
    df["RVOL"] = (
        df["Volume"]
        / df["Volume"]
        .shift(1)
        .rolling(20)
        .mean()
    )

    # ------------------------------------------------------------------
    # 15M indicators
    # ------------------------------------------------------------------

    df15 = _resample_ohlcv(df, "15min")

    if len(df15) >= 60:

        df15["EMA9"] = ta.trend.ema_indicator(
            df15["Close"],
            window=EMA_FAST_15M,
        )

        df15["EMA21"] = ta.trend.ema_indicator(
            df15["Close"],
            window=EMA_SLOW_15M,
        )

        df15["RSI"] = ta.momentum.rsi(
            df15["Close"],
            window=RSI_WINDOW,
        )

        macd = ta.trend.MACD(
            close=df15["Close"],
            window_fast=MACD_FAST,
            window_slow=MACD_SLOW,
            window_sign=MACD_SIGNAL,
        )

        df15["MACD"] = macd.macd()
        df15["MACD_SIGNAL"] = macd.macd_signal()
        df15["MACD_DIFF"] = macd.macd_diff()

        # Align ONLY completed 15M candles.
        df15 = df15.shift(1)

        df["EMA9_15M"] = df15["EMA9"].reindex(
            df.index,
            method="ffill",
        )

        df["EMA21_15M"] = df15["EMA21"].reindex(
            df.index,
            method="ffill",
        )

        df["RSI_15M"] = df15["RSI"].reindex(
            df.index,
            method="ffill",
        )

        df["MACD_15M"] = df15["MACD"].reindex(
            df.index,
            method="ffill",
        )

        df["MACD_SIGNAL_15M"] = df15[
            "MACD_SIGNAL"
        ].reindex(
            df.index,
            method="ffill",
        )

        df["MACD_DIFF_15M"] = df15[
            "MACD_DIFF"
        ].reindex(
            df.index,
            method="ffill",
        )

    else:

        for column in [
            "EMA9_15M",
            "EMA21_15M",
            "RSI_15M",
            "MACD_15M",
            "MACD_SIGNAL_15M",
            "MACD_DIFF_15M",
        ]:
            df[column] = np.nan

    # ------------------------------------------------------------------
    # 4H indicators
    # ------------------------------------------------------------------

    df4 = _resample_ohlcv(df, "4h")

    if len(df4) >= EMA_SLOW_4H + 5:

        df4["EMA20"] = ta.trend.ema_indicator(
            df4["Close"],
            window=EMA_FAST_4H,
        )

        df4["EMA50"] = ta.trend.ema_indicator(
            df4["Close"],
            window=EMA_SLOW_4H,
        )

        df4["ATR"] = ta.volatility.average_true_range(
            df4["High"],
            df4["Low"],
            df4["Close"],
            window=ATR_WINDOW,
        )

        # IMPORTANT:
        # Remove the currently forming 4H candle.
        df4 = df4.shift(1)

        df["EMA20_4H"] = df4["EMA20"].reindex(
            df.index,
            method="ffill",
        )

        df["EMA50_4H"] = df4["EMA50"].reindex(
            df.index,
            method="ffill",
        )

        df["ATR_4H"] = df4["ATR"].reindex(
            df.index,
            method="ffill",
        )

        df["CLOSE_4H"] = df4["Close"].reindex(
            df.index,
            method="ffill",
        )

    else:

        for column in [
            "EMA20_4H",
            "EMA50_4H",
            "ATR_4H",
            "CLOSE_4H",
        ]:
            df[column] = np.nan

    return df


# ============================================================================
# SIGNAL
# ============================================================================

@dataclass
class SignalResult:
    valid: bool
    score: int
    reasons: List[str]
    rejection: Optional[str] = None


def check_signal(
    row: pd.Series,
) -> bool:
    """
    Compatibility function.

    Returns only whether V13 has a valid entry.
    """

    return evaluate_signal(row).valid


def evaluate_signal(
    row: pd.Series,
) -> SignalResult:

    reasons: List[str] = []
    score = 0

    price = _safe_float(row.get("Close"))
    atr = _safe_float(row.get("ATR_1M"))

    if price <= 0:
        return SignalResult(
            False,
            0,
            [],
            "invalid price",
        )

    if atr <= 0:
        return SignalResult(
            False,
            0,
            [],
            "invalid ATR",
        )

    # ------------------------------------------------------------------
    # HARD GATE 1 — 4H trend
    # ------------------------------------------------------------------

    ema20_4h = _safe_float(row.get("EMA20_4H"))
    ema50_4h = _safe_float(row.get("EMA50_4H"))

    if ema20_4h <= ema50_4h:
        return SignalResult(
            False,
            0,
            [],
            "4H trend bearish",
        )

    reasons.append("4H bullish")

    # ------------------------------------------------------------------
    # HARD GATE 2 — 4H slope
    # ------------------------------------------------------------------
    #
    # Slope confirmation is represented by the current EMA relationship
    # and the previous completed 4H trend state when available.
    #
    # The backtester can additionally validate regime persistence.
    # ------------------------------------------------------------------

    previous_ema20 = _safe_float(
        row.get("PREV_EMA20_4H")
    )

    if previous_ema20 > 0 and ema20_4h <= previous_ema20:
        return SignalResult(
            False,
            0,
            reasons,
            "4H EMA20 slope not positive",
        )

    reasons.append("4H slope positive")

    # ------------------------------------------------------------------
    # HARD GATE 3 — avoid excessive extension
    # ------------------------------------------------------------------

    extension_atr = (
        (price - ema20_4h) / atr
        if atr > 0
        else 999
    )

    if extension_atr > MAX_EXTENSION_ATR:
        return SignalResult(
            False,
            0,
            reasons,
            f"extended {extension_atr:.2f} ATR",
        )

    # ------------------------------------------------------------------
    # 1M momentum
    # ------------------------------------------------------------------

    rsi = _safe_float(row.get("RSI_1M"))
    previous_rsi = _safe_float(
        row.get("PREV_RSI_1M")
    )

    if rsi >= 50:
        score += 2
        reasons.append("1M RSI bullish")

    elif rsi > previous_rsi and rsi >= 40:
        score += 1
        reasons.append("1M RSI recovering")

    # ------------------------------------------------------------------
    # 15M trend
    # ------------------------------------------------------------------

    ema9_15 = _safe_float(row.get("EMA9_15M"))
    ema21_15 = _safe_float(row.get("EMA21_15M"))

    if (
        ema9_15 > 0
        and ema21_15 > 0
        and ema9_15 > ema21_15
    ):
        score += 2
        reasons.append("15M EMA bullish")

    # ------------------------------------------------------------------
    # 15M MACD
    # ------------------------------------------------------------------

    macd = _safe_float(row.get("MACD_DIFF_15M"))
    previous_macd = _safe_float(
        row.get("PREV_MACD_DIFF_15M")
    )

    if macd > 0:
        score += 1
        reasons.append("15M MACD positive")

    if macd > previous_macd:
        score += 1
        reasons.append("15M MACD improving")

    # ------------------------------------------------------------------
    # 1M EMA structure
    # ------------------------------------------------------------------

    ema9 = _safe_float(row.get("EMA9_1M"))
    ema21 = _safe_float(row.get("EMA21_1M"))

    if (
        price > ema9
        and ema9 > ema21
    ):
        score += 2
        reasons.append("1M reclaim")

    # ------------------------------------------------------------------
    # Pullback quality
    # ------------------------------------------------------------------

    pullback_distance = (
        abs(price - ema21) / atr
        if atr > 0
        else 999
    )

    healthy_pullback = (
        price >= ema21 - MAX_PULLBACK_ATR * atr
        and pullback_distance <= 1.75
    )

    if healthy_pullback:
        score += 1
        reasons.append("healthy pullback")

    # ------------------------------------------------------------------
    # Volume
    # ------------------------------------------------------------------

    rvol = _safe_float(row.get("RVOL"))

    if rvol >= 1.20:
        score += 2
        reasons.append("volume expansion")

    elif rvol >= MIN_RVOL:
        score += 1
        reasons.append("adequate volume")

    # ------------------------------------------------------------------
    # Reclaim / breakout
    # ------------------------------------------------------------------

    previous_high = _safe_float(
        row.get("PREV_5_HIGH")
    )

    if previous_high > 0 and price >= previous_high:
        score += 1
        reasons.append("micro breakout")

    # ------------------------------------------------------------------
    # Minimum score
    # ------------------------------------------------------------------

    if score < MIN_SIGNAL_SCORE:
        return SignalResult(
            False,
            score,
            reasons,
            f"score {score}/{MIN_SIGNAL_SCORE}",
        )

    # ------------------------------------------------------------------
    # Reward/risk sanity check
    # ------------------------------------------------------------------

    stop_distance = STOP_ATR * atr
    target_distance = TARGET_ATR * atr

    if stop_distance <= 0:
        return SignalResult(
            False,
            score,
            reasons,
            "invalid stop distance",
        )

    reward_risk = (
        target_distance / stop_distance
    )

    if reward_risk < MIN_REWARD_RISK:
        return SignalResult(
            False,
            score,
            reasons,
            f"R:R {reward_risk:.2f} below minimum",
        )

    reasons.append(
        f"R:R {reward_risk:.2f}"
    )

    return SignalResult(
        True,
        score,
        reasons,
    )


# ============================================================================
# BACKTEST EXIT HELPERS
# ============================================================================

def calculate_exit_prices(
    entry_price: float,
    atr: float,
) -> Dict[str, float]:

    stop = (
        entry_price
        - STOP_ATR * atr
    )

    target = (
        entry_price
        + TARGET_ATR * atr
    )

    return {
        "stop": float(stop),
        "target": float(target),
    }
