"""
BULL. BEAR AND BROKE - v14 SHARED STRATEGY ENGINE

Single source of truth for:
- Indicator calculation
- Signal generation
- Position sizing
- Stop/target calculation

The live engine and backtester should both import this module.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple

import numpy as np
import pandas as pd
import ta


# ============================================================
# CONFIGURATION
# ============================================================

RSI_WINDOW = 14

EMA_FAST_1M = 9
EMA_SLOW_1M = 21

EMA_FAST_4H = 20
EMA_SLOW_4H = 50

MACD_FAST = 12
MACD_SLOW = 26
MACD_SIGNAL = 9

VOLUME_WINDOW = 20
VOLUME_SPIKE_MULTIPLIER = 1.20

RSI_HARD_FLOOR = 50.0
RSI_OVERSOLD = 40.0
RSI_REVERSAL = 45.0

ATR_WINDOW = 14

ATR_STOP_MULTIPLIER = 1.50
ATR_TARGET_MULTIPLIER = 5.25

RISK_PER_TRADE_PCT = 0.01
MAX_POSITION_PCT = 0.15


# ============================================================
# HELPERS
# ============================================================

def _flatten_yfinance_columns(
    df: pd.DataFrame,
    symbol: str = "",
) -> pd.DataFrame:

    df = df.copy()

    if not isinstance(df.columns, pd.MultiIndex):
        return df

    # Try common yfinance layouts.
    if symbol:
        for level in range(df.columns.nlevels):
            values = df.columns.get_level_values(level)

            if symbol in values:
                try:
                    result = df.xs(symbol, level=level, axis=1)
                    if not isinstance(result.columns, pd.MultiIndex):
                        return result
                except Exception:
                    pass

    # If still multi-index, use first level.
    df.columns = [
        col[0] if isinstance(col, tuple) else col
        for col in df.columns
    ]

    return df


def _normalize_columns(df: pd.DataFrame) -> pd.DataFrame:

    df = df.copy()

    rename_map = {}

    for col in df.columns:
        lower = str(col).lower()

        if lower == "open":
            rename_map[col] = "Open"
        elif lower == "high":
            rename_map[col] = "High"
        elif lower == "low":
            rename_map[col] = "Low"
        elif lower == "close":
            rename_map[col] = "Close"
        elif lower == "volume":
            rename_map[col] = "Volume"

    df = df.rename(columns=rename_map)

    required = ["Open", "High", "Low", "Close", "Volume"]

    missing = [x for x in required if x not in df.columns]

    if missing:
        raise ValueError(
            f"Missing required OHLCV columns: {missing}"
        )

    return df


def _true_range(
    high: pd.Series,
    low: pd.Series,
    close: pd.Series,
) -> pd.Series:

    previous_close = close.shift(1)

    return pd.concat(
        [
            high - low,
            (high - previous_close).abs(),
            (low - previous_close).abs(),
        ],
        axis=1,
    ).max(axis=1)


def calculate_atr(
    df: pd.DataFrame,
    window: int = ATR_WINDOW,
) -> pd.Series:

    tr = _true_range(
        df["High"],
        df["Low"],
        df["Close"],
    )

    return tr.rolling(window).mean()


# ============================================================
# 4H TREND
# ============================================================

def calculate_completed_4h_trend(
    df: pd.DataFrame,
) -> pd.Series:

    """
    Calculates 4H trend using ONLY completed 4H candles.

    The resulting trend is shifted forward so a lower-timeframe
    bar can only see information from the most recently completed
    4H candle.
    """

    close_4h = (
        df["Close"]
        .resample("4h", origin="start")
        .last()
        .dropna()
    )

    if close_4h.empty:
        return pd.Series(
            index=df.index,
            dtype=object,
        )

    ema20 = ta.trend.ema_indicator(
        close_4h,
        window=EMA_FAST_4H,
    )

    ema50 = ta.trend.ema_indicator(
        close_4h,
        window=EMA_SLOW_4H,
    )

    trend = pd.Series(
        np.where(
            ema20 > ema50,
            "BULLISH",
            "BEARISH",
        ),
        index=close_4h.index,
        dtype=object,
    )

    # A completed candle's information becomes usable only after
    # that candle has closed.
    trend = trend.shift(1)

    return trend.reindex(
        df.index,
        method="ffill",
    )


# ============================================================
# INDICATORS
# ============================================================

def calculate_indicators(
    df: pd.DataFrame,
    symbol: str = "",
) -> pd.DataFrame:

    df = _flatten_yfinance_columns(
        df,
        symbol=symbol,
    )

    df = _normalize_columns(df)

    df = df.copy()

    if df.index.tz is None:
        df.index = df.index.tz_localize("UTC")

    df = df.sort_index()

    # --------------------------------------------------------
    # RSI
    # --------------------------------------------------------

    df["RSI"] = ta.momentum.rsi(
        df["Close"],
        window=RSI_WINDOW,
    )

    df["PREVIOUS_RSI"] = df["RSI"].shift(1)

    df["RSI_REVERSAL"] = (
        (df["PREVIOUS_RSI"] < RSI_OVERSOLD)
        & (df["RSI"] >= RSI_OVERSOLD)
    )

    df["RSI_RECOVERING"] = (
        (df["RSI"] > df["PREVIOUS_RSI"])
        & (df["RSI"] >= RSI_REVERSAL)
    )

    # --------------------------------------------------------
    # Volume
    # --------------------------------------------------------

    # Shift first so the current candle does not influence its
    # own volume baseline.
    df["VOL_SMA20"] = (
        df["Volume"]
        .shift(1)
        .rolling(VOLUME_WINDOW)
        .mean()
    )

    df["RVOL"] = (
        df["Volume"] / df["VOL_SMA20"]
    )

    df["VOL_SPIKE"] = (
        df["Volume"]
        >= VOLUME_SPIKE_MULTIPLIER * df["VOL_SMA20"]
    )

    # --------------------------------------------------------
    # 1-minute EMA structure
    # --------------------------------------------------------

    df["EMA9"] = ta.trend.ema_indicator(
        df["Close"],
        window=EMA_FAST_1M,
    )

    df["EMA21"] = ta.trend.ema_indicator(
        df["Close"],
        window=EMA_SLOW_1M,
    )

    df["SHORT_TERM_BULLISH"] = (
        (df["Close"] > df["EMA9"])
        & (df["EMA9"] >= df["EMA21"])
    )

    # --------------------------------------------------------
    # Recent breakout
    # --------------------------------------------------------

    df["RECENT_HIGH"] = (
        df["High"]
        .shift(1)
        .rolling(5)
        .max()
    )

    df["BREAKOUT_CONFIRMATION"] = (
        df["Close"] >= df["RECENT_HIGH"]
    )

    # --------------------------------------------------------
    # 15-minute MACD
    # --------------------------------------------------------

    close_15m = (
        df["Close"]
        .resample("15min")
        .last()
        .dropna()
    )

    macd_diff_15m = ta.trend.macd_diff(
        close_15m,
        window_slow=MACD_SLOW,
        window_fast=MACD_FAST,
        window_sign=MACD_SIGNAL,
    )

    macd_improving_15m = (
        macd_diff_15m
        >= macd_diff_15m.shift(1)
    )

    df["MACD_DIFF"] = (
        macd_diff_15m
        .reindex(df.index, method="ffill")
    )

    df["MACD_IMPROVING"] = (
        macd_improving_15m
        .reindex(df.index, method="ffill")
        .fillna(False)
    )

    # --------------------------------------------------------
    # ATR
    # --------------------------------------------------------

    df["ATR"] = calculate_atr(
        df,
        window=ATR_WINDOW,
    )

    df["ATR_PCT"] = (
        df["ATR"] / df["Close"]
    ) * 100.0

    # --------------------------------------------------------
    # 4H trend
    # --------------------------------------------------------

    df["TREND_4H"] = (
        calculate_completed_4h_trend(df)
    )

    # --------------------------------------------------------
    # Final price
    # --------------------------------------------------------

    df["LATEST_PRICE"] = df["Close"]

    return df


# ============================================================
# SIGNAL
# ============================================================

def check_signal(
    row: pd.Series,
) -> Tuple[bool, Dict[str, Any]]:

    reasons = []

    # --------------------------------------------------------
    # Hard 4H trend gate
    # --------------------------------------------------------

    if row.get("TREND_4H") != "BULLISH":
        return False, {
            "reason": "4H trend not bullish",
            "score": 0,
            "reasons": reasons,
        }

    reasons.append("4H bullish")

    # --------------------------------------------------------
    # RSI
    # --------------------------------------------------------

    rsi = row.get("RSI")

    if pd.isna(rsi):
        return False, {
            "reason": "RSI unavailable",
            "score": 0,
            "reasons": reasons,
        }

    rsi_ok = bool(
        row.get("RSI_REVERSAL", False)
        or row.get("RSI_RECOVERING", False)
        or rsi >= RSI_HARD_FLOOR
    )

    if not rsi_ok:
        return False, {
            "reason": "RSI momentum insufficient",
            "score": 0,
            "reasons": reasons,
        }

    reasons.append("RSI favorable")

    # --------------------------------------------------------
    # MACD
    # --------------------------------------------------------

    if not bool(row.get("MACD_IMPROVING", False)):
        return False, {
            "reason": "MACD not improving",
            "score": 0,
            "reasons": reasons,
        }

    reasons.append("MACD improving")

    # --------------------------------------------------------
    # Volume
    # --------------------------------------------------------

    if not bool(row.get("VOL_SPIKE", False)):
        return False, {
            "reason": "No volume confirmation",
            "score": 0,
            "reasons": reasons,
        }

    reasons.append("Volume spike")

    # --------------------------------------------------------
    # Price confirmation
    # --------------------------------------------------------

    price_confirmation = bool(
        row.get("SHORT_TERM_BULLISH", False)
        or row.get("BREAKOUT_CONFIRMATION", False)
    )

    if not price_confirmation:
        return False, {
            "reason": "No price confirmation",
            "score": 0,
            "reasons": reasons,
        }

    reasons.append("Price confirmation")

    return True, {
        "reason": "Signal valid",
        "score": len(reasons),
        "reasons": reasons,
    }


# ============================================================
# POSITION SIZING
# ============================================================

def calculate_position_size(
    equity: float,
    entry_price: float,
    atr: float,
) -> int:

    if (
        equity <= 0
        or entry_price <= 0
        or pd.isna(atr)
        or atr <= 0
    ):
        return 0

    risk_dollars = (
        equity * RISK_PER_TRADE_PCT
    )

    stop_distance = (
        ATR_STOP_MULTIPLIER * atr
    )

    shares_by_risk = (
        risk_dollars / stop_distance
    )

    max_position_dollars = (
        equity * MAX_POSITION_PCT
    )

    shares_by_cap = (
        max_position_dollars / entry_price
    )

    shares = min(
        shares_by_risk,
        shares_by_cap,
    )

    return max(
        0,
        int(np.floor(shares)),
    )


# ============================================================
# EXIT LEVELS
# ============================================================

def calculate_exit_levels(
    entry_price: float,
    atr: float,
) -> Tuple[float, float]:

    if (
        entry_price <= 0
        or pd.isna(atr)
        or atr <= 0
    ):
        return 0.0, 0.0

    stop = (
        entry_price
        - ATR_STOP_MULTIPLIER * atr
    )

    target = (
        entry_price
        + ATR_TARGET_MULTIPLIER * atr
    )

    return (
        float(stop),
        float(target),
    )
