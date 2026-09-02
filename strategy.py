"""
Bull. Bear and Broke — Unified Strategy Engine v10

Design goals:
- One signal definition for backtest + live trading.
- No look-ahead in the 4H trend.
- Volume spike uses PRIOR bars only.
- ATR-based risk management.
- Trend + pullback + momentum confirmation.
- Fail-closed when required indicators are unavailable.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np
import pandas as pd
import ta


# ============================================================
# CONFIG
# ============================================================

RSI_WINDOW = 14

EMA_FAST_1M = 9
EMA_SLOW_1M = 21

EMA_FAST_4H = 20
EMA_SLOW_4H = 50

ATR_WINDOW = 14

# RSI is NOT simply "RSI < 30".
# We want a reversal from weakness, not a falling knife.
RSI_OVERSOLD = 38
RSI_RECOVERY = 42

VOLUME_SPIKE_MULTIPLIER = 1.20

ATR_STOP_MULTIPLIER = 1.50
ATR_TARGET_MULTIPLIER = 3.00

MAX_POSITION_PCT = 0.15
RISK_PER_TRADE_PCT = 0.01


# ============================================================
# HELPERS
# ============================================================

def _flatten_columns(df: pd.DataFrame, symbol: str = "") -> pd.DataFrame:
    """
    Normalize yfinance MultiIndex output.
    """
    if not isinstance(df.columns, pd.MultiIndex):
        return df.copy()

    # Try symbol in either level.
    if symbol:
        for level in range(df.columns.nlevels):
            values = df.columns.get_level_values(level)
            if symbol in values:
                try:
                    return df.xs(symbol, level=level, axis=1).copy()
                except Exception:
                    pass

    # If still MultiIndex, use the first level.
    out = df.copy()
    out.columns = out.columns.get_level_values(0)
    return out


def _required_columns(df: pd.DataFrame) -> bool:
    required = {"Open", "High", "Low", "Close", "Volume"}
    return required.issubset(df.columns)


# ============================================================
# 4H TREND
# ============================================================

def calculate_4h_trend(df: pd.DataFrame) -> pd.Series:
    """
    Calculate a non-lookahead 4H trend.

    Important:
    The currently forming 4H candle is removed before the trend
    is calculated. Therefore the signal only sees completed 4H bars.
    """

    close_4h = (
        df["Close"]
        .resample("4h", origin="start")
        .last()
        .dropna()
    )

    if len(close_4h) < EMA_SLOW_4H + 2:
        return pd.Series(
            index=df.index,
            dtype="object",
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
        dtype="object",
    )

    # Do not allow the incomplete 4H candle.
    trend = trend.iloc[:-1]

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

    df = _flatten_columns(df, symbol)

    if not _required_columns(df):
        raise ValueError(
            "DataFrame missing OHLCV columns."
        )

    df = df.copy()
    df = df.sort_index()

    close = df["Close"]
    high = df["High"]
    low = df["Low"]
    volume = df["Volume"]

    # --------------------------------------------------------
    # RSI
    # --------------------------------------------------------

    df["RSI"] = ta.momentum.rsi(
        close,
        window=RSI_WINDOW,
    )

    df["RSI_PREV"] = df["RSI"].shift(1)

    # Recovery from oversold.
    df["RSI_REVERSAL"] = (
        (df["RSI_PREV"] < RSI_OVERSOLD)
        & (df["RSI"] >= RSI_OVERSOLD)
    )

    df["RSI_RECOVERING"] = (
        (df["RSI"] > df["RSI_PREV"])
        & (df["RSI"] >= RSI_RECOVERY)
    )

    # --------------------------------------------------------
    # Volume
    # --------------------------------------------------------

    # CRITICAL:
    # Use previous 20 bars, NOT the current bar.
    #
    # This prevents the current volume from influencing its own
    # threshold.
    df["VOL_SMA20"] = (
        volume
        .shift(1)
        .rolling(20)
        .mean()
    )

    df["VOL_RATIO"] = (
        volume / df["VOL_SMA20"]
    )

    df["VOL_SPIKE"] = (
        df["VOL_RATIO"] >= VOLUME_SPIKE_MULTIPLIER
    )

    # --------------------------------------------------------
    # Fast momentum
    # --------------------------------------------------------

    df["EMA9"] = ta.trend.ema_indicator(
        close,
        window=EMA_FAST_1M,
    )

    df["EMA21"] = ta.trend.ema_indicator(
        close,
        window=EMA_SLOW_1M,
    )

    df["SHORT_TERM_BULLISH"] = (
        (close > df["EMA9"])
        & (df["EMA9"] >= df["EMA21"])
    )

    # --------------------------------------------------------
    # MACD
    # --------------------------------------------------------

    df["MACD_DIFF"] = ta.trend.macd_diff(
        close,
        window_slow=26,
        window_fast=12,
        window_sign=9,
    )

    df["MACD_PREV"] = df["MACD_DIFF"].shift(1)

    df["MACD_IMPROVING"] = (
        df["MACD_DIFF"] > df["MACD_PREV"]
    )

    # --------------------------------------------------------
    # ATR
    # --------------------------------------------------------

    df["ATR"] = ta.volatility.average_true_range(
        high,
        low,
        close,
        window=ATR_WINDOW,
    )

    df["ATR_PCT"] = (
        df["ATR"] / close
    )

    # --------------------------------------------------------
    # Breakout / reclaim
    # --------------------------------------------------------

    # Previous five completed bars only.
    df["RECENT_HIGH"] = (
        high.shift(1)
        .rolling(5)
        .max()
    )

    df["BREAKOUT"] = (
        close >= df["RECENT_HIGH"]
    )

    # --------------------------------------------------------
    # 4H trend
    # --------------------------------------------------------

    df["TREND_4H"] = calculate_4h_trend(df)

    return df


# ============================================================
# SIGNAL
# ============================================================

def check_signal(
    row: pd.Series,
) -> bool:
    """
    Single source of truth for entry decisions.
    """

    required = [
        "RSI",
        "RSI_PREV",
        "VOL_SPIKE",
        "TREND_4H",
        "MACD_DIFF",
        "MACD_PREV",
        "ATR",
        "SHORT_TERM_BULLISH",
        "BREAKOUT",
    ]

    if any(
        field not in row.index
        for field in required
    ):
        return False

    values = [
        row[field]
        for field in required
    ]

    if any(
        pd.isna(value)
        for value in values
    ):
        return False

    # --------------------------------------------------------
    # HARD TREND GATE
    # --------------------------------------------------------

    if row["TREND_4H"] != "BULLISH":
        return False

    # --------------------------------------------------------
    # RSI
    # --------------------------------------------------------

    rsi_ok = (
        bool(row["RSI_REVERSAL"])
        or bool(row["RSI_RECOVERING"])
    )

    if not rsi_ok:
        return False

    # --------------------------------------------------------
    # MACD
    # --------------------------------------------------------

    if not bool(row["MACD_IMPROVING"]):
        return False

    # --------------------------------------------------------
    # VOLUME
    # --------------------------------------------------------

    if not bool(row["VOL_SPIKE"]):
        return False

    # --------------------------------------------------------
    # PRICE CONFIRMATION
    # --------------------------------------------------------

    price_ok = (
        bool(row["SHORT_TERM_BULLISH"])
        or bool(row["BREAKOUT"])
    )

    if not price_ok:
        return False

    # --------------------------------------------------------
    # VOLATILITY SANITY CHECK
    # --------------------------------------------------------

    if float(row["ATR"]) <= 0:
        return False

    return True


# ============================================================
# SIGNAL REASONS
# ============================================================

def signal_reasons(
    row: pd.Series,
) -> list[str]:

    reasons = []

    if row.get("TREND_4H") == "BULLISH":
        reasons.append("4H bullish")

    if row.get("RSI_REVERSAL") or row.get("RSI_RECOVERING"):
        reasons.append("RSI recovery")

    if row.get("MACD_IMPROVING"):
        reasons.append("MACD improving")

    if row.get("VOL_SPIKE"):
        reasons.append("volume spike")

    if (
        row.get("SHORT_TERM_BULLISH")
        or row.get("BREAKOUT")
    ):
        reasons.append("price confirmation")

    return reasons


# ============================================================
# POSITION SIZE
# ============================================================

def calculate_position_size(
    equity: float,
    price: float,
    atr: float,
    risk_pct: float = RISK_PER_TRADE_PCT,
    max_position_pct: float = MAX_POSITION_PCT,
) -> int:

    if (
        equity <= 0
        or price <= 0
        or atr <= 0
    ):
        return 0

    risk_dollars = (
        equity * risk_pct
    )

    stop_distance = (
        ATR_STOP_MULTIPLIER * atr
    )

    if stop_distance <= 0:
        return 0

    shares_by_risk = (
        risk_dollars / stop_distance
    )

    shares_by_cap = (
        equity * max_position_pct
    ) / price

    qty = min(
        shares_by_risk,
        shares_by_cap,
    )

    return max(
        0,
        int(qty),
    )


# ============================================================
# TRADE LEVELS
# ============================================================

@dataclass
class TradeLevels:
    entry: float
    stop: float
    target: float


def calculate_trade_levels(
    entry_price: float,
    atr: float,
) -> Optional[TradeLevels]:

    if (
        entry_price <= 0
        or atr <= 0
    ):
        return None

    stop = (
        entry_price
        - ATR_STOP_MULTIPLIER * atr
    )

    target = (
        entry_price
        + ATR_TARGET_MULTIPLIER * atr
    )

    if stop <= 0:
        return None

    return TradeLevels(
        entry=float(entry_price),
        stop=float(stop),
        target=float(target),
    )
