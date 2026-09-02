"""
BULL. BEAR AND BROKE - STRATEGY v12
====================================

Single source of truth for:
    - indicators
    - signal scoring
    - data-quality gates
    - ATR risk
    - position sizing
    - entry/exit calculations

Used by BOTH:
    1. live/paper trading engine
    2. historical backtester

IMPORTANT:
------------
This is a research/trading framework, not a guarantee of profitability.

Core philosophy:
    HARD GATES
        - sufficient data
        - sufficient volatility/liquidity
        - 4H bullish trend
        - positive 4H trend slope
        - no excessive extension

    SETUP SCORE
        - RSI
        - MACD
        - 1m EMA structure
        - healthy pullback
        - breakout
        - volume

    RISK
        - ATR stop
        - ATR target
        - fixed percentage account risk
        - position cap
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import ta


# ============================================================================
# CONFIGURATION
# ============================================================================

RISK_PER_TRADE_PCT = 0.0075       # 0.75% account risk
MAX_POSITION_PCT = 0.15           # 15% max notional allocation

RSI_WINDOW = 14

EMA_FAST_1M = 9
EMA_SLOW_1M = 21

EMA_FAST_4H = 20
EMA_SLOW_4H = 50

MACD_FAST = 12
MACD_SLOW = 26
MACD_SIGNAL = 9

ATR_WINDOW = 14

MIN_SIGNAL_SCORE = 6

MAX_ALLOWED_EXTENSION_ATR = 1.25
MIN_RVOL = 0.70

ATR_MULTIPLIER_STOP = 1.50
ATR_MULTIPLIER_TARGET = 3.75

ENTRY_LIMIT_BUFFER_PCT = 0.0005


# ============================================================================
# DATA NORMALIZATION
# ============================================================================

def normalize_ohlcv(df: pd.DataFrame) -> Optional[pd.DataFrame]:
    """
    Normalize OHLCV column names and numeric types.
    """

    if df is None or df.empty:
        return None

    df = df.copy()

    # Handle common yfinance MultiIndex layout.
    if isinstance(df.columns, pd.MultiIndex):
        # Flatten where possible.
        if len(df.columns.levels) >= 2:
            flattened = []

            for col in df.columns:
                if isinstance(col, tuple):
                    # Prefer the actual OHLCV field.
                    selected = None

                    for value in col:
                        if str(value).lower() in {
                            "open",
                            "high",
                            "low",
                            "close",
                            "volume",
                        }:
                            selected = value
                            break

                    flattened.append(
                        selected if selected is not None else str(col[-1])
                    )
                else:
                    flattened.append(str(col))

            df.columns = flattened

    rename_map = {
        "open": "Open",
        "high": "High",
        "low": "Low",
        "close": "Close",
        "volume": "Volume",
        "adj close": "Adj Close",
    }

    df.rename(
        columns={
            col: rename_map.get(str(col).lower(), col)
            for col in df.columns
        },
        inplace=True,
    )

    required = [
        "Open",
        "High",
        "Low",
        "Close",
        "Volume",
    ]

    if not all(column in df.columns for column in required):
        return None

    df = df[required].copy()

    for column in required:
        df[column] = pd.to_numeric(
            df[column],
            errors="coerce",
        )

    df.replace([np.inf, -np.inf], np.nan, inplace=True)
    df.dropna(inplace=True)

    if len(df) < 30:
        return None

    # Ensure chronological order.
    df = df.sort_index()

    return df


# ============================================================================
# ATR
# ============================================================================

def calculate_atr(
    df: pd.DataFrame,
    window: int = ATR_WINDOW,
) -> float:

    if df is None or len(df) < window + 1:
        return 0.0

    high = df["High"]
    low = df["Low"]
    close = df["Close"]

    previous_close = close.shift(1)

    true_range = pd.concat(
        [
            high - low,
            (high - previous_close).abs(),
            (low - previous_close).abs(),
        ],
        axis=1,
    ).max(axis=1)

    atr = true_range.rolling(window).mean().iloc[-1]

    if pd.isna(atr):
        return 0.0

    return float(atr)


# ============================================================================
# RVOL
# ============================================================================

def calculate_rvol(
    volume: pd.Series,
    window: int = 20,
) -> float:

    if len(volume) < window + 1:
        return 0.0

    # Exclude current bar from baseline.
    baseline = volume.iloc[-window - 1:-1].mean()

    if baseline <= 0 or pd.isna(baseline):
        return 0.0

    return float(volume.iloc[-1] / baseline)


# ============================================================================
# 4H TREND
# ============================================================================

def calculate_4h_indicators(
    df_4h: pd.DataFrame,
) -> Dict[str, Any]:

    if df_4h is None or len(df_4h) < EMA_SLOW_4H + 4:
        return {
            "trend_4h": "UNKNOWN",
            "ema20_4h": 0.0,
            "ema50_4h": 0.0,
            "ema20_slope_positive": False,
        }

    close_4h = df_4h["Close"]

    ema20 = ta.trend.ema_indicator(
        close_4h,
        window=EMA_FAST_4H,
    )

    ema50 = ta.trend.ema_indicator(
        close_4h,
        window=EMA_SLOW_4H,
    )

    ema20_value = float(ema20.iloc[-1])
    ema50_value = float(ema50.iloc[-1])

    if pd.isna(ema20_value) or pd.isna(ema50_value):
        return {
            "trend_4h": "UNKNOWN",
            "ema20_4h": 0.0,
            "ema50_4h": 0.0,
            "ema20_slope_positive": False,
        }

    trend = (
        "BULLISH"
        if ema20_value > ema50_value
        else "BEARISH"
    )

    slope_positive = bool(
        ema20.iloc[-1] > ema20.iloc[-4]
    )

    return {
        "trend_4h": trend,
        "ema20_4h": ema20_value,
        "ema50_4h": ema50_value,
        "ema20_slope_positive": slope_positive,
    }


# ============================================================================
# INDICATOR ANALYSIS
# ============================================================================

def analyze_indicators(
    df_1m: pd.DataFrame,
    df_4h: pd.DataFrame,
) -> Dict[str, Any]:

    df_1m = normalize_ohlcv(df_1m)
    df_4h = normalize_ohlcv(df_4h)

    if df_1m is None:
        raise ValueError("Invalid 1m OHLCV data")

    if df_4h is None:
        raise ValueError("Invalid 4H OHLCV data")

    close = df_1m["Close"]
    high = df_1m["High"]
    low = df_1m["Low"]
    volume = df_1m["Volume"]

    price = float(close.iloc[-1])

    # ------------------------------------------------------------------
    # RSI
    # ------------------------------------------------------------------

    rsi_series = ta.momentum.rsi(
        close,
        window=RSI_WINDOW,
    )

    rsi = float(rsi_series.iloc[-1])
    previous_rsi = float(rsi_series.iloc[-2])

    rsi_recovering = (
        rsi > previous_rsi
        and rsi >= 45
    )

    rsi_bullish = rsi >= 50

    # ------------------------------------------------------------------
    # 1m EMA
    # ------------------------------------------------------------------

    ema9 = ta.trend.ema_indicator(
        close,
        window=EMA_FAST_1M,
    )

    ema21 = ta.trend.ema_indicator(
        close,
        window=EMA_SLOW_1M,
    )

    ema9_now = float(ema9.iloc[-1])
    ema21_now = float(ema21.iloc[-1])

    short_term_bullish = (
        price > ema9_now
        and ema9_now >= ema21_now
    )

    # ------------------------------------------------------------------
    # MACD
    #
    # Calculate from completed 15m closes.
    # ------------------------------------------------------------------

    close_15m = (
        close
        .resample("15min")
        .last()
        .dropna()
    )

    macd_diff = 0.0
    macd_previous = 0.0
    macd_available = False

    if len(close_15m) >= 40:

        macd = ta.trend.macd_diff(
            close_15m,
            window_slow=MACD_SLOW,
            window_fast=MACD_FAST,
            window_sign=MACD_SIGNAL,
        )

        clean_macd = macd.dropna()

        if len(clean_macd) >= 2:

            macd_diff = float(clean_macd.iloc[-1])
            macd_previous = float(clean_macd.iloc[-2])

            macd_available = True

    macd_improving = (
        macd_available
        and macd_diff >= macd_previous
    )

    macd_positive = (
        macd_available
        and macd_diff > 0
    )

    # ------------------------------------------------------------------
    # RVOL
    # ------------------------------------------------------------------

    rvol = calculate_rvol(volume)

    volume_spike = rvol >= 1.20

    # ------------------------------------------------------------------
    # ATR
    # ------------------------------------------------------------------

    atr = calculate_atr(df_1m)

    atr_pct = (
        atr / price * 100
        if price > 0
        else 0.0
    )

    # ------------------------------------------------------------------
    # Micro-breakout
    # ------------------------------------------------------------------

    if len(high) >= 6:
        previous_high = float(
            high.iloc[-6:-1].max()
        )
        breakout = price >= previous_high
    else:
        breakout = False

    # ------------------------------------------------------------------
    # Pullback quality
    # ------------------------------------------------------------------

    if atr > 0:

        distance_from_ema21_atr = (
            abs(price - ema21_now) / atr
        )

        extension_atr = (
            (price - ema21_now) / atr
        )

    else:

        distance_from_ema21_atr = 999.0
        extension_atr = 999.0

    healthy_pullback = (
        price >= ema21_now - (0.75 * atr)
        and distance_from_ema21_atr <= 1.50
    )

    not_overextended = (
        extension_atr <= MAX_ALLOWED_EXTENSION_ATR
    )

    # ------------------------------------------------------------------
    # 4H trend
    # ------------------------------------------------------------------

    trend_data = calculate_4h_indicators(
        df_4h
    )

    return {
        "price": price,

        "rsi": rsi,
        "previous_rsi": previous_rsi,
        "rsi_recovering": rsi_recovering,
        "rsi_bullish": rsi_bullish,

        "ema9": ema9_now,
        "ema21": ema21_now,
        "short_term_bullish": short_term_bullish,

        "macd_available": macd_available,
        "macd_diff": macd_diff,
        "macd_previous": macd_previous,
        "macd_improving": macd_improving,
        "macd_positive": macd_positive,

        "rvol": rvol,
        "volume_spike": volume_spike,

        "atr": atr,
        "atr_pct": atr_pct,

        "breakout": breakout,
        "healthy_pullback": healthy_pullback,

        "distance_from_ema21_atr": distance_from_ema21_atr,
        "extension_atr": extension_atr,
        "not_overextended": not_overextended,

        **trend_data,
    }


# ============================================================================
# DATA QUALITY
# ============================================================================

def passes_data_quality(
    df_1m: pd.DataFrame,
    indicators: Dict[str, Any],
) -> Tuple[bool, str]:

    if df_1m is None or len(df_1m) < 60:
        return False, "insufficient 1m history"

    required_columns = {
        "Open",
        "High",
        "Low",
        "Close",
        "Volume",
    }

    if not required_columns.issubset(df_1m.columns):
        return False, "missing OHLCV columns"

    recent = df_1m[
        list(required_columns)
    ].tail(60)

    if not np.isfinite(
        recent.to_numpy(dtype=float)
    ).all():
        return False, "non-finite market data"

    if indicators["price"] <= 0:
        return False, "invalid price"

    if indicators["atr"] <= 0:
        return False, "invalid ATR"

    if indicators["rvol"] < MIN_RVOL:
        return False, "insufficient RVOL"

    if indicators["atr_pct"] < 0.35:
        return False, "volatility too low"

    if indicators["trend_4h"] == "UNKNOWN":
        return False, "4H trend unavailable"

    return True, "data/volatility valid"


# ============================================================================
# SIGNAL
# ============================================================================

@dataclass
class SignalResult:
    valid: bool
    score: int
    reasons: List[str]
    rejection: Optional[str] = None


def evaluate_signal(
    indicators: Dict[str, Any],
) -> SignalResult:

    reasons: List[str] = []
    score = 0

    # ------------------------------------------------------------------
    # HARD TREND GATE
    # ------------------------------------------------------------------

    if indicators["trend_4h"] != "BULLISH":

        return SignalResult(
            valid=False,
            score=0,
            reasons=[],
            rejection=(
                f"4H trend is "
                f"{indicators['trend_4h']}"
            ),
        )

    reasons.append("4H trend bullish")

    if not indicators["ema20_slope_positive"]:

        return SignalResult(
            valid=False,
            score=0,
            reasons=reasons,
            rejection="4H EMA20 slope not positive",
        )

    reasons.append("4H trend slope positive")

    # ------------------------------------------------------------------
    # EXTENSION GATE
    # ------------------------------------------------------------------

    if not indicators["not_overextended"]:

        return SignalResult(
            valid=False,
            score=0,
            reasons=reasons,
            rejection=(
                f"price extended "
                f"{indicators['extension_atr']:.2f} ATR "
                f"above EMA21"
            ),
        )

    # ------------------------------------------------------------------
    # RSI
    # ------------------------------------------------------------------

    if indicators["rsi_bullish"]:

        score += 2
        reasons.append("RSI >= 50")

    elif indicators["rsi_recovering"]:

        score += 1
        reasons.append("RSI recovering")

    # ------------------------------------------------------------------
    # MACD
    # ------------------------------------------------------------------

    if indicators["macd_positive"]:

        score += 1
        reasons.append("MACD positive")

    if indicators["macd_improving"]:

        score += 1
        reasons.append("MACD improving")

    # ------------------------------------------------------------------
    # SHORT TERM STRUCTURE
    # ------------------------------------------------------------------

    if indicators["short_term_bullish"]:

        score += 2
        reasons.append("1m EMA structure bullish")

    # ------------------------------------------------------------------
    # PULLBACK
    # ------------------------------------------------------------------

    if indicators["healthy_pullback"]:

        score += 2
        reasons.append("healthy pullback")

    # ------------------------------------------------------------------
    # BREAKOUT
    # ------------------------------------------------------------------

    if indicators["breakout"]:

        score += 1
        reasons.append("micro-breakout")

    # ------------------------------------------------------------------
    # VOLUME
    # ------------------------------------------------------------------

    if indicators["volume_spike"]:

        score += 1
        reasons.append("volume expansion")

    # ------------------------------------------------------------------
    # FINAL QUALITY GATE
    # ------------------------------------------------------------------

    if score < MIN_SIGNAL_SCORE:

        return SignalResult(
            valid=False,
            score=score,
            reasons=reasons,
            rejection=(
                f"signal score {score}/"
                f"{MIN_SIGNAL_SCORE}"
            ),
        )

    return SignalResult(
        valid=True,
        score=score,
        reasons=reasons,
        rejection=None,
    )


# ============================================================================
# POSITION SIZE
# ============================================================================

def calculate_position_size(
    equity: float,
    buying_power: float,
    price: float,
    atr: float,
) -> int:
    """
    Position sizing based on:
        1. fixed account risk
        2. maximum portfolio allocation
        3. buying power

    Stock orders are whole shares.
    """

    if (
        equity <= 0
        or buying_power <= 0
        or price <= 0
        or atr <= 0
    ):
        return 0

    risk_dollars = (
        equity * RISK_PER_TRADE_PCT
    )

    stop_distance = (
        ATR_MULTIPLIER_STOP * atr
    )

    if stop_distance <= 0:
        return 0

    qty_by_risk = (
        risk_dollars / stop_distance
    )

    qty_by_cap = (
        equity * MAX_POSITION_PCT
    ) / price

    qty_by_buying_power = (
        buying_power * 0.95
    ) / price

    qty = min(
        qty_by_risk,
        qty_by_cap,
        qty_by_buying_power,
    )

    return max(0, int(qty))


# ============================================================================
# ENTRY / EXIT PRICES
# ============================================================================

def calculate_entry_price(
    latest_price: float,
) -> float:

    if latest_price <= 0:
        return 0.0

    raw = latest_price * (
        1 + ENTRY_LIMIT_BUFFER_PCT
    )

    return round(raw, 2)


def calculate_exit_prices(
    entry: float,
    atr: float,
) -> Tuple[float, float]:

    if entry <= 0 or atr <= 0:
        return 0.0, 0.0

    stop = (
        entry
        - ATR_MULTIPLIER_STOP * atr
    )

    target = (
        entry
        + ATR_MULTIPLIER_TARGET * atr
    )

    return (
        round(stop, 2),
        round(target, 2),
    )


# ============================================================================
# COMPATIBILITY HELPERS
# ============================================================================

def check_signal(
    row: pd.Series,
) -> bool:
    """
    Compatibility helper for older backtests.

    New code should use:
        analyze_indicators()
        evaluate_signal()

    This function supports rows that already contain the
    older indicator columns.
    """

    required = {
        "RSI",
        "VOL_SPIKE",
        "TREND_4H",
    }

    if not required.issubset(row.index):
        return False

    try:
        return bool(
            row["RSI"] < 30
            and bool(row["VOL_SPIKE"])
            and row["TREND_4H"] == "BULLISH"
        )

    except Exception:
        return False


def calculate_indicators(
    df: pd.DataFrame,
    symbol: str = "",
) -> pd.DataFrame:
    """
    Compatibility implementation for the original v13 backtester.

    This preserves the old column interface while the new v12
    engine uses analyze_indicators/evaluate_signal directly.

    IMPORTANT:
    New v12 backtesting should use the new functions above.
    """

    df = df.copy()

    if isinstance(df.columns, pd.MultiIndex) and symbol:

        try:
            df = df.xs(
                symbol,
                level=1,
                axis=1,
            )

        except Exception:
            try:
                df = df.xs(
                    symbol,
                    level=0,
                    axis=1,
                )
            except Exception:
                pass

    df = normalize_ohlcv(df)

    if df is None:
        raise ValueError(
            "Unable to normalize OHLCV data"
        )

    df["RSI"] = ta.momentum.rsi(
        df["Close"],
        window=RSI_WINDOW,
    )

    df["VOL_SMA20"] = (
        df["Volume"]
        .shift(1)
        .rolling(20)
        .mean()
    )

    df["VOL_SPIKE"] = (
        df["Volume"]
        > 1.2 * df["VOL_SMA20"]
    )

    # Approximate 4H trend from the supplied intraday data.
    df_4h = (
        df["Close"]
        .resample("4h")
        .last()
        .dropna()
        .to_frame("Close")
    )

    if len(df_4h) > 1:
        df_4h = df_4h.iloc[:-1]

    df_4h["EMA20"] = ta.trend.ema_indicator(
        df_4h["Close"],
        window=EMA_FAST_4H,
    )

    df_4h["EMA50"] = ta.trend.ema_indicator(
        df_4h["Close"],
        window=EMA_SLOW_4H,
    )

    df_4h["TREND_4H"] = np.where(
        df_4h["EMA20"] > df_4h["EMA50"],
        "BULLISH",
        "BEARISH",
    )

    df["TREND_4H"] = (
        df_4h["TREND_4H"]
        .reindex(
            df.index,
            method="ffill",
        )
    )

    return df
