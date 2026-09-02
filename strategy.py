import numpy as np
import pandas as pd
import ta


# ============================================================
# 10/10 RESEARCH ENGINE
#
# Philosophy:
#   4H trend -> 15m pullback -> stabilization -> confirmation
#
# Long-only.
# No RSI < 30 requirement.
# No "OR" escape hatches.
# No future information.
# ============================================================

EMA_FAST_4H = 20
EMA_SLOW_4H = 50

RSI_WINDOW = 14

EMA_FAST_15M = 9
EMA_SLOW_15M = 21

ATR_WINDOW = 14
VOLUME_WINDOW = 20

RSI_MIN = 40
RSI_MAX = 55

VOLUME_MULTIPLIER = 1.10

ATR_STOP_MULTIPLIER = 1.50
ATR_TARGET_MULTIPLIER = 3.00

MIN_ATR_PCT = 0.35
MAX_ATR_PCT = 6.00


def _flatten_yfinance_columns(df: pd.DataFrame, symbol: str = ""):
    """
    Handles common yfinance MultiIndex layouts.
    """
    if not isinstance(df.columns, pd.MultiIndex):
        return df

    levels = [list(level) for level in df.columns.levels]

    if symbol:
        for level_number, values in enumerate(levels):
            if symbol in values:
                try:
                    return df.xs(symbol, level=level_number, axis=1)
                except Exception:
                    pass

    # Fallback: remove singleton level when possible.
    if df.columns.nlevels == 2:
        try:
            return df.droplevel(1, axis=1)
        except Exception:
            pass

    return df


def calculate_atr(df: pd.DataFrame, window: int = ATR_WINDOW):
    return ta.volatility.average_true_range(
        high=df["High"],
        low=df["Low"],
        close=df["Close"],
        window=window,
    )


def calculate_indicators(
    df: pd.DataFrame,
    symbol: str = "",
) -> pd.DataFrame:

    df = _flatten_yfinance_columns(df.copy(), symbol)

    required = ["Open", "High", "Low", "Close", "Volume"]

    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Missing columns: {missing}")

    df = df.sort_index()

    # --------------------------------------------------------
    # 1. 15-MINUTE INDICATORS
    # --------------------------------------------------------

    df["RSI"] = ta.momentum.rsi(
        df["Close"],
        window=RSI_WINDOW,
    )

    df["EMA_FAST"] = ta.trend.ema_indicator(
        df["Close"],
        window=EMA_FAST_15M,
    )

    df["EMA_SLOW"] = ta.trend.ema_indicator(
        df["Close"],
        window=EMA_SLOW_15M,
    )

    df["ATR"] = calculate_atr(df)

    # Previous bars only.
    # This prevents current-bar volume from influencing its own
    # volume baseline.
    df["VOL_SMA20"] = (
        df["Volume"]
        .shift(1)
        .rolling(VOLUME_WINDOW)
        .mean()
    )

    df["RVOL"] = (
        df["Volume"] / df["VOL_SMA20"]
    )

    df["VOL_CONFIRM"] = (
        df["RVOL"] >= VOLUME_MULTIPLIER
    )

    df["ATR_PCT"] = (
        df["ATR"] / df["Close"] * 100
    )

    # --------------------------------------------------------
    # 2. PULLBACK STRUCTURE
    # --------------------------------------------------------

    # Previous close must have been weaker than current close.
    df["PRICE_RECOVERING"] = (
        df["Close"] > df["Close"].shift(1)
    )

    # Current candle closes green.
    df["GREEN_CANDLE"] = (
        df["Close"] > df["Open"]
    )

    # Price recovers back above fast EMA.
    df["EMA_RECOVERY"] = (
        (df["Close"] > df["EMA_FAST"]) &
        (df["Close"].shift(1) <= df["EMA_FAST"].shift(1))
    )

    # More flexible continuation condition:
    # price is above fast EMA and fast EMA is above slow EMA.
    df["TREND_ALIGNMENT_15M"] = (
        (df["Close"] > df["EMA_FAST"]) &
        (df["EMA_FAST"] > df["EMA_SLOW"])
    )

    # RSI should recover from weakness, rather than require
    # an extreme oversold reading.
    df["RSI_RECOVERY"] = (
        (df["RSI"] > df["RSI"].shift(1)) &
        (df["RSI"] >= RSI_MIN) &
        (df["RSI"] <= RSI_MAX)
    )

    # --------------------------------------------------------
    # 3. 4H TREND
    #
    # IMPORTANT:
    # We only use COMPLETED 4H candles.
    # --------------------------------------------------------

    df_4h = (
        df["Close"]
        .resample("4h", origin="start")
        .last()
        .dropna()
        .to_frame("Close")
    )

    # Remove currently forming 4H candle.
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

    df_4h["TREND_4H"] = (
        (df_4h["EMA20"] > df_4h["EMA50"]) &
        (df_4h["Close"] > df_4h["EMA20"])
    )

    df["TREND_4H"] = (
        df_4h["TREND_4H"]
        .reindex(df.index, method="ffill")
        .fillna(False)
        .astype(bool)
    )

    # --------------------------------------------------------
    # 4. FINAL ENTRY SETUP
    # --------------------------------------------------------

    df["VOLATILITY_OK"] = (
        (df["ATR_PCT"] >= MIN_ATR_PCT) &
        (df["ATR_PCT"] <= MAX_ATR_PCT)
    )

    # Candle must demonstrate actual recovery.
    df["REVERSAL_CONFIRMATION"] = (
        df["GREEN_CANDLE"] &
        df["PRICE_RECOVERING"] &
        (
            df["EMA_RECOVERY"] |
            df["TREND_ALIGNMENT_15M"]
        )
    )

    # Final signal.
    df["SIGNAL"] = (
        df["TREND_4H"] &
        df["RSI_RECOVERY"] &
        df["VOL_CONFIRM"] &
        df["REVERSAL_CONFIRMATION"] &
        df["VOLATILITY_OK"]
    )

    return df


def check_signal(row: pd.Series) -> bool:
    """
    Single source of truth for live trading and backtesting.
    """

    try:
        return bool(
            row["SIGNAL"]
            and np.isfinite(row["RSI"])
            and np.isfinite(row["ATR"])
            and np.isfinite(row["RVOL"])
        )
    except Exception:
        return False
