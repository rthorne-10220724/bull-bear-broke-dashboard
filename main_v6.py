"""
BULL. BEAR AND BROKE - TRADING ENGINE v12
==========================================

Hardened momentum / pullback trading engine.

IMPORTANT:
-----------
This is NOT guaranteed-profitable code.
"10/10" refers to the engineering/strategy structure, not expected returns.

Run PAPER_TRADING=true first and validate with:
- out-of-sample backtesting
- walk-forward testing
- realistic commissions/slippage
- trade-count statistics
- expectancy
- profit factor
- maximum drawdown
- Sharpe/Sortino
- regime analysis

Core philosophy:
---------------
HARD GATES:
    - valid market/data state
    - 4H asset trend bullish
    - adequate volatility
    - adequate liquidity
    - portfolio risk available

SETUP SCORE:
    - RSI
    - MACD
    - short-term EMA structure
    - pullback quality
    - volume
    - breakout/reclaim

This intentionally avoids requiring every indicator to fire simultaneously.
"""

from __future__ import annotations

import os
import sys
import time
import sqlite3
import logging
import datetime as dt
import zoneinfo

from dataclasses import dataclass
from logging.handlers import RotatingFileHandler
from typing import Optional, Dict, Any, List, Tuple, Set

import numpy as np
import pandas as pd
import ta
import yfinance as yf

from alpaca.trading.client import TradingClient
from alpaca.trading.requests import (
    LimitOrderRequest,
    TakeProfitRequest,
    StopLossRequest,
    GetOrdersRequest,
)
from alpaca.trading.enums import (
    OrderSide,
    TimeInForce,
    OrderClass,
    QueryOrderStatus,
)

from alpaca.data.historical import (
    StockHistoricalDataClient,
    CryptoHistoricalDataClient,
)

from alpaca.data.requests import (
    StockBarsRequest,
    CryptoBarsRequest,
)

from alpaca.data.timeframe import TimeFrame
from alpaca.data.enums import DataFeed


# ============================================================================
# 1. LOGGING
# ============================================================================

LOG_DIR = "logs"
os.makedirs(LOG_DIR, exist_ok=True)

logger = logging.getLogger("bbb_engine")
logger.setLevel(logging.DEBUG)
logger.propagate = False

formatter = logging.Formatter(
    "%(asctime)s | %(levelname)-7s | %(filename)s:%(lineno)d | %(message)s"
)

if not logger.handlers:
    console = logging.StreamHandler(sys.stdout)
    console.setLevel(logging.INFO)
    console.setFormatter(formatter)

    file_handler = RotatingFileHandler(
        os.path.join(LOG_DIR, "trading_engine.log"),
        maxBytes=10 * 1024 * 1024,
        backupCount=5,
        encoding="utf-8",
    )
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(formatter)

    logger.addHandler(console)
    logger.addHandler(file_handler)


decision_logger = logging.getLogger("decision_engine")
decision_logger.setLevel(logging.INFO)
decision_logger.propagate = False

if not decision_logger.handlers:
    decision_handler = RotatingFileHandler(
        os.path.join(LOG_DIR, "decisions.log"),
        maxBytes=10 * 1024 * 1024,
        backupCount=5,
        encoding="utf-8",
    )
    decision_handler.setFormatter(
        logging.Formatter("%(asctime)s | %(message)s")
    )
    decision_logger.addHandler(decision_handler)


def log_decision(message: str) -> None:
    decision_logger.info(message)
    logger.info("DECISION: %s", message)


# ============================================================================
# 2. ENVIRONMENT
# ============================================================================

ALPACA_API_KEY = (
    os.getenv("APCA_API_KEY_ID")
    or os.getenv("ALPACA_API_KEY")
)

ALPACA_SECRET_KEY = (
    os.getenv("APCA_API_SECRET_KEY")
    or os.getenv("ALPACA_SECRET_KEY")
)

PAPER_TRADING = (
    os.getenv("PAPER_TRADING", "true").strip().lower() == "true"
)

if not ALPACA_API_KEY or not ALPACA_SECRET_KEY:
    logger.error("Missing Alpaca API credentials.")
    sys.exit(1)

if not PAPER_TRADING:
    logger.warning(
        "LIVE TRADING IS ENABLED. This engine can submit real orders."
    )


trading_client = TradingClient(
    api_key=ALPACA_API_KEY,
    secret_key=ALPACA_SECRET_KEY,
    paper=PAPER_TRADING,
)

stock_data_client = StockHistoricalDataClient(
    api_key=ALPACA_API_KEY,
    secret_key=ALPACA_SECRET_KEY,
)

crypto_data_client = CryptoHistoricalDataClient(
    api_key=ALPACA_API_KEY,
    secret_key=ALPACA_SECRET_KEY,
)


# ============================================================================
# 3. CONFIGURATION
# ============================================================================

DB_FILE = os.path.join(LOG_DIR, "trading_state.db")

STOCK_CANDIDATES = [
    "PLTR",
    "SOFI",
    "NIO",
    "MARA",
    "RIOT",
    "HOOD",
    "RIVN",
    "SNAP",
    "AMC",
    "GME",
    "PTON",
    "AAL",
    "SPY",
    "QQQ",
    "AMD",
    "COIN",
    "DKNG",
    "UBER",
]

CRYPTO_TARGETS = [
    "BTC/USD",
    "ETH/USD",
    "SOL/USD",
    "AVAX/USD",
    "DOGE/USD",
]

# Portfolio
MAX_OPEN_POSITIONS = 3
MAX_PENDING_ENTRIES = 2
MAX_TOTAL_ACTIVE_TRADES = 3

# Risk
RISK_PER_TRADE_PCT = 0.0075
MAX_POSITION_PCT = 0.15
MAX_DAILY_DRAWDOWN_PCT = 0.04

# Trade frequency
ENTRY_COOLDOWN_SECONDS = 180
MAX_NEW_ENTRIES_PER_DAY = 5

# Indicators
RSI_WINDOW = 14

EMA_FAST_1M = 9
EMA_SLOW_1M = 21

EMA_FAST_4H = 20
EMA_SLOW_4H = 50

MACD_FAST = 12
MACD_SLOW = 26
MACD_SIGNAL = 9

ATR_WINDOW = 14

# Entry quality
MIN_SIGNAL_SCORE = 6
MAX_ALLOWED_EXTENSION_ATR = 1.25
MIN_RVOL = 0.70

# ATR exits
ATR_MULTIPLIER_STOP = 1.50
ATR_MULTIPLIER_TARGET = 3.75

# Entry price
ENTRY_LIMIT_BUFFER_PCT = 0.0005

# Cache
CACHE_TTL_4H_SECONDS = 900

# Cycle
CYCLE_SECONDS = 180

# Trading session
MARKET_CLOSE_BUFFER_SECONDS = 10 * 60

# Crypto is deliberately disabled for automatic bracket execution
# until separately validated against the broker's current crypto
# order-class capabilities.
ENABLE_CRYPTO_TRADING = False


FOUR_HOUR_CACHE: Dict[
    str,
    Tuple[pd.DataFrame, dt.datetime]
] = {}


# ============================================================================
# 4. DATABASE
# ============================================================================

def db_connection():
    return sqlite3.connect(DB_FILE, timeout=10)


def init_db() -> None:
    with db_connection() as conn:
        cursor = conn.cursor()

        cursor.execute(
            """
            CREATE TABLE IF NOT EXISTS state (
                key TEXT PRIMARY KEY,
                value TEXT NOT NULL
            )
            """
        )

        cursor.execute(
            """
            CREATE TABLE IF NOT EXISTS trade_journal (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp TEXT NOT NULL,
                symbol TEXT NOT NULL,
                side TEXT NOT NULL,
                qty REAL NOT NULL,
                entry_price REAL NOT NULL,
                stop_loss REAL NOT NULL,
                take_profit REAL NOT NULL,
                status TEXT NOT NULL,
                score INTEGER,
                reason TEXT
            )
            """
        )

        cursor.execute(
            """
            CREATE INDEX IF NOT EXISTS idx_trade_timestamp
            ON trade_journal(timestamp)
            """
        )


def set_state(key: str, value: str) -> None:
    with db_connection() as conn:
        conn.execute(
            """
            INSERT OR REPLACE INTO state(key, value)
            VALUES (?, ?)
            """,
            (key, value),
        )


def get_state(key: str) -> Optional[str]:
    with db_connection() as conn:
        row = conn.execute(
            "SELECT value FROM state WHERE key = ?",
            (key,),
        ).fetchone()

    return row[0] if row else None


def record_trade(
    symbol: str,
    side: str,
    qty: float,
    entry_price: float,
    stop_loss: float,
    take_profit: float,
    status: str,
    score: Optional[int] = None,
    reason: str = "",
) -> None:

    with db_connection() as conn:
        conn.execute(
            """
            INSERT INTO trade_journal (
                timestamp,
                symbol,
                side,
                qty,
                entry_price,
                stop_loss,
                take_profit,
                status,
                score,
                reason
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                dt.datetime.now(dt.timezone.utc).isoformat(),
                symbol,
                side,
                qty,
                entry_price,
                stop_loss,
                take_profit,
                status,
                score,
                reason,
            ),
        )


# ============================================================================
# 5. DAILY STATE
# ============================================================================

def utc_date_string() -> str:
    return dt.datetime.now(dt.timezone.utc).strftime("%Y-%m-%d")


def daily_entry_key() -> str:
    return f"entries:{utc_date_string()}"


def get_daily_entry_count() -> int:
    value = get_state(daily_entry_key())

    try:
        return int(value) if value else 0
    except ValueError:
        return 0


def increment_daily_entry_count() -> None:
    count = get_daily_entry_count()
    set_state(daily_entry_key(), str(count + 1))


def cooldown_key(symbol: str) -> str:
    return f"cooldown:{symbol.replace('/', '_')}"


def is_on_cooldown(symbol: str) -> bool:
    value = get_state(cooldown_key(symbol))

    if not value:
        return False

    try:
        return (
            time.time() - float(value)
            < ENTRY_COOLDOWN_SECONDS
        )
    except (TypeError, ValueError):
        return False


def set_cooldown(symbol: str) -> None:
    set_state(
        cooldown_key(symbol),
        str(time.time()),
    )


# ============================================================================
# 6. DATA HELPERS
# ============================================================================

def normalize_ohlcv(df: pd.DataFrame) -> Optional[pd.DataFrame]:
    if df is None or df.empty:
        return None

    df = df.copy()

    rename_map = {
        "open": "Open",
        "high": "High",
        "low": "Low",
        "close": "Close",
        "volume": "Volume",
    }

    df.rename(columns=rename_map, inplace=True)

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

    df.dropna(inplace=True)

    if len(df) < 30:
        return None

    return df


def fetch_1m_bars(
    ticker: str,
    limit: int = 180,
    retries: int = 3,
) -> Optional[pd.DataFrame]:

    end = dt.datetime.now(dt.timezone.utc)
    start = end - dt.timedelta(
        minutes=limit + 180
    )

    is_crypto = "/" in ticker

    for attempt in range(retries):
        try:
            if is_crypto:
                request = CryptoBarsRequest(
                    symbol_or_symbols=ticker,
                    timeframe=TimeFrame.Minute,
                    start=start,
                    end=end,
                )

                response = crypto_data_client.get_crypto_bars(
                    request
                )

            else:
                request = StockBarsRequest(
                    symbol_or_symbols=ticker,
                    timeframe=TimeFrame.Minute,
                    start=start,
                    end=end,
                    feed=DataFeed.IEX,
                )

                response = stock_data_client.get_stock_bars(
                    request
                )

            df = response.df

            if isinstance(df.index, pd.MultiIndex):
                try:
                    df = df.xs(
                        ticker,
                        level=0,
                    )
                except Exception:
                    return None

            df = normalize_ohlcv(df)

            if df is None:
                raise ValueError("Invalid/empty OHLCV data")

            return df.tail(limit)

        except Exception as exc:
            logger.warning(
                "[%s] 1m data attempt %s/%s failed: %s",
                ticker,
                attempt + 1,
                retries,
                exc,
            )

            if attempt < retries - 1:
                time.sleep(1.5)

    return None


def fetch_4h_bars_cached(
    ticker: str,
) -> Optional[pd.DataFrame]:

    now = dt.datetime.now(dt.timezone.utc)

    cached = FOUR_HOUR_CACHE.get(ticker)

    if cached:
        cached_df, cached_at = cached

        if (
            now - cached_at
        ).total_seconds() < CACHE_TTL_4H_SECONDS:
            return cached_df

    try:
        yf_ticker = ticker.replace("/", "-")

        obj = yf.Ticker(yf_ticker)

        hourly = obj.history(
            period="60d",
            interval="60m",
            auto_adjust=False,
        )

        if hourly.empty:
            return None

        hourly = normalize_ohlcv(hourly)

        if hourly is None:
            return None

        four_hour = (
            hourly
            .resample("4h")
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

        # Never use the currently forming 4H candle.
        if len(four_hour) > 1:
            four_hour = four_hour.iloc[:-1]

        if len(four_hour) < EMA_SLOW_4H:
            logger.warning(
                "[%s] insufficient 4H history: %s bars",
                ticker,
                len(four_hour),
            )
            return None

        FOUR_HOUR_CACHE[ticker] = (
            four_hour,
            now,
        )

        return four_hour

    except Exception as exc:
        logger.warning(
            "[%s] 4H data error: %s",
            ticker,
            exc,
        )

        if ticker in FOUR_HOUR_CACHE:
            return FOUR_HOUR_CACHE[ticker][0]

        return None


# ============================================================================
# 7. INDICATORS
# ============================================================================

def calculate_atr(
    df: pd.DataFrame,
    window: int = ATR_WINDOW,
) -> float:

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


def calculate_rvol(
    volume: pd.Series,
    window: int = 20,
) -> float:

    if len(volume) < window + 1:
        return 0.0

    baseline = volume.iloc[-window - 1:-1].mean()

    if baseline <= 0:
        return 0.0

    return float(volume.iloc[-1] / baseline)


def analyze_indicators(
    df_1m: pd.DataFrame,
    df_4h: pd.DataFrame,
) -> Dict[str, Any]:

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
    # 1m EMA structure
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
    # MACD on 15-minute closes
    # ------------------------------------------------------------------

    close_15m = close.resample("15min").last().dropna()

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

        if len(macd.dropna()) >= 2:
            macd_diff = float(macd.iloc[-1])
            macd_previous = float(macd.iloc[-2])
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
    # Volume
    # ------------------------------------------------------------------

    rvol = calculate_rvol(volume)

    volume_spike = (
        rvol >= 1.20
    )

    # ------------------------------------------------------------------
    # ATR
    # ------------------------------------------------------------------

    atr = calculate_atr(df_1m)

    atr_pct = (
        atr / price * 100
        if price > 0
        else 0
    )

    # ------------------------------------------------------------------
    # Recent structure
    # ------------------------------------------------------------------

    previous_high = float(
        high.iloc[-6:-1].max()
    )

    breakout = price >= previous_high

    # Pullback quality:
    #
    # We want price near the short EMA structure,
    # but not collapsing far below it.
    #
    distance_from_ema21_atr = (
        abs(price - ema21_now) / atr
        if atr > 0
        else 999
    )

    healthy_pullback = (
        price >= ema21_now - (0.75 * atr)
        and distance_from_ema21_atr <= 1.50
    )

    # Avoid buying extreme extension.
    extension_atr = (
        (price - ema21_now) / atr
        if atr > 0
        else 999
    )

    not_overextended = (
        extension_atr <= MAX_ALLOWED_EXTENSION_ATR
    )

    # ------------------------------------------------------------------
    # 4H trend
    # ------------------------------------------------------------------

    close_4h = df_4h["Close"]

    ema20_4h = ta.trend.ema_indicator(
        close_4h,
        window=EMA_FAST_4H,
    )

    ema50_4h = ta.trend.ema_indicator(
        close_4h,
        window=EMA_SLOW_4H,
    )

    ema20_value = float(ema20_4h.iloc[-1])
    ema50_value = float(ema50_4h.iloc[-1])

    trend_4h = (
        "BULLISH"
        if ema20_value > ema50_value
        else "BEARISH"
    )

    # Require actual slope confirmation too.
    if len(ema20_4h) >= 4:
        ema20_slope_positive = (
            ema20_4h.iloc[-1]
            > ema20_4h.iloc[-4]
        )
    else:
        ema20_slope_positive = False

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
        "macd_improving": macd_improving,
        "macd_positive": macd_positive,
        "rvol": rvol,
        "volume_spike": volume_spike,
        "atr": atr,
        "atr_pct": atr_pct,
        "breakout": breakout,
        "healthy_pullback": healthy_pullback,
        "extension_atr": extension_atr,
        "not_overextended": not_overextended,
        "trend_4h": trend_4h,
        "ema20_4h": ema20_value,
        "ema50_4h": ema50_value,
        "ema20_slope_positive": ema20_slope_positive,
    }


# ============================================================================
# 8. VOLATILITY / DATA QUALITY
# ============================================================================

def passes_data_quality(
    df_1m: pd.DataFrame,
    indicators: Dict[str, Any],
) -> Tuple[bool, str]:

    if len(df_1m) < 60:
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

    if not np.isfinite(
        df_1m[list(required_columns)].tail(30).to_numpy()
    ).all():
        return False, "non-finite market data"

    if indicators["price"] <= 0:
        return False, "invalid price"

    if indicators["atr"] <= 0:
        return False, "invalid ATR"

    if indicators["rvol"] < MIN_RVOL:
        return False, "insufficient RVOL"

    # 1% ATR was too arbitrary as a universal requirement.
    # Instead require enough movement to justify the stop.
    if indicators["atr_pct"] < 0.35:
        return False, "volatility too low"

    return True, "data/volatility valid"


# ============================================================================
# 9. SIGNAL ENGINE
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
    # SCORE
    # ------------------------------------------------------------------

    # RSI
    if indicators["rsi_bullish"]:
        score += 2
        reasons.append("RSI >= 50")
    elif indicators["rsi_recovering"]:
        score += 1
        reasons.append("RSI recovering")

    # MACD
    if indicators["macd_positive"]:
        score += 1
        reasons.append("MACD positive")

    if indicators["macd_improving"]:
        score += 1
        reasons.append("MACD improving")

    # Short-term structure
    if indicators["short_term_bullish"]:
        score += 2
        reasons.append("1m EMA structure bullish")

    # Pullback
    if indicators["healthy_pullback"]:
        score += 2
        reasons.append("healthy pullback")

    # Breakout/reclaim
    if indicators["breakout"]:
        score += 1
        reasons.append("micro-breakout")

    # Volume
    if indicators["volume_spike"]:
        score += 1
        reasons.append("volume expansion")

    # Minimum setup quality
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
    )


# ============================================================================
# 10. PORTFOLIO RISK
# ============================================================================

def get_account_equity() -> Optional[float]:
    try:
        account = trading_client.get_account()
        return float(account.equity)
    except Exception as exc:
        logger.error(
            "Unable to retrieve equity: %s",
            exc,
        )
        return None


def get_account_buying_power() -> Optional[float]:
    try:
        account = trading_client.get_account()
        return float(account.buying_power)
    except Exception as exc:
        logger.error(
            "Unable to retrieve buying power: %s",
            exc,
        )
        return None


def audit_portfolio_risk_state() -> bool:
    """
    Returns True if trading should STOP.
    """

    try:
        account = trading_client.get_account()

        equity = float(account.equity)
        last_equity = float(account.last_equity)

        if last_equity <= 0:
            return False

        drawdown = (
            equity - last_equity
        ) / last_equity

        if drawdown <= -MAX_DAILY_DRAWDOWN_PCT:

            today = utc_date_string()

            if get_state("circuit_breaker_date") != today:

                log_decision(
                    "CIRCUIT BREAKER: "
                    f"daily drawdown "
                    f"{drawdown * 100:.2f}%"
                )

                try:
                    trading_client.close_all_positions(
                        cancel_orders=True
                    )
                except Exception as exc:
                    logger.error(
                        "Failed liquidating portfolio: %s",
                        exc,
                    )

                set_state(
                    "circuit_breaker_date",
                    today,
                )

            return True

    except Exception as exc:
        logger.exception(
            "Risk audit failed: %s",
            exc,
        )

        # Fail closed.
        return True

    return False


def calculate_position_size(
    equity: float,
    buying_power: float,
    price: float,
    atr: float,
) -> float:

    if (
        equity <= 0
        or buying_power <= 0
        or price <= 0
        or atr <= 0
    ):
        return 0.0

    risk_dollars = (
        equity * RISK_PER_TRADE_PCT
    )

    stop_distance = (
        ATR_MULTIPLIER_STOP * atr
    )

    if stop_distance <= 0:
        return 0.0

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

    # Stocks are whole-share orders.
    return float(max(0, int(qty)))


# ============================================================================
# 11. POSITIONS / ORDERS
# ============================================================================

def get_open_positions() -> Dict[str, Any]:

    try:
        positions = trading_client.get_all_positions()

        return {
            position.symbol: position
            for position in positions
        }

    except Exception as exc:
        logger.error(
            "Failed fetching positions: %s",
            exc,
        )
        return {}


def get_open_orders() -> List[Any]:

    try:
        request = GetOrdersRequest(
            status=QueryOrderStatus.OPEN,
            nested=True,
        )

        return trading_client.get_orders(
            filter=request
        )

    except Exception as exc:
        logger.error(
            "Failed fetching orders: %s",
            exc,
        )
        return []


def get_pending_symbols(
    orders: List[Any],
) -> Set[str]:

    symbols: Set[str] = set()

    for order in orders:
        symbol = getattr(
            order,
            "symbol",
            None,
        )

        if symbol:
            symbols.add(symbol)

    return symbols


def symbol_available(
    symbol: str,
    positions: Dict[str, Any],
    pending: Set[str],
) -> bool:

    if symbol in positions:
        return False

    if symbol in pending:
        return False

    if is_on_cooldown(symbol):
        return False

    return True


# ============================================================================
# 12. MARKET SESSION
# ============================================================================

def market_is_tradeable() -> bool:

    try:
        clock = trading_client.get_clock()

        if not clock.is_open:
            return False

        now_et = dt.datetime.now(
            zoneinfo.ZoneInfo(
                "America/New_York"
            )
        )

        close = now_et.replace(
            hour=16,
            minute=0,
            second=0,
            microsecond=0,
        )

        seconds_remaining = (
            close - now_et
        ).total_seconds()

        if (
            0
            < seconds_remaining
            <= MARKET_CLOSE_BUFFER_SECONDS
        ):
            logger.info(
                "Market close approaching. "
                "No new stock entries."
            )
            return False

        return True

    except Exception as exc:
        logger.error(
            "Market clock failure: %s",
            exc,
        )

        return False


# ============================================================================
# 13. WATCHLIST
# ============================================================================

def build_watchlist(
    max_stocks: int = 12,
) -> List[str]:

    scored: List[Tuple[str, float]] = []

    for symbol in STOCK_CANDIDATES:

        try:
            df = fetch_4h_bars_cached(symbol)

            if df is None:
                continue

            close = df["Close"]

            ema20 = ta.trend.ema_indicator(
                close,
                window=EMA_FAST_4H,
            )

            ema50 = ta.trend.ema_indicator(
                close,
                window=EMA_SLOW_4H,
            )

            if (
                pd.isna(ema20.iloc[-1])
                or pd.isna(ema50.iloc[-1])
            ):
                continue

            # Hard trend filter.
            if ema20.iloc[-1] <= ema50.iloc[-1]:
                continue

            atr = calculate_atr(df)

            if atr <= 0:
                continue

            price = float(close.iloc[-1])

            # How far price is from EMA20 in ATR units.
            #
            # We prefer stocks that remain strong but aren't
            # extremely extended.
            extension = (
                price - ema20.iloc[-1]
            ) / atr

            if extension > 2.5:
                continue

            # Favor modest pullbacks / proximity to EMA20.
            distance_score = -abs(extension)

            slope_bonus = (
                1.0
                if ema20.iloc[-1] > ema20.iloc[-4]
                else 0.0
            )

            score = (
                distance_score
                + slope_bonus
            )

            scored.append(
                (symbol, score)
            )

        except Exception as exc:
            logger.debug(
                "[SCREENER] %s failed: %s",
                symbol,
                exc,
            )

    scored.sort(
        key=lambda x: x[1],
        reverse=True,
    )

    stocks = [
        symbol
        for symbol, _ in scored[:max_stocks]
    ]

    if ENABLE_CRYPTO_TRADING:
        stocks.extend(CRYPTO_TARGETS)

    logger.info(
        "[SCREENER] Watchlist: %s",
        stocks,
    )

    return stocks


# ============================================================================
# 14. ORDER PRICING
# ============================================================================

def calculate_entry_price(
    latest_price: float,
) -> float:

    # Small buffer to improve fill probability without
    # blindly using a market order.
    raw = latest_price * (
        1 + ENTRY_LIMIT_BUFFER_PCT
    )

    return round(raw, 2)


def calculate_exit_prices(
    entry: float,
    atr: float,
) -> Tuple[float, float]:

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
# 15. STOCK BRACKET ORDER
# ============================================================================

def place_stock_bracket(
    symbol: str,
    qty: float,
    price: float,
    atr: float,
    score: int,
    reasons: List[str],
) -> Optional[Any]:

    if qty <= 0:
        return None

    entry = calculate_entry_price(price)

    stop, target = calculate_exit_prices(
        entry,
        atr,
    )

    if stop <= 0 or target <= entry:
        return None

    try:

        request = LimitOrderRequest(
            symbol=symbol,
            qty=qty,
            side=OrderSide.BUY,
            limit_price=entry,
            time_in_force=TimeInForce.GTC,
            order_class=OrderClass.BRACKET,
            stop_loss=StopLossRequest(
                stop_price=stop
            ),
            take_profit=TakeProfitRequest(
                limit_price=target
            ),
        )

        response = trading_client.submit_order(
            request
        )

        set_cooldown(symbol)
        increment_daily_entry_count()

        record_trade(
            symbol=symbol,
            side="BUY",
            qty=qty,
            entry_price=entry,
            stop_loss=stop,
            take_profit=target,
            status="SUBMITTED",
            score=score,
            reason=" | ".join(reasons),
        )

        log_decision(
            f"BUY {symbol} "
            f"qty={qty} "
            f"entry={entry:.2f} "
            f"stop={stop:.2f} "
            f"target={target:.2f} "
            f"score={score} "
            f"| {' | '.join(reasons)}"
        )

        return response

    except Exception as exc:

        logger.error(
            "[%s] order submission failed: %s",
            symbol,
            exc,
        )

        return None


# ============================================================================
# 16. STALE ORDER MANAGEMENT
# ============================================================================

def cancel_stale_orders(
    orders: List[Any],
    max_age_minutes: int = 30,
) -> None:

    now = dt.datetime.now(dt.timezone.utc)

    for order in orders:

        try:
            status = str(
                getattr(order, "status", "")
            ).lower()

            if "open" not in status:
                continue

            created = getattr(
                order,
                "created_at",
                None,
            )

            if created is None:
                continue

            if created.tzinfo is None:
                created = created.replace(
                    tzinfo=dt.timezone.utc
                )

            age = (
                now - created
            ).total_seconds() / 60

            if age >= max_age_minutes:

                order_id = getattr(
                    order,
                    "id",
                    None,
                )

                if order_id:

                    logger.info(
                        "Cancelling stale order "
                        "%s (%s minutes old)",
                        order_id,
                        age,
                    )

                    trading_client.cancel_order_by_id(
                        order_id
                    )

        except Exception as exc:
            logger.warning(
                "Failed processing stale order: %s",
                exc,
            )


# ============================================================================
# 17. MAIN SCAN
# ============================================================================

def run_cycle() -> None:

    logger.info(
        "========== TRADING CYCLE =========="
    )

    # ------------------------------------------------------------------
    # Risk
    # ------------------------------------------------------------------

    if audit_portfolio_risk_state():
        logger.warning(
            "Risk circuit breaker active."
        )
        return

    # ------------------------------------------------------------------
    # Session
    # ------------------------------------------------------------------

    if not market_is_tradeable():
        return

    # ------------------------------------------------------------------
    # Daily limit
    # ------------------------------------------------------------------

    entries_today = get_daily_entry_count()

    if entries_today >= MAX_NEW_ENTRIES_PER_DAY:
        logger.info(
            "Daily entry limit reached: %s",
            entries_today,
        )
        return

    # ------------------------------------------------------------------
    # Account
    # ------------------------------------------------------------------

    equity = get_account_equity()
    buying_power = get_account_buying_power()

    if (
        equity is None
        or buying_power is None
        or equity <= 0
        or buying_power <= 0
    ):
        return

    # ------------------------------------------------------------------
    # Portfolio
    # ------------------------------------------------------------------

    positions = get_open_positions()
    orders = get_open_orders()

    pending = get_pending_symbols(orders)

    cancel_stale_orders(orders)

    active_count = (
        len(positions)
        + len(pending)
    )

    logger.info(
        "Portfolio | equity=$%.2f "
        "buying_power=$%.2f "
        "positions=%s "
        "pending=%s",
        equity,
        buying_power,
        len(positions),
        len(pending),
    )

    if active_count >= MAX_TOTAL_ACTIVE_TRADES:
        logger.info(
            "Maximum active trade capacity reached."
        )
        return

    # ------------------------------------------------------------------
    # Watchlist
    # ------------------------------------------------------------------

    watchlist = build_watchlist(
        max_stocks=12
    )

    for symbol in watchlist:

        try:

            # Re-fetch state because an earlier symbol may
            # have created a new position/order.
            positions = get_open_positions()
            orders = get_open_orders()
            pending = get_pending_symbols(
                orders
            )

            active_count = (
                len(positions)
                + len(pending)
            )

            if (
                active_count
                >= MAX_TOTAL_ACTIVE_TRADES
            ):
                break

            if not symbol_available(
                symbol,
                positions,
                pending,
            ):
                continue

            if "/" in symbol:

                if not ENABLE_CRYPTO_TRADING:
                    logger.debug(
                        "[%s] Crypto execution disabled.",
                        symbol,
                    )
                    continue

            # ----------------------------------------------------------
            # Data
            # ----------------------------------------------------------

            df_1m = fetch_1m_bars(
                symbol,
                limit=180,
            )

            if df_1m is None:
                logger.info(
                    "[%s] rejected: no 1m data",
                    symbol,
                )
                continue

            df_4h = fetch_4h_bars_cached(
                symbol
            )

            if df_4h is None:
                logger.info(
                    "[%s] rejected: no 4H data",
                    symbol,
                )
                continue

            # ----------------------------------------------------------
            # Indicators
            # ----------------------------------------------------------

            indicators = analyze_indicators(
                df_1m,
                df_4h,
            )

            # ----------------------------------------------------------
            # Data quality
            # ----------------------------------------------------------

            quality_ok, quality_reason = (
                passes_data_quality(
                    df_1m,
                    indicators,
                )
            )

            if not quality_ok:

                logger.info(
                    "[%s] rejected: %s",
                    symbol,
                    quality_reason,
                )

                continue

            # ----------------------------------------------------------
            # Signal
            # ----------------------------------------------------------

            signal = evaluate_signal(
                indicators
            )

            logger.info(
                "[%s] price=%.2f "
                "RSI=%.1f "
                "MACD=%.4f "
                "RVOL=%.2f "
                "ATR%%=%.2f "
                "4H=%s "
                "score=%s "
                "valid=%s",
                symbol,
                indicators["price"],
                indicators["rsi"],
                indicators["macd_diff"],
                indicators["rvol"],
                indicators["atr_pct"],
                indicators["trend_4h"],
                signal.score,
                signal.valid,
            )

            if not signal.valid:

                logger.debug(
                    "[%s] rejected: %s",
                    symbol,
                    signal.rejection,
                )

                continue

            # ----------------------------------------------------------
            # Final sizing
            # ----------------------------------------------------------

            qty = calculate_position_size(
                equity=equity,
                buying_power=buying_power,
                price=indicators["price"],
                atr=indicators["atr"],
            )

            if qty <= 0:

                logger.info(
                    "[%s] rejected: "
                    "position size <= 0",
                    symbol,
                )

                continue

            # ----------------------------------------------------------
            # Final state check
            # ----------------------------------------------------------

            positions = get_open_positions()
            orders = get_open_orders()
            pending = get_pending_symbols(
                orders
            )

            if not symbol_available(
                symbol,
                positions,
                pending,
            ):
                continue

            # ----------------------------------------------------------
            # Execute
            # ----------------------------------------------------------

            if "/" in symbol:
                # Explicitly disabled above.
                continue

            response = place_stock_bracket(
                symbol=symbol,
                qty=qty,
                price=indicators["price"],
                atr=indicators["atr"],
                score=signal.score,
                reasons=signal.reasons,
            )

            if response is not None:
                logger.info(
                    "[%s] order accepted by broker.",
                    symbol,
                )

                # Prevent immediately submitting another
                # order in the same cycle.
                time.sleep(2)

        except Exception as exc:

            logger.exception(
                "[%s] scan error: %s",
                symbol,
                exc,
            )


# ============================================================================
# 18. MAIN
# ============================================================================

def main() -> None:

    init_db()

    logger.info(
        "=========================================="
    )

    logger.info(
        "Bull. Bear and Broke v12 ONLINE"
    )

    logger.info(
        "Paper trading: %s",
        PAPER_TRADING,
    )

    logger.info(
        "Risk/trade: %.2f%%",
        RISK_PER_TRADE_PCT * 100,
    )

    logger.info(
        "Minimum signal score: %s",
        MIN_SIGNAL_SCORE,
    )

    logger.info(
        "Crypto trading: %s",
        ENABLE_CRYPTO_TRADING,
    )

    logger.info(
        "=========================================="
    )

    while True:

        try:
            run_cycle()

        except KeyboardInterrupt:

            logger.info(
                "Trading engine stopped."
            )
            break

        except Exception as exc:

            logger.exception(
                "Fatal cycle error: %s",
                exc,
            )

        logger.info(
            "Sleeping %s seconds...",
            CYCLE_SECONDS,
        )

        time.sleep(CYCLE_SECONDS)


if __name__ == "__main__":
    main()
