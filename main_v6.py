# =====================================================================
# # BULL. BEAR AND BROKE - TRADING ENGINE (main_v10_dynamic.py)
# Dynamic Momentum Screener with Bulk Batch Processing & Smart Watchlists
# =====================================================================

import os
import sys
import time
import sqlite3
import datetime
import logging
import zoneinfo

from logging.handlers import RotatingFileHandler
from typing import Tuple, List, Dict, Optional, Any, Set

import pandas as pd
import numpy as np
import ta
import yfinance as yf

from alpaca.trading.client import TradingClient
from alpaca.trading.requests import (
    LimitOrderRequest,
    TakeProfitRequest,
    StopLossRequest,
    GetOrdersRequest,
    GetAssetsRequest,
)
from alpaca.trading.enums import (
    OrderSide,
    TimeInForce,
    OrderClass,
    QueryOrderStatus,
    AssetStatus,
    AssetExchange,
)
from alpaca.data.historical import (
    StockHistoricalDataClient,
    CryptoHistoricalDataClient,
)
from alpaca.data.requests import (
    StockBarsRequest,
    CryptoBarsRequest,
    StockLatestBarRequest,
    CryptoLatestBarRequest,
)
from alpaca.data.timeframe import TimeFrame
from alpaca.data.enums import DataFeed

# =====================================================================
# 1. LOGGING SETUP
# =====================================================================

LOG_DIR = "logs"
os.makedirs(LOG_DIR, exist_ok=True)

logger = logging.getLogger("trading_engine")
logger.setLevel(logging.DEBUG)
logger.propagate = False

_fmt = logging.Formatter(
    "%(asctime)s | %(levelname)-7s | %(filename)s:%(lineno)d | %(message)s"
)

console_handler = logging.StreamHandler(sys.stdout)
console_handler.setLevel(logging.DEBUG)
console_handler.setFormatter(_fmt)

file_handler = RotatingFileHandler(
    os.path.join(LOG_DIR, "trading_engine.log"),
    maxBytes=10 * 1024 * 1024,
    backupCount=5,
    encoding="utf-8",
)
file_handler.setLevel(logging.DEBUG)
file_handler.setFormatter(_fmt)

if not logger.handlers:
    logger.addHandler(console_handler)
    logger.addHandler(file_handler)

decision_logger = logging.getLogger("decision_engine")
decision_logger.setLevel(logging.DEBUG)
decision_logger.propagate = False

decision_file = RotatingFileHandler(
    os.path.join(LOG_DIR, "decisions.log"),
    maxBytes=5 * 1024 * 1024,
    backupCount=3,
    encoding="utf-8",
)

decision_file.setFormatter(
    logging.Formatter("%(asctime)s | %(message)s")
)

if not decision_logger.handlers:
    decision_logger.addHandler(decision_file)

def log_decision(msg: str):
    decision_logger.info(msg)
    logger.info(f"DECISION: {msg}")

# =====================================================================
# 2. CONFIGURATION & CREDENTIALS
# =====================================================================

ALPACA_API_KEY = (
    os.environ.get("APCA_API_KEY_ID")
    or os.environ.get("ALPACA_API_KEY")
)

ALPACA_SECRET_KEY = (
    os.environ.get("APCA_API_SECRET_KEY")
    or os.environ.get("ALPACA_SECRET_KEY")
)

PAPER_TRADING = (
    os.environ.get("PAPER_TRADING", "true").lower() == "true"
)

if not ALPACA_API_KEY or not ALPACA_SECRET_KEY:
    logger.error("❌ Missing Alpaca API credentials!")
    sys.exit(1)

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

# =====================================================================
# 3. STRATEGY CONFIGURATION (DYNAMIC MOMENTUM SCREENER)
# =====================================================================

DB_FILE = os.path.join(LOG_DIR, "trading_state.db")

# Core liquid crypto pairs always kept on watch for fast momentum
CRYPTO_TARGETS = [
    "BTC/USD", "ETH/USD", "SOL/USD", "AVAX/USD", "DOGE/USD"
]

MAX_OPEN_POSITIONS = 3
MAX_TOTAL_ACTIVE_TRADES = 3

# Fast re-entry cooldown
ENTRY_COOLDOWN_SECONDS = 120

# Risk configuration
RISK_PER_TRADE_PCT = 0.01
MAX_POSITION_PCT = 0.15

# Daily protection
MAX_DAILY_DRAWDOWN_PCT = 0.04

# Indicator configuration
RSI_WINDOW = 14
OVERSOLD_RSI = 48
REVERSAL_RSI = 45

EMA_FAST = 20
EMA_SLOW = 50

# Asymmetric reward-to-risk multipliers (~3.5x reward profile)
ATR_MULTIPLIER_STOP = 1.5
ATR_MULTIPLIER_TARGET = 5.25

# Limit order offset
ENTRY_LIMIT_BUFFER_PCT = 0.0005

# 4H data cache
CACHE_TTL_4H_SECONDS = 900

FOUR_HOUR_CACHE: Dict[
    str,
    Tuple[pd.DataFrame, datetime.datetime]
] = {}

# =====================================================================
# 4. DATABASE STATE MANAGEMENT
# =====================================================================

def init_db():
    conn = sqlite3.connect(DB_FILE)
    cursor = conn.cursor()

    cursor.execute(
        """
        CREATE TABLE IF NOT EXISTS state (
            key TEXT PRIMARY KEY,
            value TEXT
        )
        """
    )

    cursor.execute(
        """
        CREATE TABLE IF NOT EXISTS trade_journal (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp TEXT,
            symbol TEXT,
            side TEXT,
            qty REAL,
            entry_price REAL,
            stop_loss REAL,
            take_profit REAL,
            status TEXT
        )
        """
    )

    conn.commit()
    conn.close()

def set_db_state(key: str, value: str):
    conn = sqlite3.connect(DB_FILE)
    cursor = conn.cursor()

    cursor.execute(
        """
        INSERT OR REPLACE INTO state (key, value)
        VALUES (?, ?)
        """,
        (key, value),
    )

    conn.commit()
    conn.close()

def get_db_state(key: str) -> Optional[str]:
    conn = sqlite3.connect(DB_FILE)
    cursor = conn.cursor()

    cursor.execute(
        "SELECT value FROM state WHERE key = ?",
        (key,),
    )

    row = cursor.fetchone()

    conn.close()

    return row[0] if row else None

def record_trade(
    symbol: str,
    side: str,
    qty: float,
    entry_price: float,
    stop_loss: float,
    take_profit: float,
    status: str,
):
    conn = sqlite3.connect(DB_FILE)
    cursor = conn.cursor()

    cursor.execute(
        """
        INSERT INTO trade_journal (
            timestamp,
            symbol,
            side,
            qty,
            entry_price,
            stop_loss,
            take_profit,
            status
        )
        VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (
            datetime.datetime.now(
                datetime.timezone.utc
            ).isoformat(),
            symbol,
            side,
            qty,
            entry_price,
            stop_loss,
            take_profit,
            status,
        ),
    )

    conn.commit()
    conn.close()

# =====================================================================
# 5. COOLDOWN MANAGEMENT
# =====================================================================

def cooldown_key(symbol: str) -> str:
    clean_symbol = symbol.replace("/", "_")
    return f"last_entry_time:{clean_symbol}"

def is_symbol_on_cooldown(symbol: str) -> bool:
    value = get_db_state(cooldown_key(symbol))

    if not value:
        return False

    try:
        last_entry = float(value)
        elapsed = time.time() - last_entry

        return elapsed < ENTRY_COOLDOWN_SECONDS

    except (TypeError, ValueError):
        return False

def set_symbol_cooldown(symbol: str):
    set_db_state(
        cooldown_key(symbol),
        str(time.time()),
    )

def cooldown_remaining(symbol: str) -> int:
    value = get_db_state(cooldown_key(symbol))

    if not value:
        return 0

    try:
        elapsed = time.time() - float(value)
        remaining = ENTRY_COOLDOWN_SECONDS - elapsed

        return max(0, int(remaining))

    except (TypeError, ValueError):
        return 0

# =====================================================================
# 6. DYNAMIC MOMENTUM SCREENER & WATCHLIST GENERATOR
# =====================================================================

def get_dynamic_momentum_watchlist(max_stocks: int = 15) -> List[str]:
    """
    Dynamically scans a pool of candidate small/mid-cap high-beta equities
    and high-momentum crypto targets using bulk latest bar requests to avoid rate limits.
    """
    # Curated pool of high-beta / small-mid cap symbols prone to momentum moves
    candidate_stocks = [
        "PLTR", "SOFI", "NIO", "MARA", "RIOT", "HOOD", 
        "RIVN", "SNAP", "AMC", "GME", "PTON", "AAL",
        "SPY", "QQQ", "AMD", "COIN", "DKNG", "UBER"
    ]
    
    scored_targets = []
    
    try:
        # Bulk fetch latest bars in a single request (zero rate-limit penalty)
        request_params = StockLatestBarRequest(symbol_or_symbols=candidate_stocks)
        latest_bars = stock_data_client.get_stock_latest_bar(request_params)
        
        for symbol, bar in latest_bars.items():
            if bar and bar.open > 0:
                # Calculate intraday return percentage
                price_change_pct = (bar.close - bar.open) / bar.open * 100
                scored_targets.append((symbol, price_change_pct))
                
        # Sort by strongest upside price momentum
        scored_targets.sort(key=lambda x: x[1], reverse=True)
        
        # Take the top performing stocks
        top_stocks = [item[0] for item in scored_targets[:max_stocks]]
        
        logger.info(f"[SCREENER] Dynamic watchlist updated with top active movers: {top_stocks}")
        return top_stocks + CRYPTO_TARGETS
        
    except Exception as e:
        logger.warning(f"[SCREENER] Failed dynamic stock scan: {e}. Falling back to default list.")
        return ["PLTR", "SOFI", "MARA", "RIOT", "SPY", "QQQ"] + CRYPTO_TARGETS

# =====================================================================
# 7. MARKET DATA (WITH RETRIES & ROUTING)
# =====================================================================

def fetch_4h_bars_cached(
    ticker: str,
) -> Optional[pd.DataFrame]:

    now_utc = datetime.datetime.now(
        datetime.timezone.utc
    )

    if ticker in FOUR_HOUR_CACHE:

        cached_df, last_fetch = FOUR_HOUR_CACHE[ticker]

        if (
            now_utc - last_fetch
        ).total_seconds() < CACHE_TTL_4H_SECONDS:

            return cached_df

    try:

        ticker_obj = yf.Ticker(ticker)

        df_4h = ticker_obj.history(
            period="60d",
            interval="60m",
        )

        if df_4h.empty:
            logger.warning(
                f"[{ticker}] Empty 4H source data."
            )
            return None

        df_4h = df_4h.resample(
            "4h",
            origin="start",
        ).agg(
            {
                "Open": "first",
                "High": "max",
                "Low": "min",
                "Close": "last",
                "Volume": "sum",
            }
        ).dropna()

        if len(df_4h) > 1:
            df_4h = df_4h.iloc[:-1]

        if len(df_4h) < EMA_SLOW:
            logger.warning(
                f"[{ticker}] Only {len(df_4h)} "
                "4H bars available."
            )
            return None

        FOUR_HOUR_CACHE[ticker] = (
            df_4h,
            now_utc,
        )

        return df_4h

    except Exception as e:
        logger.error(f"[{ticker}] Failed fetching 4H bars: {e}")

    if ticker in FOUR_HOUR_CACHE:
        return FOUR_HOUR_CACHE[ticker][0]

    return None

def fetch_1m_bars(
    ticker: str,
    limit: int = 120,
    retries: int = 3,
    delay: int = 2,
) -> Optional[pd.DataFrame]:

    end_dt = datetime.datetime.now(
        datetime.timezone.utc
    )
    start_dt = end_dt - datetime.timedelta(
        minutes=limit + 120
    )

    is_crypto = "/" in ticker

    for attempt in range(retries):
        try:
            if is_crypto:
                request_params = CryptoBarsRequest(
                    symbol_or_symbols=ticker,
                    timeframe=TimeFrame.Minute,
                    start=start_dt,
                    end=end_dt,
                )
                bars = crypto_data_client.get_crypto_bars(
                    request_params
                )
            else:
                request_params = StockBarsRequest(
                    symbol_or_symbols=ticker,
                    timeframe=TimeFrame.Minute,
                    start=start_dt,
                    end=end_dt,
                    feed=DataFeed.IEX,
                )
                bars = stock_data_client.get_stock_bars(
                    request_params
                )

            df = bars.df

            if df.empty:
                if attempt < retries - 1:
                    time.sleep(delay)
                    continue
                return None

            if isinstance(df.index, pd.MultiIndex):
                df = df.xs(
                    ticker,
                    level=0,
                )

            df = df.sort_index()
            return df.tail(limit)

        except Exception as e:
            if attempt < retries - 1:
                time.sleep(delay)
            else:
                return None

    return None

# =====================================================================
# 8. INDICATORS, VOLATILITY & RVOL FILTERS
# =====================================================================

def get_column(
    df: pd.DataFrame,
    lower_name: str,
    upper_name: str,
) -> pd.Series:

    if lower_name in df.columns:
        return df[lower_name]

    return df[upper_name]

def calculate_atr(
    high: pd.Series,
    low: pd.Series,
    close: pd.Series,
    window: int = 14,
) -> float:

    previous_close = close.shift(1)

    high_low = high - low
    high_close = (
        high - previous_close
    ).abs()

    low_close = (
        low - previous_close
    ).abs()

    true_range = pd.concat(
        [
            high_low,
            high_close,
            low_close,
        ],
        axis=1,
    ).max(axis=1)

    atr = true_range.rolling(
        window
    ).mean().iloc[-1]

    return float(atr)

def passes_volatility_filter(df_1m: pd.DataFrame) -> bool:
    try:
        close = get_column(df_1m, "close", "Close")
        high = get_column(df_1m, "high", "High")
        low = get_column(df_1m, "low", "Low")
        volume = get_column(df_1m, "volume", "Volume")

        atr = calculate_atr(high, low, close, window=14)
        current_price = float(close.iloc[-1])
        if current_price <= 0:
            return False
            
        atr_percentage = (atr / current_price) * 100

        avg_vol = volume.rolling(window=20).mean().iloc[-1]
        current_vol = volume.iloc[-1]
        
        if pd.isna(avg_vol) or avg_vol == 0:
            rvol = 1.0
        else:
            rvol = current_vol / avg_vol

        return atr_percentage >= 1.0 and rvol >= 0.7
    except Exception as e:
        logger.warning(f"Volatility filter check failed: {e}")
        return True

def analyze_indicators(
    df_1m: pd.DataFrame,
    df_4h: Optional[pd.DataFrame] = None,
) -> Dict[str, Any]:

    close_1m = get_column(df_1m, "close", "Close")
    high_1m = get_column(df_1m, "high", "High")
    low_1m = get_column(df_1m, "low", "Low")

    rsi_series = ta.momentum.rsi(
        close_1m,
        window=RSI_WINDOW,
    )

    rsi = float(rsi_series.iloc[-1])
    previous_rsi = float(rsi_series.iloc[-2])

    rsi_reversal = (
        previous_rsi < OVERSOLD_RSI
        and rsi >= OVERSOLD_RSI
    )

    rsi_recovering = (
        rsi > previous_rsi
        and rsi >= REVERSAL_RSI
    )

    volume_spike = True

    ema_fast_1m = ta.trend.ema_indicator(
        close_1m,
        window=9,
    )

    ema_slow_1m = ta.trend.ema_indicator(
        close_1m,
        window=21,
    )

    short_term_bullish = (
        close_1m.iloc[-1]
        > ema_fast_1m.iloc[-1]
        and ema_fast_1m.iloc[-1]
        >= ema_slow_1m.iloc[-1]
    )

    recent_high = (
        high_1m
        .iloc[-6:-1]
        .max()
    )

    breakout_confirmation = (
        close_1m.iloc[-1]
        >= recent_high
    )

    close_15m = (
        close_1m
        .resample("15min")
        .last()
        .dropna()
    )

    macd_diff = 0.0
    macd_improving = True

    if len(close_15m) >= 35:
        macd_series = ta.trend.macd_diff(
            close_15m,
            window_slow=26,
            window_fast=12,
            window_sign=9,
        )
        macd_diff = float(macd_series.iloc[-1])
        previous_macd_diff = float(macd_series.iloc[-2])
        macd_improving = (
            macd_diff >= previous_macd_diff
            or macd_diff > -0.05
        )

    df_15m = (
        df_1m
        .resample("15min")
        .agg(
            {
                get_column(df_1m, "open", "Open").name: "first",
                high_1m.name: "max",
                low_1m.name: "min",
                close_1m.name: "last",
            }
        )
        .dropna()
    )

    if len(df_15m) >= 15:
        atr_15m = calculate_atr(
            df_15m[high_1m.name],
            df_15m[low_1m.name],
            df_15m[close_1m.name],
        )
    else:
        atr_15m = float(close_1m.iloc[-1]) * 0.01

    trend_4h = "BULLISH"
    if df_4h is not None and len(df_4h) >= EMA_SLOW:
        close_4h = df_4h["Close"]
        ema_20_4h = ta.trend.ema_indicator(close_4h, window=EMA_FAST).iloc[-1]
        ema_50_4h = ta.trend.ema_indicator(close_4h, window=EMA_SLOW).iloc[-1]
        trend_4h = "BULLISH" if ema_20_4h >= ema_50_4h else "BEARISH"

    return {
        "rsi": rsi,
        "previous_rsi": previous_rsi,
        "rsi_reversal": rsi_reversal,
        "rsi_recovering": rsi_recovering,
        "volume_spike": volume_spike,
        "macd_diff": macd_diff,
        "macd_improving": macd_improving,
        "short_term_bullish": short_term_bullish,
        "breakout_confirmation": breakout_confirmation,
        "trend_4h": trend_4h,
        "latest_price": float(close_1m.iloc[-1]),
        "atr_15m": atr_15m if atr_15m > 0 else float(close_1m.iloc[-1]) * 0.01,
    }

# =====================================================================
# 9. SIGNAL ENGINE
# =====================================================================

def bullish_signal(
    indicators: Dict[str, Any],
) -> Tuple[bool, List[str]]:

    reasons = []

    rsi_ok = (
        indicators["rsi_reversal"]
        or indicators["rsi_recovering"]
        or indicators["rsi"] >= 45
    )

    if not rsi_ok:
        return False, ["RSI momentum insufficient"]

    reasons.append("RSI structure favorable")

    if not indicators["macd_improving"]:
        return False, ["MACD weak"]

    reasons.append("MACD acceptable")

    price_confirmation = (
        indicators["short_term_bullish"]
        or indicators["breakout_confirmation"]
    )

    if not price_confirmation:
        return False, ["No momentum confirmation"]

    reasons.append("momentum breakout confirmed")

    return True, reasons

# =====================================================================
# 10. RISK & CIRCUIT BREAKERS
# =====================================================================

def get_account_equity() -> Optional[float]:
    try:
        account = trading_client.get_account()
        return float(account.equity)
    except Exception as e:
        logger.error(f"Failed to fetch account equity: {e}")
        return None

def audit_portfolio_risk_state() -> bool:
    try:
        account = trading_client.get_account()
        equity = float(account.equity)
        last_equity = float(account.last_equity)

        if last_equity <= 0:
            return True

        drawdown_pct = (equity - last_equity) / last_equity
        today_str = datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%d")

        if get_db_state("circuit_breaker_date") == today_str:
            return True

        if drawdown_pct <= -MAX_DAILY_DRAWDOWN_PCT:
            log_decision(f"🚨 CIRCUIT BREAKER TRIPPED! Drawdown={drawdown_pct * 100:.2f}%")
            trading_client.close_all_positions(cancel_orders=True)
            set_db_state("circuit_breaker_date", today_str)
            return True
    except Exception as e:
        logger.error(f"Error checking portfolio risk state: {e}")

    return False

def calculate_position_size(
    equity: float,
    price: float,
    atr: float,
    risk_pct: float = RISK_PER_TRADE_PCT,
) -> float:
    if equity <= 0 or price <= 0 or atr <= 0:
        return 0.0

    risk_dollar = equity * risk_pct
    stop_distance = ATR_MULTIPLIER_STOP * atr
    shares_by_risk = risk_dollar / stop_distance
    max_position_dollar = equity * MAX_POSITION_PCT
    max_shares_by_cap = max_position_dollar / price

    shares = min(shares_by_risk, max_shares_by_cap)

    if "/" in str(price):
        return max(0.0, round(shares, 4))
    
    return float(max(0, int(shares)))

# =====================================================================
# 11. POSITION / ORDER PROTECTION
# =====================================================================

def get_open_position_symbols() -> Set[str]:
    try:
        positions = trading_client.get_all_positions()
        return {position.symbol for position in positions}
    except Exception as e:
        logger.error(f"Failed fetching positions: {e}")
        return set()

def get_pending_order_symbols() -> Set[str]:
    try:
        request = GetOrdersRequest(status=QueryOrderStatus.OPEN, nested=True)
        orders = trading_client.get_orders(filter=request)
        return {order.symbol for order in orders if hasattr(order, "symbol")}
    except Exception as e:
        logger.error(f"Failed fetching pending orders: {e}")
        return set()

def symbol_available_for_entry(
    symbol: str,
    held_symbols: Set[str],
    pending_symbols: Set[str],
) -> bool:
    if symbol in held_symbols or symbol in pending_symbols:
        return False

    if is_symbol_on_cooldown(symbol):
        return False

    return True

# =====================================================================
# 12. ORDER EXECUTION (GTC TIME-IN-FORCE)
# =====================================================================

def calculate_buy_limit_price(latest_price: float) -> float:
    return round(latest_price * (1 + ENTRY_LIMIT_BUFFER_PCT), 2)

def place_long_bracket_order(
    symbol: str,
    qty: float,
    latest_price: float,
    atr: float,
) -> Optional[Any]:

    if qty <= 0 or latest_price <= 0 or atr <= 0:
        return None

    entry_price = calculate_buy_limit_price(latest_price)
    stop_loss = round(entry_price - (ATR_MULTIPLIER_STOP * atr), 2)
    take_profit = round(entry_price + (ATR_MULTIPLIER_TARGET * atr), 2)

    if stop_loss <= 0:
        return None

    try:
        order_data = LimitOrderRequest(
            symbol=symbol,
            limit_price=entry_price,
            qty=qty,
            side=OrderSide.BUY,
            time_in_force=TimeInForce.GTC,
            order_class=OrderClass.BRACKET,
            stop_loss=StopLossRequest(stop_price=stop_loss),
            take_profit=TakeProfitRequest(limit_price=take_profit),
        )

        response = trading_client.submit_order(order_data)
        set_symbol_cooldown(symbol)

        record_trade(
            symbol=symbol,
            side="BUY",
            qty=qty,
            entry_price=entry_price,
            stop_loss=stop_loss,
            take_profit=take_profit,
            status="SUBMITTED",
        )

        log_decision(
            f"🚀 ORDER SUBMITTED (GTC) | {symbol} BUY {qty} | "
            f"Entry={entry_price:.2f} | SL={stop_loss:.2f} | TP={take_profit:.2f}"
        )

        return response

    except Exception as e:
        logger.error(f"❌ Failed placing order for {symbol}: {e}")
        return None

# =====================================================================
# 13. MARKET SESSION MANAGEMENT
# =====================================================================

def market_is_tradeable(is_crypto: bool = False) -> bool:
    if is_crypto:
        return True

    try:
        clock = trading_client.get_clock()
        if not clock.is_open:
            return False

        now_et = datetime.datetime.now(zoneinfo.ZoneInfo("America/New_York"))
        market_close = now_et.replace(hour=16, minute=0, second=0, microsecond=0)
        seconds_to_close = (market_close - now_et).total_seconds()

        if 0 < seconds_to_close <= 600:
            log_decision("⏰ Within 10 minutes of market close. Liquidating.")
            trading_client.close_all_positions(cancel_orders=True)
            return False

        return True
    except Exception as e:
        logger.error(f"Failed checking market clock: {e}")
        return False

# =====================================================================
# 14. MAIN TRADING CYCLE (DYNAMIC SCREENER LOOP)
# =====================================================================

def run_cycle():
    logger.info("=== STARTING TRADING CYCLE ===")

    if audit_portfolio_risk_state():
        return

    equity = get_account_equity()
    if equity is None or equity <= 0:
        return

    held_symbols = get_open_position_symbols()
    pending_symbols = get_pending_order_symbols()
    active_symbols = held_symbols | pending_symbols

    logger.info(
        f"Portfolio state | Positions={len(held_symbols)} | "
        f"Pending={len(pending_symbols)} | Active={len(active_symbols)}"
    )

    if len(active_symbols) >= MAX_TOTAL_ACTIVE_TRADES:
        logger.info("Maximum active trade capacity reached.")
        return

    # Dynamically generate fresh high-momentum target list every cycle
    current_targets = get_dynamic_momentum_watchlist(max_stocks=12)

    for ticker in current_targets:
        is_crypto_target = "/" in ticker

        if not is_crypto_target and not market_is_tradeable(is_crypto=False):
            continue

        active_symbols = get_open_position_symbols() | get_pending_order_symbols()
        if len(active_symbols) >= MAX_TOTAL_ACTIVE_TRADES:
            logger.info("Active trade capacity reached during scan.")
            break

        if not symbol_available_for_entry(ticker, get_open_position_symbols(), get_pending_order_symbols()):
            continue

        df_1m = fetch_1m_bars(ticker, limit=120)
        if df_1m is None or len(df_1m) < 30:
            continue

        if not passes_volatility_filter(df_1m):
            logger.info(f"[{ticker}] Skipped: Fails momentum/RVOL hurdle.")
            continue

        df_4h = fetch_4h_bars_cached("SPY" if not is_crypto_target else "BTC-USD")
        indicators = analyze_indicators(df_1m, df_4h)

        valid, reasons = bullish_signal(indicators)

        logger.info(
            f"[{ticker}] Price={indicators['latest_price']:.2f} | "
            f"RSI={indicators['rsi']:.1f} | MACD={indicators['macd_diff']:.4f} | Signal={valid}"
        )

        if not valid:
            continue

        if not symbol_available_for_entry(ticker, get_open_position_symbols(), get_pending_order_symbols()):
            continue

        qty = calculate_position_size(
            equity=equity,
            price=indicators["latest_price"],
            atr=indicators["atr_15m"],
        )

        if qty <= 0:
            continue

        log_decision(f"📈 SIGNAL VALID | {ticker} | " + " | ".join(reasons))

        response = place_long_bracket_order(
            symbol=ticker,
            qty=qty,
            latest_price=indicators["latest_price"],
            atr=indicators["atr_15m"],
        )

        if response is not None:
            time.sleep(2)

# =====================================================================
# 15. MAIN
# =====================================================================

def main():
    init_db()
    logger.info("🤖 Bull. Bear and Broke - Trading Engine Online (v10 Dynamic Momentum)")
    logger.info(f"Paper trading: {PAPER_TRADING}")

    while type(True):
        try:
            run_cycle()
        except KeyboardInterrupt:
            logger.info("Trading engine stopped by user.")
            break
        except Exception as e:
            logger.exception(f"❌ Critical error in main trading loop: {e}")

        logger.info("💤 Sleeping for 180 seconds (3 minutes) before next cycle...")
        time.sleep(180)

if __name__ == "__main__":
    main()
