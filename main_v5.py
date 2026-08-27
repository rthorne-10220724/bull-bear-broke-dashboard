import os
import sys
import time
import uuid
import sqlite3
import datetime
import zoneinfo
import logging
from logging.handlers import RotatingFileHandler
from typing import Tuple, List, Dict, Optional, Any
import pandas as pd
import numpy as np
import ta
import yfinance as yf
from alpaca.trading.client import TradingClient
from alpaca.trading.requests import LimitOrderRequest, TakeProfitRequest, StopLossRequest, GetOrdersRequest
from alpaca.trading.enums import OrderSide, TimeInForce, OrderClass, QueryOrderStatus
from alpaca.data.historical import StockHistoricalDataClient
from alpaca.data.requests import StockBarsRequest
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
    "%(asctime)s [%(levelname)-8s] %(message)s", datefmt="%Y-%m-%d %H:%M:%S"
)

console_handler = logging.StreamHandler(sys.stdout)
console_handler.setLevel(logging.INFO)
console_handler.setFormatter(_fmt)

file_handler = RotatingFileHandler(
    os.path.join(LOG_DIR, "trading_engine.log"), maxBytes=10_000_000, backupCount=5
)
file_handler.setLevel(logging.DEBUG)
file_handler.setFormatter(_fmt)

logger.addHandler(console_handler)
logger.addHandler(file_handler)

decision_logger = logging.getLogger("trading_engine.decisions")
decision_logger.setLevel(logging.INFO)
decision_logger.propagate = False
decision_handler = RotatingFileHandler(
    os.path.join(LOG_DIR, "decisions.log"), maxBytes=10_000_000, backupCount=5
)
decision_handler.setFormatter(logging.Formatter("%(asctime)s | %(message)s", datefmt="%Y-%m-%d %H:%M:%S"))
decision_logger.addHandler(decision_handler)
decision_logger.addHandler(console_handler)


def log_decision(cycle_id: str, ticker: str, action: str, reason: str):
    decision_logger.info(f"[{cycle_id}] {ticker:<6} | {action:<20} | {reason}")


# =====================================================================
# 2. ENVIRONMENT / CONFIG & SECTOR BUCKETS
# =====================================================================
ALPACA_API_KEY = os.getenv("ALPACA_API_KEY", "")
ALPACA_SECRET_KEY = os.getenv("ALPACA_SECRET_KEY", "")
PAPER_TRADING = os.getenv("PAPER_TRADING", "true").lower() == "true"

if not ALPACA_API_KEY or not ALPACA_SECRET_KEY:
    logger.warning("[STARTUP] Missing Alpaca API credentials in environment variables!")

trading_client = TradingClient(ALPACA_API_KEY, ALPACA_SECRET_KEY, paper=PAPER_TRADING)
data_client = StockHistoricalDataClient(ALPACA_API_KEY, ALPACA_SECRET_KEY)

DB_FILE = "trading_decisions.db"
MAX_OPEN_POSITIONS = 3
ENTRY_COOLDOWN_SECONDS = 3600
CACHE_TTL_4H_SECONDS = 900  # 15-minute TTL for yfinance 4h bars
EASTERN_TZ = zoneinfo.ZoneInfo("America/New_York")

# Sector Buckets to prevent correlated wipeouts
SECTOR_BUCKETS = {
    "CRYPTO_ADJACENT": ["MSTR", "COIN", "MARA", "RIOT"],
    "SEMI_TECH": ["NVDA", "AMD", "AAPL", "MSFT"],
    "BROAD_INDEX": ["SPY", "QQQ"],
    "AUTO_TECH": ["TSLA"]
}

ALL_TICKERS = [ticker for group in SECTOR_BUCKETS.values() for ticker in group]

RISK_PARAMS = {
    "MAX_PORTFOLIO_DRAWDOWN_PCT": 0.05,
    "DAILY_CIRCUIT_BREAKER_LOSS_PCT": 0.04,
    "FALLBACK_POSITION_LOSS_PCT": 0.045,
    "RISK_PER_TRADE_PCT": 0.01,         # Risk exactly 1% of equity per trade
    "MAX_POSITION_ALLOCATION_PCT": 0.25, # Cap single position size at 25% equity
    "ATR_MULTIPLIER_SL": 1.5,           # Stop loss = 1.5x ATR
    "ATR_MULTIPLIER_TP": 3.5,           # Take profit target = 3.5x ATR
}

PROCESSED_BARS: Dict[str, datetime.datetime] = {}
SIGNAL_COOLDOWN_CACHE: Dict[str, datetime.datetime] = {}
FOUR_HOUR_CACHE: Dict[str, Tuple[pd.DataFrame, datetime.datetime]] = {}

TRADING_HALTED_FOR_DAY = False
HALT_DATE: Optional[datetime.date] = None


def reset_daily_halt_if_new_day():
    global TRADING_HALTED_FOR_DAY, HALT_DATE
    today = datetime.datetime.now(EASTERN_TZ).date()
    if HALT_DATE != today:
        TRADING_HALTED_FOR_DAY = False
        HALT_DATE = today


def get_ticker_bucket(ticker: str) -> Optional[str]:
    for bucket, tickers in SECTOR_BUCKETS.items():
        if ticker in tickers:
            return bucket
    return None


# =====================================================================
# 3. DATABASE
# =====================================================================
def init_db():
    with sqlite3.connect(DB_FILE) as conn:
        cursor = conn.cursor()
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS audit_logs (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp TEXT NOT NULL,
                cycle_id TEXT,
                ticker TEXT NOT NULL,
                action TEXT NOT NULL,
                reason TEXT NOT NULL,
                details TEXT
            )
        """)
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_audit_ticker_ts ON audit_logs(ticker, timestamp)")
        conn.commit()

init_db()


def log_diagnostic(cycle_id: str, ticker: str, action: str, reason: str, details: str = ""):
    now_str = datetime.datetime.now(EASTERN_TZ).strftime("%Y-%m-%d %H:%M:%S EST")
    try:
        with sqlite3.connect(DB_FILE) as conn:
            cursor = conn.cursor()
            cursor.execute(
                "INSERT INTO audit_logs (timestamp, cycle_id, ticker, action, reason, details) VALUES (?, ?, ?, ?, ?, ?)",
                (now_str, cycle_id, ticker, action, reason, details),
            )
            conn.commit()
    except Exception as e:
        logger.error(f"[DB ERROR] Failed to log diagnostic: {e}")
    logger.debug(f"[DIAGNOSTIC] [{ticker}] {action} | {reason} | {details}")


# =====================================================================
# 4. PORTFOLIO & RISK HELPERS
# =====================================================================
def audit_portfolio_risk_state(cycle_id: str, active_positions: List[Any]) -> List[Any]:
    global TRADING_HALTED_FOR_DAY
    try:
        account = trading_client.get_account()
        equity = float(account.equity)
        last_equity = float(account.last_equity)
        portfolio_drawdown = (last_equity - equity) / last_equity if last_equity > 0 else 0

        logger.debug(f"[RISK] Equity ${equity:.2f} | Last equity ${last_equity:.2f} | Drawdown {portfolio_drawdown:.2%}")

        if portfolio_drawdown >= RISK_PARAMS["DAILY_CIRCUIT_BREAKER_LOSS_PCT"] or \
           portfolio_drawdown >= RISK_PARAMS["MAX_PORTFOLIO_DRAWDOWN_PCT"]:
            logger.critical(
                f"[CIRCUIT BREAKER] Drawdown {portfolio_drawdown:.2%} exceeds threshold. "
                f"Liquidating portfolio and HALTING new entries for the rest of the day."
            )
            close_all_open_positions(cycle_id)
            TRADING_HALTED_FOR_DAY = True
            log_diagnostic(cycle_id, "PORTFOLIO", "CIRCUIT_BREAKER_TRIPPED",
                            f"Drawdown {portfolio_drawdown:.2%}", "Trading halted for remainder of day")
            return []

        for pos in active_positions:
            unrealized_plpc = float(pos.unrealized_plpc)
            if unrealized_plpc <= -RISK_PARAMS["FALLBACK_POSITION_LOSS_PCT"]:
                logger.warning(
                    f"[FALLBACK RISK STOP] {pos.symbol} reached {unrealized_plpc:.2%} "
                    f"(exceeded {RISK_PARAMS['FALLBACK_POSITION_LOSS_PCT']:.2%}). Closing directly."
                )
                try:
                    symbol_orders = trading_client.get_orders(
                        filter=GetOrdersRequest(status=QueryOrderStatus.OPEN, symbols=[pos.symbol])
                    )
                    for order in symbol_orders:
                        trading_client.cancel_order_by_id(order.id)
                except Exception as e:
                    logger.error(f"[{pos.symbol}] Failed cancelling symbol-specific orders before fallback close: {e}")
                trading_client.close_position(pos.symbol)
                log_diagnostic(cycle_id, pos.symbol, "FALLBACK_STOP", f"plpc={unrealized_plpc:.2%}")

    except Exception as e:
        logger.error(f"[RISK AUDIT ERROR] Failed checking portfolio state: {e}")

    return active_positions


def close_all_open_positions(cycle_id: str):
    try:
        trading_client.cancel_orders()
        trading_client.close_all_positions(cancel_orders=True)
        logger.info("[LIQUIDATION COMPLETE] All orders canceled and positions closed.")
        log_diagnostic(cycle_id, "PORTFOLIO", "LIQUIDATION_COMPLETE", "All positions and orders closed")
    except Exception as e:
        logger.error(f"[LIQUIDATION ERROR] Failed liquidating positions: {e}")


# =====================================================================
# 5. MARKET DATA & TECHNICAL ANALYSIS
# =====================================================================
def fetch_4h_bars_cached(ticker: str) -> Optional[pd.DataFrame]:
    now_utc = datetime.datetime.now(datetime.timezone.utc)
    if ticker in FOUR_HOUR_CACHE:
        cached_df, last_fetch = FOUR_HOUR_CACHE[ticker]
        if (now_utc - last_fetch).total_seconds() < CACHE_TTL_4H_SECONDS:
            return cached_df

    try:
        ticker_obj = yf.Ticker(ticker)
        df_4h = ticker_obj.history(period="10d", interval="60m")
        if not df_4h.empty:
            df_4h = df_4h.resample("4h").agg({
                'Open': 'first', 'High': 'max', 'Low': 'min', 'Close': 'last', 'Volume': 'sum'
            }).dropna()
            FOUR_HOUR_CACHE[ticker] = (df_4h, now_utc)
            return df_4h
    except Exception as e:
        logger.error(f"[{ticker}] Failed fetching 4h bars via yfinance: {e}")
        if ticker in FOUR_HOUR_CACHE:
            return FOUR_HOUR_CACHE[ticker][0]

    return None


def fetch_market_bars(ticker: str) -> Tuple[Optional[pd.DataFrame], Optional[pd.DataFrame]]:
    now_utc = datetime.datetime.now(datetime.timezone.utc)
    start_utc = now_utc - datetime.timedelta(days=5)

    try:
        request = StockBarsRequest(
            symbol_or_symbols=ticker, timeframe=TimeFrame.Minute,
            start=start_utc, end=now_utc, feed=DataFeed.IEX,
        )
        bars_1m = data_client.get_stock_bars(request).df
        if isinstance(bars_1m.index, pd.MultiIndex):
            bars_1m = bars_1m.xs(ticker)
    except Exception as e:
        logger.error(f"[{ticker}] Failed fetching 1m bars via Alpaca (IEX): {e}")
        bars_1m = None

    df_4h = fetch_4h_bars_cached(ticker)

    return bars_1m, df_4h


def analyze_indicators(df_1m: pd.DataFrame, df_4h: pd.DataFrame) -> Dict[str, Any]:
    close_1m = df_1m['close'] if 'close' in df_1m.columns else df_1m['Close']
    high_1m = df_1m['high'] if 'high' in df_1m.columns else df_1m['High']
    low_1m = df_1m['low'] if 'low' in df_1m.columns else df_1m['Low']
    vol_1m = df_1m['volume'] if 'volume' in df_1m.columns else df_1m['Volume']

    # 1. 1m RSI & Volume Spike
    rsi_1m = ta.momentum.rsi(close_1m, window=14).iloc[-1]
    vol_sma20 = vol_1m.shift(1).rolling(20).mean().iloc[-1]
    vol_spike = vol_1m.iloc[-1] > (1.2 * vol_sma20) if pd.notna(vol_sma20) and vol_sma20 > 0 else False

    # 2. Multi-Timeframe Confirmation (15m MACD)
    df_15m = close_1m.resample("15min").agg('last').dropna()
    macd_15m_val = 0.0
    if len(df_15m) >= 26:
        macd_15m_val = ta.trend.macd_diff(df_15m, window_slow=26, window_fast=12, window_sign=9).iloc[-1]

    # 3. 4h Trend EMA Alignment
    close_4h = df_4h['Close']
    ema_20 = ta.trend.ema_indicator(close_4h, window=20).iloc[-1]
    ema_50 = ta.trend.ema_indicator(close_4h, window=50).iloc[-1]

    # 4. Volatility (1m ATR calculation)
    high_low = high_1m - low_1m
    high_close = (high_1m - close_1m.shift()).abs()
    low_close = (low_1m - close_1m.shift()).abs()
    tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
    atr_1m = tr.rolling(14).mean().iloc[-1]

    return {
        "rsi_1m": rsi_1m,
        "vol_spike": vol_spike,
        "macd_15m_diff": macd_15m_val,
        "trend_4h": "BULLISH" if ema_20 > ema_50 else "BEARISH",
        "latest_price": float(close_1m.iloc[-1]),
        "atr_1m": float(atr_1m) if pd.notna(atr_1m) and atr_1m > 0 else float(close_1m.iloc[-1]) * 0.01,
    }


def process_pipeline(cycle_id: str, ticker: str, bars_1m: pd.DataFrame, df_4h: pd.DataFrame, active_positions: List[Any]):
    latest_bar_time = bars_1m.index[-1]
    if ticker in PROCESSED_BARS and PROCESSED_BARS[ticker] == latest_bar_time:
        log_decision(cycle_id, ticker, "SKIP", f"Bar already processed ({latest_bar_time})")
        return

    PROCESSED_BARS[ticker] = latest_bar_time
    indicators = analyze_indicators(bars_1m, df_4h)
    logger.debug(f"[{ticker}] Indicators: {indicators}")

    rsi = indicators["rsi_1m"]
    vol_spike = indicators["vol_spike"]
    macd_15m = indicators["macd_15m_diff"]
    trend = indicators["trend_4h"]
    latest_price = indicators["latest_price"]
    atr = indicators["atr_1m"]
    pos_symbols = [p.symbol for p in active_positions]

    # Rule 1: Prevent Pyramiding
    if ticker in pos_symbols:
        log_decision(cycle_id, ticker, "HOLD", "Already holding a position in this ticker")
        return

    # Rule 2: Max Open Positions Cap
    if len(pos_symbols) >= MAX_OPEN_POSITIONS:
        reason = f"Max open positions reached ({len(pos_symbols)}/{MAX_OPEN_POSITIONS})"
        log_decision(cycle_id, ticker, "HOLD_MAX_POS", reason)
        log_diagnostic(cycle_id, ticker, "HOLD_MAX_POSITIONS", reason)
        return

    # Rule 3: Sector Correlation Limit (Max 1 per bucket)
    ticker_bucket = get_ticker_bucket(ticker)
    for pos in pos_symbols:
        if get_ticker_bucket(pos) == ticker_bucket and ticker_bucket is not None:
            reason = f"Sector bucket cap reached ({ticker_bucket} active via {pos})"
            log_decision(cycle_id, ticker, "HOLD_SECTOR_CAP", reason)
            log_diagnostic(cycle_id, ticker, "HOLD_SECTOR_CAP", reason)
            return

    # Rule 4: Multi-Timeframe Signal Filters
    if rsi >= 30 or not vol_spike or trend != "BULLISH" or macd_15m <= 0:
        reasons = []
        if rsi >= 30:
            reasons.append(f"RSI {rsi:.2f}>=30")
        if not vol_spike:
            reasons.append("no vol spike")
        if trend != "BULLISH":
            reasons.append(f"4h trend {trend}")
        if macd_15m <= 0:
            reasons.append(f"15m MACD {macd_15m:.3f}<=0")
        reason_str = ", ".join(reasons)
        log_decision(cycle_id, ticker, "NO_SIGNAL", reason_str)
        log_diagnostic(cycle_id, ticker, "SKIP_SIGNAL", reason_str, str(indicators))
        return

    # Rule 5: Entry Cooldown Check
    if ticker in SIGNAL_COOLDOWN_CACHE:
        elapsed = (datetime.datetime.now(EASTERN_TZ) - SIGNAL_COOLDOWN_CACHE[ticker]).total_seconds()
        if elapsed < ENTRY_COOLDOWN_SECONDS:
            remaining = int(ENTRY_COOLDOWN_SECONDS - elapsed)
            reason = f"Cooldown active ({remaining}s remaining)"
            log_decision(cycle_id, ticker, "HOLD_COOLDOWN", reason)
            log_diagnostic(cycle_id, ticker, "HOLD_COOLDOWN", reason)
            return

    # Rule 6: ATR Volatility-Based Position Sizing
    try:
        account = trading_client.get_account()
        equity = float(account.equity)
        buying_power = float(account.buying_power)
    except Exception as e:
        logger.error(f"[{ticker}] Could not verify account state, skipping entry: {e}")
        return

    risk_per_share = round(atr * RISK_PARAMS["ATR_MULTIPLIER_SL"], 2)
    dollar_risk = equity * RISK_PARAMS["RISK_PER_TRADE_PCT"]
    qty = int(dollar_risk // risk_per_share) if risk_per_share > 0 else 0

    # Apply Maximum Single Position Capital Cap (25% Equity)
    max_position_dollars = equity * RISK_PARAMS["MAX_POSITION_ALLOCATION_PCT"]
    if (qty * latest_price) > max_position_dollars:
        qty = int(max_position_dollars // latest_price)

    allocated_dollars = qty * latest_price

    if qty < 1 or allocated_dollars > buying_power:
        reason = f"Allocation issue: qty={qty}, capital=${allocated_dollars:.2f}, BP=${buying_power:.2f}"
        log_decision(cycle_id, ticker, "HOLD_ALLOCATION", reason)
        log_diagnostic(cycle_id, ticker, "HOLD_ALLOCATION", reason)
        return

    # Calculate ATR Dynamic Bracket Levels
    stop_loss_price = round(latest_price - risk_per_share, 2)
    take_profit_price = round(latest_price + (atr * RISK_PARAMS["ATR_MULTIPLIER_TP"]), 2)

    # Marketable Limit Price (Ask + 0.05% buffer to prevent slippage)
    limit_entry_price = round(latest_price * 1.0005, 2)

    SIGNAL_COOLDOWN_CACHE[ticker] = datetime.datetime.now(EASTERN_TZ)

    try:
        order_data = LimitOrderRequest(
            symbol=ticker, qty=qty, side=OrderSide.BUY, time_in_force=TimeInForce.DAY,
            limit_price=limit_entry_price,
            order_class=OrderClass.BRACKET,
            take_profit=TakeProfitRequest(limit_price=take_profit_price),
            stop_loss=StopLossRequest(stop_price=stop_loss_price),
        )
        trading_client.submit_order(order_data)
        reason = f"qty={qty} limit=${limit_entry_price} TP=${take_profit_price} SL=${stop_loss_price} (ATR=${atr:.2f})"
        log_decision(cycle_id, ticker, "BUY_LIMIT_SUBMITTED", reason)
        log_diagnostic(cycle_id, ticker, "BUY_BRACKET_SUBMITTED", reason)
    except Exception as e:
        logger.error(f"[{ticker}] Bracket order execution failed: {e}")
        log_decision(cycle_id, ticker, "BUY_FAILED", str(e))


# =====================================================================
# 6. UNIFIED MAIN LOOP
# =====================================================================
def run_trading_cycle() -> int:
    cycle_id = uuid.uuid4().hex[:8]
    reset_daily_halt_if_new_day()

    now_utc = datetime.datetime.now(datetime.timezone.utc)
    now_est = datetime.datetime.now(EASTERN_TZ)

    try:
        active_positions = trading_client.get_all_positions()
    except Exception as e:
        logger.error(f"[PORTFOLIO FETCH ERROR] Failed fetching positions: {e}")
        active_positions = []

    active_positions = audit_portfolio_risk_state(cycle_id, active_positions)

    logger.info(
        f"[CYCLE {cycle_id}] START | {now_est.strftime('%H:%M:%S EST')} | "
        f"positions={[p.symbol for p in active_positions]} | halted_today={TRADING_HALTED_FOR_DAY}"
    )

    if TRADING_HALTED_FOR_DAY:
        logger.warning(f"[CYCLE {cycle_id}] Trading halted for the day (circuit breaker tripped earlier). Skipping scan.")
        return 180

    try:
        clock = trading_client.get_clock()
    except Exception as e:
        logger.error(f"[CLOCK FETCH ERROR] Could not verify market status: {e}")
        return 180

    sleep_interval = 180

    if clock.is_open:
        time_to_close = (clock.next_close - now_utc).total_seconds()

        if time_to_close <= 300:
            logger.info(f"[CYCLE {cycle_id}] Market closing in {int(time_to_close)}s. Buying frozen.")
            if active_positions:
                logger.info(f"[CYCLE {cycle_id}] EOD liquidation: closing open positions before close...")
                close_all_open_positions(cycle_id)
            return sleep_interval

        if time_to_close <= 600:
            logger.info(f"[CYCLE {cycle_id}] Approaching close ({int(time_to_close)}s left). Polling every 30s.")
            sleep_interval = 30
    else:
        logger.info(f"[CYCLE {cycle_id}] Market CLOSED. Next open: {clock.next_open}. Sleeping 3m.")
        if active_positions:
            logger.warning(f"[CYCLE {cycle_id}] Orphaned positions detected after hours. Liquidating...")
            close_all_open_positions(cycle_id)
        return 180

    scanned = 0
    for ticker in ALL_TICKERS:
        try:
            bars_1m, df_4h = fetch_market_bars(ticker)
            if bars_1m is not None and df_4h is not None:
                process_pipeline(cycle_id, ticker, bars_1m, df_4h, active_positions)
                scanned += 1
            else:
                log_decision(cycle_id, ticker, "SKIP", "Missing bar data (1m or 4h)")
        except Exception as e:
            logger.error(f"[{ticker}] Unhandled pipeline error: {e}")
            log_diagnostic(cycle_id, ticker, "FETCH_ERROR", f"{type(e).__name__}: {e}")

    logger.info(f"[CYCLE {cycle_id}] END | scanned {scanned}/{len(ALL_TICKERS)} tickers | next sleep {sleep_interval}s")
    return sleep_interval


if __name__ == "__main__":
    logger.info("[ENGINE V6] Starting continuous loop with ATR sizing & multi-timeframe confirmation.")
    while True:
        try:
            next_sleep = run_trading_cycle()
        except Exception as e:
            logger.error(f"[ERROR] Cycle failure: {e}")
            next_sleep = 180
        time.sleep(next_sleep)
