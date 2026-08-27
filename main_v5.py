import os
import sys
import time
import sqlite3
import datetime
import zoneinfo
import logging
from typing import Tuple, List, Dict, Optional, Any

import pandas as pd
import numpy as np
import ta
import yfinance as yf

from alpaca.trading.client import TradingClient
from alpaca.trading.requests import MarketOrderRequest, TakeProfitRequest, StopLossRequest
from alpaca.trading.enums import OrderSide, TimeInForce, OrderClass
from alpaca.data.historical import StockHistoricalDataClient
from alpaca.data.requests import StockBarsRequest
from alpaca.data.timeframe import TimeFrame
from alpaca.data.enums import DataFeed

# =====================================================================
# 1. SETUP LOGGING & ENVIRONMENT
# =====================================================================
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler("trading_engine.log", mode="a")
    ]
)

ALPACA_API_KEY = os.getenv("ALPACA_API_KEY", "")
ALPACA_SECRET_KEY = os.getenv("ALPACA_SECRET_KEY", "")
PAPER_TRADING = os.getenv("PAPER_TRADING", "true").lower() == "true"
DEBUG_VERBOSE = os.getenv("DEBUG_VERBOSE", "false").lower() == "true"

if not ALPACA_API_KEY or not ALPACA_SECRET_KEY:
    logging.warning("[WARN] Missing Alpaca API credentials in environment variables!")

trading_client = TradingClient(ALPACA_API_KEY, ALPACA_SECRET_KEY, paper=PAPER_TRADING)
data_client = StockHistoricalDataClient(ALPACA_API_KEY, ALPACA_SECRET_KEY)

DB_FILE = "trading_decisions.db"
DEFAULT_DOLLAR_ALLOCATION = 1000.0
MAX_OPEN_POSITIONS = 3
ENTRY_COOLDOWN_SECONDS = 3600  # 1-hour per-ticker re-entry cooldown
EASTERN_TZ = zoneinfo.ZoneInfo("America/New_York")

CRYPTO_ADJACENT_TICKERS = ["MSTR", "COIN", "MARA", "RIOT"]
STANDARD_TICKERS = ["SPY", "QQQ", "AAPL", "NVDA", "AMD", "MSFT", "TSLA"]
ALL_TICKERS = STANDARD_TICKERS + CRYPTO_ADJACENT_TICKERS

RISK_PARAMS = {
    "MAX_PORTFOLIO_DRAWDOWN_PCT": 0.05,
    "DAILY_CIRCUIT_BREAKER_LOSS_PCT": 0.04,
    "MAX_POSITION_LOSS_PCT": 0.03,        # Broker-side bracket stop-loss (-3%)
    "FALLBACK_POSITION_LOSS_PCT": 0.045,  # Engine safety backstop (-4.5% / 1.5x)
    "TAKE_PROFIT_PCT": 0.06,              # Broker-side bracket take-profit (+6%)
    "MIN_ALLOCATION_PCT": 0.80            # Require at least 80% deployment ($800 minimum)
}

PROCESSED_BARS: Dict[str, datetime.datetime] = {}
SIGNAL_COOLDOWN_CACHE: Dict[str, datetime.datetime] = {}

# =====================================================================
# 2. DATABASE & LOGGING UTILITIES
# =====================================================================
def init_db():
    with sqlite3.connect(DB_FILE) as conn:
        cursor = conn.cursor()
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS audit_logs (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp TEXT NOT NULL,
                ticker TEXT NOT NULL,
                action TEXT NOT NULL,
                reason TEXT NOT NULL,
                details TEXT
            )
        """)
        conn.commit()

init_db()

def log_diagnostic(ticker: str, action: str, reason: str, details: str = ""):
    now_str = datetime.datetime.now(EASTERN_TZ).strftime("%Y-%m-%d %H:%M:%S EST")
    try:
        with sqlite3.connect(DB_FILE) as conn:
            cursor = conn.cursor()
            cursor.execute(
                "INSERT INTO audit_logs (timestamp, ticker, action, reason, details) VALUES (?, ?, ?, ?, ?)",
                (now_str, ticker, action, reason, details)
            )
            conn.commit()
    except Exception as e:
        logging.error(f"[DB ERROR] Failed to log diagnostic: {e}")
    if DEBUG_VERBOSE:
        logging.info(f"[DIAGNOSTIC] [{ticker}] {action} | {reason} | {details}")

# =====================================================================
# 3. PORTFOLIO & RISK HELPERS
# =====================================================================
def audit_portfolio_risk_state(active_positions: List[Any]):
    try:
        account = trading_client.get_account()
        equity = float(account.equity)
        last_equity = float(account.last_equity)
        
        portfolio_drawdown = (last_equity - equity) / last_equity if last_equity > 0 else 0

        if portfolio_drawdown >= RISK_PARAMS["DAILY_CIRCUIT_BREAKER_LOSS_PCT"] or \
           portfolio_drawdown >= RISK_PARAMS["MAX_PORTFOLIO_DRAWDOWN_PCT"]:
            logging.error(f"[CIRCUIT BREAKER] Drawdown {portfolio_drawdown:.2%} exceeds threshold. Liquidating portfolio!")
            close_all_open_positions()
            return

        for pos in active_positions:
            unrealized_plpc = float(pos.unrealized_plpc)
            if unrealized_plpc <= -RISK_PARAMS["FALLBACK_POSITION_LOSS_PCT"]:
                logging.warning(
                    f"[FALLBACK RISK STOP] {pos.symbol} reached {unrealized_plpc:.2%} "
                    f"(exceeded {RISK_PARAMS['FALLBACK_POSITION_LOSS_PCT']:.2%}). Closing directly."
                )
                trading_client.close_position(pos.symbol)
    except Exception as e:
        logging.error(f"[RISK AUDIT ERROR] Failed checking portfolio state: {e}")

def close_all_open_positions():
    try:
        trading_client.cancel_orders()
        trading_client.close_all_positions(cancel_orders=True)
        logging.info("[LIQUIDATION COMPLETE] All orders canceled and positions closed.")
    except Exception as e:
        logging.error(f"[LIQUIDATION ERROR] Failed liquidating positions: {e}")

# =====================================================================
# 4. MARKET DATA & TECHNICAL ANALYSIS
# =====================================================================
def fetch_market_bars(ticker: str) -> Tuple[Optional[pd.DataFrame], Optional[pd.DataFrame]]:
    now_utc = datetime.datetime.now(datetime.timezone.utc)
    start_utc = now_utc - datetime.timedelta(days=5)

    try:
        request = StockBarsRequest(
            symbol_or_symbols=ticker,
            timeframe=TimeFrame.Minute,
            start=start_utc,
            end=now_utc,
            feed=DataFeed.IEX
        )
        bars_1m = data_client.get_stock_bars(request).df
        if isinstance(bars_1m.index, pd.MultiIndex):
            bars_1m = bars_1m.xs(ticker)
    except Exception as e:
        logging.error(f"[{ticker}] Failed fetching 1m bars via Alpaca (IEX): {e}")
        bars_1m = None

    try:
        ticker_obj = yf.Ticker(ticker)
        df_4h = ticker_obj.history(period="10d", interval="60m")
        if not df_4h.empty:
            df_4h = df_4h.resample("4h").agg({
                'Open': 'first', 'High': 'max', 'Low': 'min', 'Close': 'last', 'Volume': 'sum'
            }).dropna()
        else:
            df_4h = None
    except Exception as e:
        logging.error(f"[{ticker}] Failed fetching 4h bars via yfinance: {e}")
        df_4h = None

    return bars_1m, df_4h

def analyze_indicators(df_1m: pd.DataFrame, df_4h: pd.DataFrame) -> Dict[str, Any]:
    close_1m = df_1m['close'] if 'close' in df_1m.columns else df_1m['Close']
    vol_1m = df_1m['volume'] if 'volume' in df_1m.columns else df_1m['Volume']
    
    rsi_1m = ta.momentum.rsi(close_1m, window=14).iloc[-1]
    
    # Pure trailing baseline: shift(1) excludes the current bar from its own SMA calculation
    vol_sma20 = vol_1m.shift(1).rolling(20).mean().iloc[-1]
    vol_spike = vol_1m.iloc[-1] > (1.2 * vol_sma20) if pd.notna(vol_sma20) and vol_sma20 > 0 else False
    
    close_4h = df_4h['Close']
    ema_20 = ta.trend.ema_indicator(close_4h, window=20).iloc[-1]
    ema_50 = ta.trend.ema_indicator(close_4h, window=50).iloc[-1]
    
    return {
        "rsi_1m": rsi_1m,
        "vol_spike": vol_spike,
        "trend_4h": "BULLISH" if ema_20 > ema_50 else "BEARISH",
        "latest_price": close_1m.iloc[-1]
    }

def process_pipeline(ticker: str, bars_1m: pd.DataFrame, df_4h: pd.DataFrame, active_positions: List[Any]):
    latest_bar_time = bars_1m.index[-1]
    if ticker in PROCESSED_BARS and PROCESSED_BARS[ticker] == latest_bar_time:
        return
    PROCESSED_BARS[ticker] = latest_bar_time

    indicators = analyze_indicators(bars_1m, df_4h)
    log_diagnostic(ticker, "SCAN", "Indicators Calculated", str(indicators))

    pos_symbols = [p.symbol for p in active_positions]

    if len(pos_symbols) >= MAX_OPEN_POSITIONS and ticker not in pos_symbols:
        log_diagnostic(ticker, "SKIP", f"Max concurrent positions reached ({len(pos_symbols)}/{MAX_OPEN_POSITIONS})")
        return

    # Signal Check: RSI < 30 + Volume Spike + 4h Bullish Trend
    if (indicators["rsi_1m"] < 30 and 
        indicators["vol_spike"] and 
        indicators["trend_4h"] == "BULLISH" and 
        ticker not in pos_symbols):
        
        if ticker in SIGNAL_COOLDOWN_CACHE:
            if (datetime.datetime.now(EASTERN_TZ) - SIGNAL_COOLDOWN_CACHE[ticker]).total_seconds() < ENTRY_COOLDOWN_SECONDS:
                log_diagnostic(ticker, "SKIP", "Re-entry Cooldown Active")
                return

        latest_price = float(indicators["latest_price"])
        qty = int(DEFAULT_DOLLAR_ALLOCATION // latest_price)
        allocated_dollars = qty * latest_price
        min_required_dollars = DEFAULT_DOLLAR_ALLOCATION * RISK_PARAMS["MIN_ALLOCATION_PCT"]

        if qty < 1 or allocated_dollars < min_required_dollars:
            log_diagnostic(ticker, "SKIP", f"Allocation efficiency low: ${allocated_dollars:.2f} < ${min_required_dollars:.2f}")
            return

        take_profit_price = round(latest_price * (1 + RISK_PARAMS["TAKE_PROFIT_PCT"]), 2)
        stop_loss_price = round(latest_price * (1 - RISK_PARAMS["MAX_POSITION_LOSS_PCT"]), 2)

        logging.info(f"[{ticker}] SIGNAL CONFIRMED: Submitting Bracket Order for {qty} shares.")
        SIGNAL_COOLDOWN_CACHE[ticker] = datetime.datetime.now(EASTERN_TZ)
        
        try:
            order_data = MarketOrderRequest(
                symbol=ticker,
                qty=qty,
                side=OrderSide.BUY,
                time_in_force=TimeInForce.DAY,
                order_class=OrderClass.BRACKET,
                take_profit=TakeProfitRequest(limit_price=take_profit_price),
                stop_loss=StopLossRequest(stop_price=stop_loss_price)
            )
            trading_client.submit_order(order_data)
            log_diagnostic(ticker, "BUY_BRACKET_SUBMITTED", f"Qty: {qty} | TP: ${take_profit_price} | SL: ${stop_loss_price}")
        except Exception as e:
            logging.error(f"[{ticker}] Bracket order execution failed: {e}")

# =====================================================================
# 5. UNIFIED MAIN LOOP
# =====================================================================
def run_trading_cycle() -> int:
    now_utc = datetime.datetime.now(datetime.timezone.utc)
    now_est = datetime.datetime.now(EASTERN_TZ)

    try:
        active_positions = trading_client.get_all_positions()
    except Exception as e:
        logging.error(f"[PORTFOLIO FETCH ERROR] Failed fetching positions: {e}")
        active_positions = []

    audit_portfolio_risk_state(active_positions)

    try:
        clock = trading_client.get_clock()
    except Exception as e:
        logging.error(f"[CLOCK FETCH ERROR] Could not verify market status: {e}")
        return 180

    sleep_interval = 180

    if clock.is_open:
        time_to_close = (clock.next_close - now_utc).total_seconds()
        if time_to_close <= 600:
            sleep_interval = 30

        if time_to_close <= 300:
            if active_positions:
                logging.info(f"[EOD LIQUIDATION] Liquidating positions before market close ({int(time_to_close)}s remaining)...")
                close_all_open_positions()
            return sleep_interval
    else:
        if active_positions:
            logging.warning("⚠️ [WARN] Orphaned positions detected after hours. Liquidating...")
            close_all_open_positions()

        logging.info(f"[MARKET CLOSED] Next open: {clock.next_open}.")
        return 180

    logging.info(f"Starting Market Scan [{now_est.strftime('%Y-%m-%d %H:%M:%S EST')}].")

    for ticker in ALL_TICKERS:
        try:
            bars_1m, df_4h = fetch_market_bars(ticker)
            if bars_1m is not None and df_4h is not None:
                process_pipeline(ticker, bars_1m, df_4h, active_positions)
        except Exception as e:
            log_diagnostic(ticker, "FETCH_ERROR", f"{type(e).__name__}: {e}")

    return sleep_interval

if __name__ == "__main__":
    logging.info("[ENGINE V5] Cleaned Continuous Loop Active.")
    while True:
        try:
            next_sleep = run_trading_cycle()
        except Exception as e:
            logging.error(f"[ERROR] Cycle failure: {e}")
            next_sleep = 180
            
        time.sleep(next_sleep)
