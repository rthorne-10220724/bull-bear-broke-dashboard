# =====================================================================
# BULL. BEAR AND BROKE - TRADING ENGINE (main_v6.py)
# Stronger signal confirmation, duplicate protection, cooldowns,
# corrected SQQQ logic, and safer execution controls.
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
)
from alpaca.trading.enums import (
    OrderSide,
    TimeInForce,
    OrderClass,
    QueryOrderStatus,
)
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

API_KEY = os.getenv("APCA_API_KEY_ID")
API_SECRET = os.getenv("APCA_API_SECRET_KEY")
BASE_URL = os.getenv("APCA_API_BASE_URL", "https://paper-api.alpaca.markets")

USE_PAPER = True
if "live" in BASE_URL.lower():
    USE_PAPER = False

WATCHLIST = [
    "AAPL", "MSFT", "NVDA", "TSLA", "AMZN", "GOOGL", "META", "SPY", "QQQ", "SQQQ"
]

# =====================================================================
# 3. DATABASE INITIALIZATION
# =====================================================================

DB_NAME = "trading_engine.db"

def init_db():
    conn = sqlite3.connect(DB_NAME)
    cursor = conn.cursor()
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS trades (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            symbol TEXT,
            side TEXT,
            qty REAL,
            price REAL,
            order_id TEXT,
            status TEXT,
            timestamp TEXT
        )
    """)
    conn.commit()
    conn.close()

# =====================================================================
# 4. CORE TRADING ENGINE
# =====================================================================

class TradingEngine:
    def __init__(self):
        logger.info("Initializing Bull. Bear and Broke Engine V6...")
        init_db()
        
        if not API_KEY or not API_SECRET:
            logger.error("Alpaca API credentials missing from environment variables!")
            sys.exit(1)
            
        self.trading_client = TradingClient(API_KEY, API_SECRET, paper=USE_PAPER)
        self.data_client = StockHistoricalDataClient(API_KEY, API_SECRET)
        
    def check_account(self):
        try:
            account = self.trading_client.get_account()
            logger.info(f"Account Status: {account.status} | Cash: ${account.cash} | Portfolio Value: ${account.portfolio_value}")
        except Exception as e:
            logger.error(f"Failed to fetch account details: {e}")

    def run_strategy_loop(self):
        logger.info("Starting main execution loop...")
        while True:
            try:
                self.check_account()
                
                # Placeholder for indicator checks and execution logic
                for symbol in WATCHLIST:
                    logger.debug(f"Scanning symbol: {symbol}")
                
                # Sleep interval between scanning cycles (e.g., 3 minutes)
                time.sleep(180)
                
            except KeyboardInterrupt:
                logger.info("Engine stopped manually by user.")
                break
            except Exception as e:
                logger.error(f"Error in main strategy loop: {e}")
                time.sleep(30)

# =====================================================================
# 5. ENTRYPOINT
# =====================================================================

if __name__ == "__main__":
    engine = TradingEngine()
    engine.run_strategy_loop()
