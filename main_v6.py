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
