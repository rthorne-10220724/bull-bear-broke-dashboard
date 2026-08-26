import os
import time
import sqlite3
import smtplib
import datetime
import requests
import threading
from pathlib import Path
from collections import deque
from zoneinfo import ZoneInfo
from email.message import EmailMessage
import numpy as np
import pandas as pd
import yfinance as yf
from dotenv import load_dotenv
from typing import Literal, Optional
from pydantic import BaseModel, Field
from openai import OpenAI
from crewai import Agent, Crew, Process, Task, LLM

# Alpaca SDK Imports
from alpaca.trading.client import TradingClient
from alpaca.trading.requests import MarketOrderRequest, StopLossRequest, TakeProfitRequest
from alpaca.trading.enums import OrderSide, TimeInForce, OrderClass
from alpaca.data.historical import StockHistoricalDataClient
from alpaca.data.requests import StockBarsRequest
from alpaca.data.timeframe import TimeFrame

# Disable OpenTelemetry background warning logs
os.environ["OTEL_SDK_DISABLED"] = "true"

# Safe environment variable loader for Render Linux vs Local Windows
if os.path.exists(r"C:\MarketAgents\.env"):
    load_dotenv(dotenv_path=Path(r"C:\MarketAgents\.env"))
else:
    load_dotenv()

# =====================================================================
# 1. GLOBAL CONFIGURATION & ENVIRONMENT SETUP
# =====================================================================
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
ALPACA_API_KEY = os.getenv("ALPACA_API_KEY")
ALPACA_SECRET_KEY = os.getenv("ALPACA_SECRET_KEY")
PAPER_TRADING = True  # Set to False only when going live

if not all([OPENAI_API_KEY, ALPACA_API_KEY, ALPACA_SECRET_KEY]):
    raise ValueError("Missing critical API credentials in environment variables.")

openai_client = OpenAI(api_key=OPENAI_API_KEY)
trading_client = TradingClient(ALPACA_API_KEY, ALPACA_SECRET_KEY, paper=PAPER_TRADING)
data_client = StockHistoricalDataClient(ALPACA_API_KEY, ALPACA_SECRET_KEY)

DB_FILE = "paper_trail_audit_v5.db"
RISK_PER_TRADE_PCT = 0.05  # Raised to 5% equity risk for stronger position sizing
LLM_COOLDOWN_SECONDS = 900  # 15-minute cooldown per ticker for LLM API calls
EASTERN_TZ = ZoneInfo("America/New_York")

# Watchlist Universe
CRYPTO_ADJACENT_TICKERS = {"MSTR", "COIN", "MARA", "RIOT", "CLSK", "BITF", "HUT"}
STANDARD_TICKERS = {"NVDA", "AMD", "TSLA", "PLTR"}
ALL_TICKERS = list(CRYPTO_ADJACENT_TICKERS.union(STANDARD_TICKERS))

SEC_CIK_MAP = {
    "STLD": "0001084712",
    "NVDA": "0001045810",
    "TSLA": "0001318605",
    "MSTR": "0001050446",
    "AMD": "0000002488",
    "PLTR": "0001321655",
    "COIN": "0001834488"
}

# Optimized Sensitivity Parameters for High Execution & Low Cost
RISK_PARAMS = {
    "STANDARD": {
        "min_rvol": 1.0,           # Baseline normal volume triggers technical check
        "sl_atr_mult": 1.5,
        "rr_target": 2.5,
        "trailing_atr_mult": 2.0,
        "allow_overnight": False
    },
    "CRYPTO_ADJACENT": {
        "min_rvol": 1.0,           # Baseline crypto volume
        "sl_atr_mult": 2.0,
        "rr_target": 3.0,
        "trailing_atr_mult": 2.5,
        "allow_overnight": True
    }
}

PROCESSED_BARS = deque(maxlen=1000)
DECISION_CACHE = {}

# =====================================================================
# 2. CREWAI AGENTS & EMAIL REPORTING SYSTEM
# =====================================================================
openai_llm = LLM(
    model="gpt-4o-mini",
    temperature=0.2,
    api_key=OPENAI_API_KEY
)

def fetch_market_data(ticker_symbol: str) -> str:
    try:
        ticker = yf.Ticker(ticker_symbol)
        info = ticker.fast_info
        return f"Ticker: {ticker_symbol} | Price: ${info.last_price:.2f} | Market Cap: ${info.market_cap:,}"
    except Exception as e:
        return f"Error fetching market data for {ticker_symbol}: {str(e)}"

def fetch_sec_executive_trades(ticker_symbol: str) -> str:
    cik = SEC_CIK_MAP.get(ticker_symbol.upper())
    if not cik:
        return f"No SEC CIK mapping configured for {ticker_symbol}."
    headers = {'User-Agent': 'MarketResearchApp admin@marketresearch.com'}
    url = f"https://data.sec.gov/submissions/CIK{cik}.json"
    try:
        res = requests.get(url, headers=headers)
        if res.status_code == 200:
            data = res.json()
            recent_filings = data['filings']['recent']
            forms = recent_filings['form'][:10]
            dates = recent_filings['filingDate'][:10]
            insider_activity = [f"{form} on {date}" for form, date in zip(forms, dates) if form == '4']
            return f"Recent Form 4 Filings ({ticker_symbol}): {', '.join(insider_activity) if insider_activity else 'No recent Form 4 filings.'}"
        return f"SEC Data stream unavailable (Status {res.status_code})."
    except Exception as e:
        return f"Could not retrieve SEC EDGAR data for {ticker_symbol}: {str(e)}"

scraper_agent = Agent(
    role="Data Scraper Specialist",
    goal="Extract quantitative market metrics, executive Form 4 filings, and crypto trends.",
    backstory="You gather raw numbers, executive share disclosures, and market trends.",
    verbose=True,
    llm=openai_llm
)

analyst_agent = Agent(
    role="Portfolio Risk Analyst",
    goal="Synthesize research into a highly structured, diverse multi-asset investment report.",
    backstory="You are an expert market strategist who specializes in tiering risk profiles across distinct price categories.",
    verbose=True,
    llm=openai_llm
)

def _run_crewai_report_task():
    print("\n[CREWAI THREAD] Running daily intelligence report generation...")
    stld_data = fetch_market_data("STLD")
    stld_sec = fetch_sec_executive_trades("STLD")
    
    task_scrape = Task(
        description=f"Gather financial context:\n1. STLD Raw Market Data: {stld_data}\n2. Executive Insider Filings: {stld_sec}\n3. Research broader stock market and cryptocurrency momentum drivers.",
        expected_output="Aggregated raw market data brief.",
        agent=scraper_agent
    )
    
    task_report = Task(
        description="Generate a comprehensive 12:00 PM daily investment intelligence report in HTML format.",
        expected_output="Final HTML-formatted 12:00 PM daily investment intelligence report.",
        agent=analyst_agent
    )
    
    crew = Crew(
        agents=[scraper_agent, analyst_agent],
        tasks=[task_scrape, task_report],
        process=Process.sequential
    )
    
    try:
        result = crew.kickoff()
        report_text = result.raw if hasattr(result, 'raw') else str(result)
        send_daily_email(report_text)
    except Exception as e:
        print(f"[REPORT ERROR] CrewAI generation failed: {e}")

def generate_and_send_crewai_report_async():
    report_thread = threading.Thread(target=_run_crewai_report_task, daemon=True)
    report_thread.start()

def send_daily_email(html_summary: str):
    sender_email = (os.getenv("SENDER_EMAIL") or "").strip()
    app_password = (os.getenv("SENDER_APP_PASSWORD") or "").strip()
    recipient = (os.getenv("RECIPIENT_EMAIL") or "").strip()
    
    if not all([sender_email, app_password, recipient]):
        print("[EMAIL ERROR] Missing email credentials.")
        return

    clean_content = str(html_summary).replace('\xa0', ' ')
    email_html = f"<html><body>{clean_content}</body></html>"
    
    msg = EmailMessage()
    msg['Subject'] = "12 PM Market Intel Report"
    msg['From'] = sender_email
    msg['To'] = recipient
    msg.set_content("Your email reader does not support HTML.", charset='utf-8')
    msg.add_alternative(email_html, subtype='html')

    try:
        with smtplib.SMTP_SSL('smtp.gmail.com', 465) as server:
            server.login(sender_email, app_password)
            server.send_message(msg)
            print("\nFormatted HTML email report successfully delivered!")
    except Exception as e:
        print(f"\n[EMAIL ERROR] Failed to send email: {e}")

# =====================================================================
# 3. TRADING ENGINE DATABASE & LOGGERS
# =====================================================================
def init_db():
    with sqlite3.connect(DB_FILE) as conn:
        cursor = conn.cursor()
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS llm_decisions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp TEXT, ticker TEXT, asset_class TEXT, action TEXT,
                bearish_thesis TEXT, confidence REAL, executed INTEGER
            )
        ''')
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS broker_executions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp TEXT, ticker TEXT, side TEXT, qty REAL, price REAL,
                stop_loss REAL, take_profit REAL, alpaca_order_id TEXT, status TEXT
            )
        ''')
        conn.commit()

init_db()

def log_decision(timestamp, ticker, asset_class, d, executed: int):
    with sqlite3.connect(DB_FILE) as conn:
        cursor = conn.cursor()
        cursor.execute('''
            INSERT INTO llm_decisions (timestamp, ticker, asset_class, action, bearish_thesis, confidence, executed)
            VALUES (?, ?, ?, ?, ?, ?, ?)
        ''', (timestamp, ticker, asset_class, d.action, d.bearish_thesis, d.confidence_score, executed))
        conn.commit()

def log_execution(timestamp, ticker, side, qty, price, stop_loss, take_profit, alpaca_order_id):
    with sqlite3.connect(DB_FILE) as conn:
        cursor = conn.cursor()
        cursor.execute('''
            INSERT INTO broker_executions (timestamp, ticker, side, qty, price, stop_loss, take_profit, alpaca_order_id, status)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        ''', (timestamp, ticker, side, qty, price, stop_loss, take_profit, alpaca_order_id, "FILLED"))
        conn.commit()

# =====================================================================
# 4. DETERMINISTIC GATEKEEPER & POSITION SIZER
# =====================================================================
class TechnicalGatekeeper:
    @staticmethod
    def get_asset_class(ticker: str) -> str:
        return "CRYPTO_ADJACENT" if ticker.upper() in CRYPTO_ADJACENT_TICKERS else "STANDARD"

    @staticmethod
    def calculate_indicators(ticker: str, df_1m: pd.DataFrame, df_4h: pd.DataFrame) -> dict:
        asset_class = TechnicalGatekeeper.get_asset_class(ticker)
        params = RISK_PARAMS[asset_class]
        
        df_4h['ema50'] = df_4h['close'].ewm(span=50, adjust=False).mean()
        macro_uptrend = bool(df_4h['close'].iloc[-1] > df_4h['ema50'].iloc[-1])
        
        v = df_1m['volume']
        tp = (df_1m['high'] + df_1m['low'] + df_1m['close']) / 3
        vwap = (tp * v).cumsum() / (v.cumsum() + 1e-9)
        current_vwap = vwap.iloc[-1]
        
        avg_volume = df_1m['volume'].tail(20).mean()
        current_volume = df_1m['volume'].iloc[-1]
        rvol = current_volume / avg_volume if avg_volume > 0 else 0.0
        
        high_low = df_1m['high'] - df_1m['low']
        high_close = np.abs(df_1m['high'] - df_1m['close'].shift())
        low_close = np.abs(df_1m['low'] - df_1m['close'].shift())
        tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
        atr = tr.rolling(14).mean().iloc[-1]
        
        current_price = float(df_1m['close'].iloc[-1])
        
        # 3-Bar Breakout Check (faster momentum detection)
        recent_high = float(df_1m['high'].iloc[-4:-1].max())
        is_breakout = current_price >= recent_high
        
        stop_loss = round(current_price - (params['sl_atr_mult'] * atr), 2)
        take_profit = round(current_price + (params['sl_atr_mult'] * atr * params['rr_target']), 2)
        
        passed_filter = bool(macro_uptrend and (current_price > current_vwap) and (rvol >= params['min_rvol']) and is_breakout)
        
        return {
            "passed": passed_filter,
            "asset_class": asset_class,
            "current_price": round(current_price, 2),
            "vwap": round(current_vwap, 2),
            "rvol": round(rvol, 2),
            "atr": round(atr, 2),
            "stop_loss": stop_loss,
            "take_profit": take_profit,
            "macro_uptrend": macro_uptrend,
            "min_rvol_required": params['min_rvol']
        }

    @staticmethod
    def calculate_position_size(entry_price: float, stop_loss_price: float) -> float:
        try:
            account = trading_client.get_account()
            equity = float(account.equity)
        except Exception as e:
            print(f"[ERROR] Could not fetch Alpaca equity: {e}")
            equity = 1000.0
            
        risk_per_share = abs(entry_price - stop_loss_price)
        if risk_per_share == 0:
            return 0.0
            
        max_capital_risk = equity * RISK_PER_TRADE_PCT
        shares = max_capital_risk / risk_per_share
        return round(shares, 4)

# =====================================================================
# 5. QUALITATIVE LLM RISK ADVERSARY
# =====================================================================
class TradeDecisionSchema(BaseModel):
    action: Literal["BUY", "HOLD"] = Field(description="BUY if momentum conviction is pristine, else HOLD")
    bearish_thesis: str = Field(description="3 concrete market structural failure risks")
    confidence_score: float = Field(description="Score between 0.0 and 1.0 evaluating thesis strength")

def run_llm_strategy_agent(ticker: str, metrics: dict) -> Optional[TradeDecisionSchema]:
    now = time.time()
    if ticker in DECISION_CACHE:
        cached_decision, cached_time = DECISION_CACHE[ticker]
        if now - cached_time < LLM_COOLDOWN_SECONDS:
            return cached_decision

    prompt = (
        f"Ticker: {ticker} ({metrics['asset_class']})\n"
        f"Price: ${metrics['current_price']} | SL: ${metrics['stop_loss']} | TP: ${metrics['take_profit']}\n"
        f"RVOL: {metrics['rvol']}x (Req: {metrics['min_rvol_required']}x)\n"
        f"4H Trend: {'Strong Uptrend' if metrics['macro_uptrend'] else 'Downtrend'}\n"
        "Evaluate breakout failure probability. Output action (BUY/HOLD), 3 failure reasons, and confidence score."
    )

    try:
        completion = openai_client.beta.chat.completions.parse(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": "You are a disciplined risk control agent. Veto unsafe entries."},
                {"role": "user", "content": prompt}
            ],
            response_format=TradeDecisionSchema,
            temperature=0.0
        )
        decision = completion.choices[0].message.parsed
        DECISION_CACHE[ticker] = (decision, now)
        return decision
    except Exception as e:
        print(f"[ERROR] LLM Query Failed for {ticker}: {e}")
        return None

# =====================================================================
# 6. EXECUTION PIPELINE & EOD LIQUIDATION
# =====================================================================
def execute_alpaca_order(ticker: str, qty: float, price: float, stop_loss: float, take_profit: float) -> Optional[str]:
    try:
        order_data = MarketOrderRequest(
            symbol=ticker,
            qty=qty,
            side=OrderSide.BUY,
            time_in_force=TimeInForce.GTC,
            order_class=OrderClass.BRACKET,
            take_profit=TakeProfitRequest(limit_price=take_profit),
            stop_loss=StopLossRequest(stop_price=stop_loss)
        )
        order = trading_client.submit_order(order_data)
        print(f"[ALPACA BRACKET ORDER SUBMITTED] {qty} shares of {ticker} | SL: ${stop_loss} | TP: ${take_profit} | Order ID: {order.id}")
        return str(order.id)
    except Exception as e:
        print(f"[ERROR] Alpaca Order Execution Failed: {e}")
        return None

def process_pipeline(ticker: str, df_1m: pd.DataFrame, df_4h: pd.DataFrame):
    current_time_est = datetime.datetime.now(EASTERN_TZ).time()
    if current_time_est >= datetime.time(15, 45):
        return

    bar_time = df_1m.index[-1]
    bar_id = f"{ticker}_{bar_time}"
    if bar_id in PROCESSED_BARS:
        return

    metrics = TechnicalGatekeeper.calculate_indicators(ticker, df_1m, df_4h)
    if not metrics['passed']:
        return

    PROCESSED_BARS.append(bar_id)
    timestamp = datetime.datetime.now(datetime.timezone.utc).isoformat()
    print(f"[{ticker}] {metrics['asset_class']} Technical Breakout Triggered! Running LLM Adversary...")

    decision = run_llm_strategy_agent(ticker, metrics)
    if not decision:
        return

    executed = 0
    # Lowered confidence threshold to 0.50 for active paper trading
    if decision.action == "BUY" and decision.confidence_score >= 0.50:
        qty = TechnicalGatekeeper.calculate_position_size(metrics['current_price'], metrics['stop_loss'])
        if qty > 0:
            order_id = execute_alpaca_order(ticker, qty, metrics['current_price'], metrics['stop_loss'], metrics['take_profit'])
            if order_id:
                executed = 1
                log_execution(timestamp, ticker, "BUY", qty, metrics['current_price'], metrics['stop_loss'], metrics['take_profit'], order_id)

    log_decision(timestamp, ticker, metrics['asset_class'], decision, executed)

def fetch_market_bars(ticker: str):
    now = datetime.datetime.now(datetime.timezone.utc)
    start_1m = now - datetime.timedelta(days=2)
    start_4h = now - datetime.timedelta(days=30)

    request_1m = StockBarsRequest(symbol_or_symbols=ticker, timeframe=TimeFrame.Minute, start=start_1m)
    request_4h = StockBarsRequest(symbol_or_symbols=ticker, timeframe=TimeFrame.Hour, start=start_4h)

    bars_1m = data_client.get_stock_bars(request_1m).df
    bars_4h = data_client.get_stock_bars(request_4h).df

    if bars_1m.empty or bars_4h.empty:
        return None, None

    if isinstance(bars_1m.index, pd.MultiIndex):
        bars_1m = bars_1m.xs(ticker)
    if isinstance(bars_4h.index, pd.MultiIndex):
        bars_4h = bars_4h.xs(ticker)

    df_4h = bars_4h.resample('4h').agg({
        'open': 'first', 'high': 'max', 'low': 'min', 'close': 'last', 'volume': 'sum'
    }).dropna()

    return bars_1m, df_4h

# =====================================================================
# 7. UNIFIED MAIN EXECUTION LOOP
# =====================================================================
def run_trading_cycle():
    now_est = datetime.datetime.now(EASTERN_TZ)
    print(f"\n[{now_est.strftime('%Y-%m-%d %H:%M:%S')}] Starting Market Execution Cycle...")

    # Only fire CrewAI email report once per day at 12:00 PM EST (Saves API tokens)
    if now_est.hour == 12 and now_est.minute < 3:
        try:
            generate_and_send_crewai_report_async()
        except Exception as e:
            print(f"[ERROR] CrewAI report task failed: {e}")

    # Process market universe
    for ticker in ALL_TICKERS:
        try:
            bars_1m, df_4h = fetch_market_bars(ticker)
            if bars_1m is not None and df_4h is not None:
                process_pipeline(ticker, bars_1m, df_4h)
        except Exception as e:
            print(f"[ERROR] Failed processing ticker {ticker}: {e}")

    print(f"[{now_est.strftime('%Y-%m-%d %H:%M:%S')}] Cycle complete.")

if __name__ == "__main__":
    print("[ENGINE V5] Continuous Local Trading Loop Active.")
    
    while True:
        try:
            run_trading_cycle()
        except Exception as e:
            print(f"[ERROR] Unexpected cycle failure: {e}")
            
        print("Sleeping 3 minutes until next market scan...")
        time.sleep(180)
