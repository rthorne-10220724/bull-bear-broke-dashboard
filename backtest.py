import pandas as pd
import numpy as np
import ta
import yfinance as yf
from typing import List, Dict, Any

# =====================================================================
# BACKTEST CONFIGURATION (Matching Engine Settings)
# =====================================================================
TICKERS = ["SPY", "QQQ", "AAPL", "NVDA", "AMD", "MSFT", "TSLA", "MSTR", "COIN", "MARA", "RIOT"]
DEFAULT_DOLLAR_ALLOCATION = 1000.0
RISK_PARAMS = {
    "MAX_POSITION_LOSS_PCT": 0.03,  # Stop-loss (-3%)
    "TAKE_PROFIT_PCT": 0.06,        # Take-profit (+6%)
    "MIN_ALLOCATION_PCT": 0.80      # Require at least 80% deployment ($800 min)
}

def run_backtest(symbol: str, period: str = "60d", interval: str = "15m") -> Dict[str, Any]:
    # Fetch historical intraday bars via yfinance
    df = yf.download(symbol, period=period, interval=interval, progress=False)
    if df.empty:
        return {"symbol": symbol, "total_trades": 0, "win_rate": 0.0, "total_pnl": 0.0}
    
    # Flatten MultiIndex columns if present
    if isinstance(df.columns, pd.MultiIndex):
        df = df.xs(symbol, level=1, axis=1)

    # 1. Technical Indicators
    df['RSI'] = ta.momentum.rsi(df['Close'], window=14)
    
    # Volume Spike Baseline: Shift by 1 period to prevent current bar self-dilution
    df['VOL_SMA20'] = df['Volume'].shift(1).rolling(20).mean()
    df['VOL_SPIKE'] = df['Volume'] > (1.2 * df['VOL_SMA20'])

    # 4H Trend Simulation (Resampled from intraday data)
    df_4h = df.resample("4h").agg({'Close': 'last'}).dropna()
    df_4h['EMA20'] = ta.trend.ema_indicator(df_4h['Close'], window=20)
    df_4h['EMA50'] = ta.trend.ema_indicator(df_4h['Close'], window=50)
    df_4h['TREND_4H'] = np.where(df_4h['EMA20'] > df_4h['EMA50'], "BULLISH", "BEARISH")

    # Forward-fill 4H trend to match intraday bar timestamps
    df['TREND_4H'] = df_4h['TREND_4H'].reindex(df.index, method='ffill')

    # 2. Execution Simulation Loop
    trades = []
    in_trade = False
    entry_price, tp_price, sl_price = 0.0, 0.0, 0.0
    allocated_cash = 0.0

    for i in range(50, len(df)):
        row = df.iloc[i]
        price = float(row['Close'])

        if not in_trade:
            # Entry Signal: RSI < 30 + Volume Spike + 4H Bullish Trend
            if row['RSI'] < 30 and row['VOL_SPIKE'] and row['TREND_4H'] == "BULLISH":
                # Sizing & Allocation Guard
                qty = int(DEFAULT_DOLLAR_ALLOCATION // price)
                allocated_dollars = qty * price
                min_required_dollars = DEFAULT_DOLLAR_ALLOCATION * RISK_PARAMS["MIN_ALLOCATION_PCT"]

                if qty >= 1 and allocated_dollars >= min_required_dollars:
                    in_trade = True
                    entry_price = price
                    allocated_cash = allocated_dollars
                    tp_price = entry_price * (1 + RISK_PARAMS["TAKE_PROFIT_PCT"])
                    sl_price = entry_price * (1 - RISK_PARAMS["MAX_POSITION_LOSS_PCT"])
        else:
            high = float(row['High'])
            low = float(row['Low'])

            # Bracket Order Simulation
            if high >= tp_price:
                pnl = allocated_cash * RISK_PARAMS["TAKE_PROFIT_PCT"]
                trades.append({'outcome': 'WIN', 'pnl': pnl, 'return_pct': RISK_PARAMS["TAKE_PROFIT_PCT"]})
                in_trade = False
            elif low <= sl_price:
                pnl = -allocated_cash * RISK_PARAMS["MAX_POSITION_LOSS_PCT"]
                trades.append({'outcome': 'LOSS', 'pnl': pnl, 'return_pct': -RISK_PARAMS["MAX_POSITION_LOSS_PCT"]})
                in_trade = False

    # 3. Aggregated Performance Metrics
    total_trades = len(trades)
    wins = [t for t in trades if t['outcome'] == 'WIN']
    win_rate = (len(wins) / total_trades * 100) if total_trades > 0 else 0.0
    total_pnl = sum(t['pnl'] for t in trades)

    return {
        "symbol": symbol,
        "total_trades": total_trades,
        "win_rate": win_rate,
        "total_pnl": total_pnl
    }

if __name__ == "__main__":
    print("=========================================================")
    print("  STRATEGY BACKTEST: RSI < 30 + VOL SPIKE + 4H BULL TREND")
    print("=========================================================")
    
    summary = []
    for ticker in TICKERS:
        res = run_backtest(ticker, period="60d", interval="15m")
        summary.append(res)
        print(f"[{res['symbol']:<5}] Trades: {res['total_trades']:<3} | Win Rate: {res['win_rate']:>5.1f}% | Net P&L: ${res['total_pnl']:>8.2f}")

    total_all_pnl = sum(s['total_pnl'] for s in summary)
    total_all_trades = sum(s['total_trades'] for s in summary)
    avg_win_rate = np.mean([s['win_rate'] for s in summary if s['total_trades'] > 0]) if total_all_trades > 0 else 0.0

    print("---------------------------------------------------------")
    print(f"OVERALL SUMMARY: {total_all_trades} Total Trades | Avg Win Rate: {avg_win_rate:.1f}% | Net Strategy P&L: ${total_all_pnl:.2f}")
    print("=========================================================")
