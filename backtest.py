import os
import pandas as pd
import numpy as np
import yfinance as yf
from typing import List, Dict, Any
from supabase import create_client, Client
from strategy import calculate_indicators, check_signal  # <-- Single source of truth

# =====================================================================
# SUPABASE INITIALIZATION
# =====================================================================
SUPABASE_URL = os.environ.get("SUPABASE_URL")
SUPABASE_KEY = os.environ.get("SUPABASE_KEY")

supabase: Client = None
if SUPABASE_URL and SUPABASE_KEY:
    try:
        supabase = create_client(SUPABASE_URL, SUPABASE_KEY)
        print("Connected to Supabase successfully.")
    except Exception as e:
        print(f"⚠️ Failed to initialize Supabase client: {e}")
else:
    print("⚠️ Supabase environment variables missing. Results will only print to console.")

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

SLIPPAGE_PCT = 0.0005     # 5 bps in each direction (entry + exit)
COMMISSION_PER_TRADE = 0.0  

def run_backtest(symbol: str, period: str = "60d", interval: str = "15m") -> Dict[str, Any]:
    df = yf.download(symbol, period=period, interval=interval, progress=False)
    if df.empty:
        return {
            "symbol": symbol, "total_trades": 0, "win_rate": 0.0, "total_pnl": 0.0,
            "expectancy": 0.0, "profit_factor": 0.0, "ambiguous_bars": 0,
        }

    # Delegate indicator creation to the shared strategy module
    df = calculate_indicators(df, symbol=symbol)

    trades = []
    ambiguous_bars = 0
    in_trade = False
    entry_price, tp_price, sl_price = 0.0, 0.0, 0.0
    allocated_cash = 0.0

    for i in range(50, len(df)):
        row = df.iloc[i]
        price = float(row['Close'])

        if not in_trade:
            # Delegate signal evaluation to the shared strategy module
            if check_signal(row):
                qty = int(DEFAULT_DOLLAR_ALLOCATION // price)
                allocated_dollars = qty * price
                min_required_dollars = DEFAULT_DOLLAR_ALLOCATION * RISK_PARAMS["MIN_ALLOCATION_PCT"]

                if qty >= 1 and allocated_dollars >= min_required_dollars:
                    in_trade = True
                    entry_price = price * (1 + SLIPPAGE_PCT)
                    allocated_cash = allocated_dollars
                    tp_price = entry_price * (1 + RISK_PARAMS["TAKE_PROFIT_PCT"])
                    sl_price = entry_price * (1 - RISK_PARAMS["MAX_POSITION_LOSS_PCT"])
        else:
            high = float(row['High'])
            low = float(row['Low'])

            hit_tp = high >= tp_price
            hit_sl = low <= sl_price

            if hit_tp and hit_sl:
                ambiguous_bars += 1
                exit_price = sl_price * (1 - SLIPPAGE_PCT)
                pnl = allocated_cash * ((exit_price - entry_price) / entry_price) - COMMISSION_PER_TRADE
                trades.append({'outcome': 'LOSS', 'pnl': pnl, 'return_pct': (exit_price - entry_price) / entry_price})
                in_trade = False
            elif hit_tp:
                exit_price = tp_price * (1 - SLIPPAGE_PCT)
                pnl = allocated_cash * ((exit_price - entry_price) / entry_price) - COMMISSION_PER_TRADE
                trades.append({'outcome': 'WIN', 'pnl': pnl, 'return_pct': (exit_price - entry_price) / entry_price})
                in_trade = False
            elif hit_sl:
                exit_price = sl_price * (1 - SLIPPAGE_PCT)
                pnl = allocated_cash * ((exit_price - entry_price) / entry_price) - COMMISSION_PER_TRADE
                trades.append({'outcome': 'LOSS', 'pnl': pnl, 'return_pct': (exit_price - entry_price) / entry_price})
                in_trade = False

    total_trades = len(trades)
    wins = [t for t in trades if t['outcome'] == 'WIN']
    losses = [t for t in trades if t['outcome'] == 'LOSS']

    win_rate = (len(wins) / total_trades * 100) if total_trades > 0 else 0.0
    total_pnl = sum(t['pnl'] for t in trades)
    expectancy = (total_pnl / total_trades) if total_trades > 0 else 0.0
    gross_win = sum(t['pnl'] for t in wins)
    gross_loss = abs(sum(t['pnl'] for t in losses))
    profit_factor = (gross_win / gross_loss) if gross_loss > 0 else (float('inf') if gross_win > 0 else 0.0)

    profit_factor_display = "inf" if profit_factor == float('inf') else round(profit_factor, 2)
    profit_factor_json = None if profit_factor == float('inf') else round(profit_factor, 2)

    return {
        "symbol": symbol,
        "total_trades": total_trades,
        "win_rate": round(win_rate, 2),
        "total_pnl": round(total_pnl, 2),
        "expectancy": round(expectancy, 2),
        "profit_factor": profit_factor_json,
        "profit_factor_display": profit_factor_display,
        "ambiguous_bars": ambiguous_bars,
    }


if __name__ == "__main__":
    print("=========================================================")
    print("  STRATEGY BACKTEST: RSI < 30 + VOL SPIKE + 4H BULL TREND")
    print("  (v13 Unified Engine Architecture)")
    print("=========================================================")

    summary = []
    for ticker in TICKERS:
        res = run_backtest(ticker, period="60d", interval="15m")
        summary.append(res)
        amb_note = f" (⚠ {res['ambiguous_bars']} same-bar TP/SL)" if res['ambiguous_bars'] else ""
        print(
            f"[{res['symbol']:<5}] Trades: {res['total_trades']:<3} | "
            f"Win Rate: {res['win_rate']:>5.1f}% | Net P&L: ${res['total_pnl']:>8.2f} | "
            f"Expectancy: ${res['expectancy']:>7.2f} | PF: {res['profit_factor_display']:>5}{amb_note}"
        )

    total_all_pnl = float(sum(s['total_pnl'] for s in summary))
    total_all_trades = int(sum(s['total_trades'] for s in summary))
    total_ambiguous = int(sum(s['ambiguous_bars'] for s in summary))
    avg_win_rate = float(np.mean([s['win_rate'] for s in summary if s['total_trades'] > 0])) if total_all_trades > 0 else 0.0
    avg_expectancy = (total_all_pnl / total_all_trades) if total_all_trades > 0 else 0.0

    print("---------------------------------------------------------")
    print(
        f"OVERALL SUMMARY: {total_all_trades} Total Trades | Avg Win Rate: {avg_win_rate:.1f}% | "
        f"Net Strategy P&L: ${total_all_pnl:.2f} | Avg Expectancy/Trade: ${avg_expectancy:.2f}"
    )
    if total_ambiguous:
        print(f"⚠️  {total_ambiguous} trade(s) had both TP and SL touched in the same bar — resolved as losses (conservative).")
    if total_all_trades < 100:
        print(f"⚠️  Sample size is small ({total_all_trades} trades). Treat these win-rate numbers as directional, not reliable.")
    print("=========================================================")

    if supabase:
        try:
            rows = [
                {
                    "ticker": s["symbol"],
                    "trades": s["total_trades"],
                    "win_rate": s["win_rate"],
                    "net_pnl": s["total_pnl"],
                }
                for s in summary
            ]
            supabase.table("backtest_runs").insert(rows).execute()
            print(f"🚀 Successfully saved {len(rows)} ticker rows to Supabase.")
        except Exception as e:
            print(f"❌ Failed to push results to Supabase: {e}")
