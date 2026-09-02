import os
import pandas as pd
import numpy as np
import ta
import yfinance as yf
from typing import List, Dict, Any
from supabase import create_client, Client

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

# [FIX] Realistic-ish cost assumptions. Zero was flattering every result.
SLIPPAGE_PCT = 0.0005     # 5 bps in each direction (entry + exit)
COMMISSION_PER_TRADE = 0.0  # set >0 if your broker charges per-trade fees


def run_backtest(symbol: str, period: str = "60d", interval: str = "15m") -> Dict[str, Any]:
    df = yf.download(symbol, period=period, interval=interval, progress=False)
    if df.empty:
        return {
            "symbol": symbol, "total_trades": 0, "win_rate": 0.0, "total_pnl": 0.0,
            "expectancy": 0.0, "profit_factor": 0.0, "ambiguous_bars": 0,
        }

    if isinstance(df.columns, pd.MultiIndex):
        df = df.xs(symbol, level=1, axis=1)

    # 1. Technical Indicators
    df['RSI'] = ta.momentum.rsi(df['Close'], window=14)

    df['VOL_SMA20'] = df['Volume'].shift(1).rolling(20).mean()
    df['VOL_SPIKE'] = df['Volume'] > (1.2 * df['VOL_SMA20'])

    # 4H Trend Simulation (Resampled from intraday data)
    # [FIX] origin="start" + drop the still-forming last 4H bar, matching
    # what the live engine actually does in fetch_4h_bars_cached(). Without
    # this, the backtest's 4H trend at any given moment could reflect an
    # incomplete bar the live engine would never have used yet — a subtle
    # backtest/live mismatch, not outright lookahead, but still worth fixing
    # so the backtest is honest about what the live bot would have seen.
    df_4h = df.resample("4h", origin="start").agg({'Close': 'last'}).dropna()
    if len(df_4h) > 1:
        df_4h = df_4h.iloc[:-1]
    df_4h['EMA20'] = ta.trend.ema_indicator(df_4h['Close'], window=20)
    df_4h['EMA50'] = ta.trend.ema_indicator(df_4h['Close'], window=50)
    df_4h['TREND_4H'] = np.where(df_4h['EMA20'] > df_4h['EMA50'], "BULLISH", "BEARISH")

    df['TREND_4H'] = df_4h['TREND_4H'].reindex(df.index, method='ffill')

    # 2. Execution Simulation Loop
    trades = []
    ambiguous_bars = 0
    in_trade = False
    entry_price, tp_price, sl_price = 0.0, 0.0, 0.0
    allocated_cash = 0.0

    for i in range(50, len(df)):
        row = df.iloc[i]
        price = float(row['Close'])

        if not in_trade:
            if row['RSI'] < 30 and row['VOL_SPIKE'] and row['TREND_4H'] == "BULLISH":
                qty = int(DEFAULT_DOLLAR_ALLOCATION // price)
                allocated_dollars = qty * price
                min_required_dollars = DEFAULT_DOLLAR_ALLOCATION * RISK_PARAMS["MIN_ALLOCATION_PCT"]

                if qty >= 1 and allocated_dollars >= min_required_dollars:
                    in_trade = True
                    # [FIX] apply entry slippage — you very rarely get filled
                    # at the exact signal-bar close in live trading
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
                # [FIX] This was the main bug: previously always resolved as
                # a WIN whenever both levels were touched in the same bar,
                # which is impossible to determine from OHLC alone and
                # systematically inflated the win rate. Standard conservative
                # convention: assume the stop-loss was hit first.
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

    # 3. Aggregated Performance Metrics
    total_trades = len(trades)
    wins = [t for t in trades if t['outcome'] == 'WIN']
    losses = [t for t in trades if t['outcome'] == 'LOSS']

    win_rate = (len(wins) / total_trades * 100) if total_trades > 0 else 0.0
    total_pnl = sum(t['pnl'] for t in trades)

    # [FIX] Added: win rate alone is a misleading optimization target when
    # win/loss sizes are asymmetric. Expectancy (avg $ per trade) and profit
    # factor (gross win $ / gross loss $) tell you whether the strategy is
    # actually worth trading.
    expectancy = (total_pnl / total_trades) if total_trades > 0 else 0.0
    gross_win = sum(t['pnl'] for t in wins)
    gross_loss = abs(sum(t['pnl'] for t in losses))
    profit_factor = (gross_win / gross_loss) if gross_loss > 0 else (float('inf') if gross_win > 0 else 0.0)

    # [FIX] float('inf') isn't valid JSON — it was silently breaking the
    # Supabase insert at the very end of the run. Store it as None for
    # JSON/DB purposes; the console print still shows "inf" separately.
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
    print("  (same-bar TP/SL ambiguity now resolved conservatively)")
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
        print(f"⚠️  Sample size is small ({total_all_trades} trades). Treat these win-rate numbers as directional, not reliable, until you have a much larger sample (e.g. longer history at 1h/4h resolution).")
    print("=========================================================")

    # Push metrics to Supabase
    if supabase:
        try:
            payload = {
                "strategy_name": "RSI_VOL_4H_BULL",
                "total_trades": total_all_trades,
                "avg_win_rate": round(avg_win_rate, 2),
                "total_pnl": round(total_all_pnl, 2),
                "ticker_summary": summary
            }
            supabase.table("backtest_runs").insert(payload).execute()
            print("🚀 Successfully saved backtest results to Supabase.")
        except Exception as e:
            print(f"❌ Failed to push results to Supabase: {e}")
