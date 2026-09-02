import os
import math
import numpy as np
import pandas as pd
import yfinance as yf

from typing import Dict, Any

from strategy import (
    calculate_indicators,
    check_signal,
    ATR_STOP_MULTIPLIER,
    ATR_TARGET_MULTIPLIER,
)


# ============================================================
# CONFIG
# ============================================================

TICKERS = [
    "SPY",
    "QQQ",
    "AAPL",
    "NVDA",
    "AMD",
    "MSFT",
    "TSLA",
    "MSTR",
    "COIN",
    "MARA",
    "RIOT",
]

INITIAL_CAPITAL = 10_000.0
RISK_PER_TRADE = 0.01
MAX_ALLOCATION = 0.15

SLIPPAGE_PCT = 0.0005
COMMISSION_PER_TRADE = 0.0

PERIOD = "60d"
INTERVAL = "15m"


# ============================================================
# HELPERS
# ============================================================

def safe_float(value):
    try:
        value = float(value)
        return value if np.isfinite(value) else None
    except Exception:
        return None


def calculate_position_size(
    equity: float,
    entry: float,
    stop: float,
) -> int:

    if equity <= 0 or entry <= 0 or stop >= entry:
        return 0

    risk_dollars = equity * RISK_PER_TRADE
    risk_per_share = entry - stop

    if risk_per_share <= 0:
        return 0

    qty_by_risk = math.floor(
        risk_dollars / risk_per_share
    )

    max_dollars = equity * MAX_ALLOCATION

    qty_by_cap = math.floor(
        max_dollars / entry
    )

    return max(
        0,
        min(qty_by_risk, qty_by_cap)
    )


# ============================================================
# BACKTEST
# ============================================================

def run_backtest(
    symbol: str,
    period: str = PERIOD,
    interval: str = INTERVAL,
) -> Dict[str, Any]:

    raw = yf.download(
        symbol,
        period=period,
        interval=interval,
        progress=False,
        auto_adjust=False,
    )

    if raw.empty:
        return {
            "symbol": symbol,
            "trades": 0,
            "wins": 0,
            "losses": 0,
            "win_rate": 0.0,
            "pnl": 0.0,
            "expectancy": 0.0,
            "profit_factor": 0.0,
            "avg_r": 0.0,
            "max_drawdown": 0.0,
            "ambiguous": 0,
            "forced_exits": 0,
        }

    df = calculate_indicators(
        raw,
        symbol=symbol,
    )

    equity = INITIAL_CAPITAL
    peak_equity = equity

    trades = []

    in_trade = False

    entry_price = 0.0
    stop_price = 0.0
    target_price = 0.0
    qty = 0

    entry_risk_dollars = 0.0

    ambiguous_bars = 0
    forced_exits = 0

    equity_curve = []

    # Start sufficiently late for indicators + 4H EMA50.
    for i in range(100, len(df)):

        row = df.iloc[i]

        close = safe_float(row["Close"])
        high = safe_float(row["High"])
        low = safe_float(row["Low"])

        if close is None or high is None or low is None:
            continue

        # ----------------------------------------------------
        # ENTRY
        # ----------------------------------------------------

        if not in_trade:

            if not check_signal(row):
                equity_curve.append(equity)
                continue

            atr = safe_float(row["ATR"])

            if atr is None or atr <= 0:
                continue

            raw_entry = close

            # Realistic marketable entry slippage.
            entry_price = raw_entry * (
                1 + SLIPPAGE_PCT
            )

            stop_distance = (
                ATR_STOP_MULTIPLIER * atr
            )

            target_distance = (
                ATR_TARGET_MULTIPLIER * atr
            )

            stop_price = (
                entry_price - stop_distance
            )

            target_price = (
                entry_price + target_distance
            )

            qty = calculate_position_size(
                equity=equity,
                entry=entry_price,
                stop=stop_price,
            )

            if qty <= 0:
                continue

            entry_risk_dollars = (
                qty * stop_distance
            )

            in_trade = True

            equity_curve.append(equity)

            continue

        # ----------------------------------------------------
        # EXIT
        # ----------------------------------------------------

        hit_target = high >= target_price
        hit_stop = low <= stop_price

        # We DO NOT know whether TP or SL happened first
        # inside a 15m OHLC bar.
        #
        # Therefore ambiguity is resolved against us.
        if hit_target and hit_stop:

            ambiguous_bars += 1

            exit_price = (
                stop_price *
                (1 - SLIPPAGE_PCT)
            )

            pnl = (
                qty *
                (exit_price - entry_price)
                - COMMISSION_PER_TRADE
            )

            outcome = "LOSS"

        elif hit_target:

            exit_price = (
                target_price *
                (1 - SLIPPAGE_PCT)
            )

            pnl = (
                qty *
                (exit_price - entry_price)
                - COMMISSION_PER_TRADE
            )

            outcome = "WIN"

        elif hit_stop:

            exit_price = (
                stop_price *
                (1 - SLIPPAGE_PCT)
            )

            pnl = (
                qty *
                (exit_price - entry_price)
                - COMMISSION_PER_TRADE
            )

            outcome = "LOSS"

        else:
            equity_curve.append(equity)
            continue

        r_multiple = (
            pnl / entry_risk_dollars
            if entry_risk_dollars > 0
            else 0.0
        )

        equity += pnl

        trades.append({
            "outcome": outcome,
            "pnl": pnl,
            "r": r_multiple,
        })

        peak_equity = max(
            peak_equity,
            equity,
        )

        in_trade = False
        qty = 0
        entry_price = 0.0
        stop_price = 0.0
        target_price = 0.0
        entry_risk_dollars = 0.0

        equity_curve.append(equity)

    # --------------------------------------------------------
    # FORCE OPEN TRADE CLOSED AT END OF DATA
    # --------------------------------------------------------

    if in_trade and qty > 0:

        final_close = safe_float(
            df.iloc[-1]["Close"]
        )

        if final_close is not None:

            exit_price = (
                final_close *
                (1 - SLIPPAGE_PCT)
            )

            pnl = (
                qty *
                (exit_price - entry_price)
                - COMMISSION_PER_TRADE
            )

            r_multiple = (
                pnl / entry_risk_dollars
                if entry_risk_dollars > 0
                else 0.0
            )

            trades.append({
                "outcome": "FORCED_EXIT",
                "pnl": pnl,
                "r": r_multiple,
            })

            equity += pnl
            forced_exits += 1

    # --------------------------------------------------------
    # STATISTICS
    # --------------------------------------------------------

    total_trades = len(trades)

    wins = [
        t for t in trades
        if t["outcome"] == "WIN"
    ]

    losses = [
        t for t in trades
        if t["outcome"] == "LOSS"
        or t["outcome"] == "FORCED_EXIT"
        and t["pnl"] < 0
    ]

    win_count = len(wins)

    win_rate = (
        win_count / total_trades * 100
        if total_trades
        else 0.0
    )

    total_pnl = sum(
        t["pnl"] for t in trades
    )

    expectancy = (
        total_pnl / total_trades
        if total_trades
        else 0.0
    )

    gross_profit = sum(
        t["pnl"]
        for t in trades
        if t["pnl"] > 0
    )

    gross_loss = abs(sum(
        t["pnl"]
        for t in trades
        if t["pnl"] < 0
    ))

    if gross_loss > 0:
        profit_factor = (
            gross_profit / gross_loss
        )
    elif gross_profit > 0:
        profit_factor = float("inf")
    else:
        profit_factor = 0.0

    avg_r = (
        np.mean([t["r"] for t in trades])
        if trades
        else 0.0
    )

    # --------------------------------------------------------
    # MAX DRAWDOWN
    # --------------------------------------------------------

    if equity_curve:

        curve = pd.Series(
            equity_curve,
            dtype=float,
        )

        rolling_peak = curve.cummax()

        drawdown = (
            curve - rolling_peak
        ) / rolling_peak

        max_drawdown = (
            abs(drawdown.min()) * 100
        )

    else:
        max_drawdown = 0.0

    return {
        "symbol": symbol,
        "trades": total_trades,
        "wins": win_count,
        "losses": total_trades - win_count,
        "win_rate": win_rate,
        "pnl": total_pnl,
        "expectancy": expectancy,
        "profit_factor": profit_factor,
        "avg_r": avg_r,
        "max_drawdown": max_drawdown,
        "ambiguous": ambiguous_bars,
        "forced_exits": forced_exits,
    }


# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":

    print("=" * 78)
    print("10/10 RESEARCH ENGINE BACKTEST")
    print(
        "4H TREND + 15M PULLBACK + RSI RECOVERY + "
        "RVOL + VOLATILITY"
    )
    print("=" * 78)

    results = []

    for ticker in TICKERS:

        try:
            result = run_backtest(ticker)

        except Exception as e:

            print(
                f"[{ticker:<5}] ERROR: {e}"
            )

            continue

        results.append(result)

        pf = result["profit_factor"]

        if math.isinf(pf):
            pf_display = "inf"
        else:
            pf_display = f"{pf:.2f}"

        print(
            f"[{ticker:<5}] "
            f"Trades={result['trades']:<3} | "
            f"W={result['wins']:<3} | "
            f"L={result['losses']:<3} | "
            f"Win={result['win_rate']:>5.1f}% | "
            f"P&L=${result['pnl']:>8.2f} | "
            f"Exp=${result['expectancy']:>6.2f} | "
            f"PF={pf_display:>5} | "
            f"AvgR={result['avg_r']:>6.3f} | "
            f"DD={result['max_drawdown']:>5.2f}%"
        )

    print("-" * 78)

    total_trades = sum(
        r["trades"] for r in results
    )

    total_wins = sum(
        r["wins"] for r in results
    )

    total_losses = (
        total_trades - total_wins
    )

    total_pnl = sum(
        r["pnl"] for r in results
    )

    avg_r = (
        np.mean([
            r["avg_r"]
            for r in results
            if r["trades"] > 0
        ])
        if any(r["trades"] > 0 for r in results)
        else 0.0
    )

    overall_win_rate = (
        total_wins / total_trades * 100
        if total_trades
        else 0.0
    )

    expectancy = (
        total_pnl / total_trades
        if total_trades
        else 0.0
    )

    gross_profit = sum(
        max(0, r["pnl"])
        for r in results
    )

    gross_loss = abs(sum(
        min(0, r["pnl"])
        for r in results
    ))

    overall_pf = (
        gross_profit / gross_loss
        if gross_loss > 0
        else float("inf")
        if gross_profit > 0
        else 0.0
    )

    ambiguous = sum(
        r["ambiguous"] for r in results
    )

    forced = sum(
        r["forced_exits"] for r in results
    )

    print(
        f"TOTAL TRADES: {total_trades}"
    )

    print(
        f"WINS / LOSSES: "
        f"{total_wins} / {total_losses}"
    )

    print(
        f"OVERALL WIN RATE: "
        f"{overall_win_rate:.2f}%"
    )

    print(
        f"NET P&L: "
        f"${total_pnl:.2f}"
    )

    print(
        f"EXPECTANCY / TRADE: "
        f"${expectancy:.2f}"
    )

    print(
        f"PROFIT FACTOR: "
        f"{overall_pf:.3f}"
    )

    print(
        f"AVERAGE R: "
        f"{avg_r:.3f}"
    )

    print(
        f"AMBIGUOUS TP/SL BARS: "
        f"{ambiguous}"
    )

    print(
        f"FORCED END-OF-DATA EXITS: "
        f"{forced}"
    )

    print("=" * 78)

    if total_trades < 100:
        print(
            f"⚠️ Only {total_trades} trades. "
            "Do not judge strategy quality yet."
        )
    else:
        print(
            "✅ 100+ trades reached. "
            "Run a separate out-of-sample period."
        )
