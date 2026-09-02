"""
Bull. Bear and Broke — Unified Backtester v10
"""

from __future__ import annotations

import os
from typing import Dict, Any, List

import numpy as np
import pandas as pd
import yfinance as yf

from strategy import (
    calculate_indicators,
    check_signal,
    calculate_trade_levels,
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

PERIOD = "60d"
INTERVAL = "15m"

STARTING_ALLOCATION = 1000.0

SLIPPAGE_PCT = 0.0005

COMMISSION_PER_SIDE = 0.0


# ============================================================
# DATA
# ============================================================

def load_data(
    symbol: str,
) -> pd.DataFrame:

    df = yf.download(
        symbol,
        period=PERIOD,
        interval=INTERVAL,
        auto_adjust=False,
        progress=False,
    )

    if df.empty:
        return pd.DataFrame()

    if isinstance(df.columns, pd.MultiIndex):
        try:
            df = df.xs(
                symbol,
                level=1,
                axis=1,
            )
        except Exception:
            df.columns = (
                df.columns
                .get_level_values(0)
            )

    required = [
        "Open",
        "High",
        "Low",
        "Close",
        "Volume",
    ]

    missing = [
        c
        for c in required
        if c not in df.columns
    ]

    if missing:
        raise ValueError(
            f"{symbol}: missing columns {missing}"
        )

    df = df[required].copy()

    df = df.dropna()

    if df.index.tz is None:
        df.index = df.index.tz_localize("UTC")

    return df


# ============================================================
# BACKTEST
# ============================================================

def run_backtest(
    symbol: str,
) -> Dict[str, Any]:

    df = load_data(symbol)

    if df.empty:
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
        df,
        symbol=symbol,
    )

    trades: List[Dict[str, Any]] = []

    in_trade = False

    entry = 0.0
    stop = 0.0
    target = 0.0

    allocation = 0.0
    initial_risk = 0.0

    ambiguous = 0
    forced_exits = 0

    equity_curve = [
        STARTING_ALLOCATION
    ]

    equity = STARTING_ALLOCATION

    # Start far enough into the dataset for indicators.
    for i in range(60, len(df)):

        row = df.iloc[i]

        # ====================================================
        # OPEN POSITION
        # ====================================================

        if in_trade:

            high = float(row["High"])
            low = float(row["Low"])

            hit_stop = (
                low <= stop
            )

            hit_target = (
                high >= target
            )

            # ------------------------------------------------
            # SAME BAR = CONSERVATIVE STOP
            # ------------------------------------------------

            if hit_stop and hit_target:

                ambiguous += 1

                exit_price = (
                    stop
                    * (1 - SLIPPAGE_PCT)
                )

                outcome = "LOSS"

            elif hit_stop:

                exit_price = (
                    stop
                    * (1 - SLIPPAGE_PCT)
                )

                outcome = "LOSS"

            elif hit_target:

                exit_price = (
                    target
                    * (1 - SLIPPAGE_PCT)
                )

                outcome = "WIN"

            else:
                continue

            pnl = (
                allocation
                * (
                    exit_price - entry
                )
                / entry
            )

            pnl -= (
                COMMISSION_PER_SIDE * 2
            )

            r_multiple = (
                pnl / initial_risk
                if initial_risk > 0
                else 0.0
            )

            equity += pnl

            equity_curve.append(equity)

            trades.append(
                {
                    "symbol": symbol,
                    "entry": entry,
                    "exit": exit_price,
                    "pnl": pnl,
                    "outcome": outcome,
                    "r": r_multiple,
                }
            )

            in_trade = False

            continue

        # ====================================================
        # LOOK FOR NEW SIGNAL
        # ====================================================

        if not check_signal(row):
            continue

        # ====================================================
        # ENTER NEXT BAR
        #
        # This is important.
        #
        # The signal is known at the CLOSE of bar i.
        # We therefore cannot realistically fill at that same
        # close without introducing look-ahead/execution bias.
        # ====================================================

        if i + 1 >= len(df):
            break

        next_row = df.iloc[i + 1]

        raw_entry = float(
            next_row["Open"]
        )

        if raw_entry <= 0:
            continue

        entry_price = (
            raw_entry
            * (1 + SLIPPAGE_PCT)
        )

        atr = float(
            row["ATR"]
        )

        if not np.isfinite(atr) or atr <= 0:
            continue

        levels = calculate_trade_levels(
            entry_price,
            atr,
        )

        if levels is None:
            continue

        qty = int(
            STARTING_ALLOCATION
            // entry_price
        )

        if qty < 1:
            continue

        allocated = (
            qty * entry_price
        )

        # Keep allocation approximately at
        # the requested $1,000 budget.
        if allocated <= 0:
            continue

        entry = levels.entry
        stop = levels.stop
        target = levels.target

        allocation = allocated

        initial_risk = (
            qty
            * (entry - stop)
        )

        if initial_risk <= 0:
            continue

        in_trade = True

    # ========================================================
    # FORCE CLOSE REMAINING POSITION
    # ========================================================

    if in_trade:

        last_close = float(
            df.iloc[-1]["Close"]
        )

        exit_price = (
            last_close
            * (1 - SLIPPAGE_PCT)
        )

        pnl = (
            allocation
            * (
                exit_price - entry
            )
            / entry
        )

        pnl -= (
            COMMISSION_PER_SIDE * 2
        )

        r_multiple = (
            pnl / initial_risk
            if initial_risk > 0
            else 0.0
        )

        equity += pnl

        equity_curve.append(equity)

        trades.append(
            {
                "symbol": symbol,
                "entry": entry,
                "exit": exit_price,
                "pnl": pnl,
                "outcome": (
                    "WIN"
                    if pnl > 0
                    else "LOSS"
                ),
                "r": r_multiple,
            }
        )

        forced_exits += 1

    # ========================================================
    # STATISTICS
    # ========================================================

    total = len(trades)

    wins = [
        t for t in trades
        if t["outcome"] == "WIN"
    ]

    losses = [
        t for t in trades
        if t["outcome"] == "LOSS"
    ]

    win_rate = (
        len(wins) / total * 100
        if total
        else 0.0
    )

    total_pnl = sum(
        t["pnl"]
        for t in trades
    )

    expectancy = (
        total_pnl / total
        if total
        else 0.0
    )

    gross_profit = sum(
        t["pnl"]
        for t in wins
    )

    gross_loss = abs(
        sum(
            t["pnl"]
            for t in losses
        )
    )

    if gross_loss > 0:
        profit_factor = (
            gross_profit
            / gross_loss
        )
    elif gross_profit > 0:
        profit_factor = float("inf")
    else:
        profit_factor = 0.0

    avg_r = (
        np.mean(
            [t["r"] for t in trades]
        )
        if trades
        else 0.0
    )

    # ========================================================
    # MAX DRAWDOWN
    # ========================================================

    curve = np.array(
        equity_curve,
        dtype=float,
    )

    if len(curve):

        peaks = np.maximum.accumulate(
            curve
        )

        drawdowns = (
            curve - peaks
        ) / peaks

        max_drawdown = (
            float(drawdowns.min())
            * 100
        )

    else:
        max_drawdown = 0.0

    return {
        "symbol": symbol,
        "trades": total,
        "wins": len(wins),
        "losses": len(losses),
        "win_rate": round(
            win_rate,
            2,
        ),
        "pnl": round(
            total_pnl,
            2,
        ),
        "expectancy": round(
            expectancy,
            2,
        ),
        "profit_factor": (
            None
            if profit_factor == float("inf")
            else round(
                profit_factor,
                2,
            )
        ),
        "profit_factor_display": (
            "inf"
            if profit_factor == float("inf")
            else f"{profit_factor:.2f}"
        ),
        "avg_r": round(
            float(avg_r),
            3,
        ),
        "max_drawdown": round(
            max_drawdown,
            2,
        ),
        "ambiguous": ambiguous,
        "forced_exits": forced_exits,
    }


# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":

    print("=" * 78)
    print(
        "  BULL. BEAR AND BROKE — UNIFIED STRATEGY BACKTEST v10"
    )
    print(
        "  Shared signal engine | Next-bar execution | ATR exits"
    )
    print("=" * 78)

    results = []

    for ticker in TICKERS:

        try:

            result = run_backtest(ticker)

            results.append(result)

            print(
                f"[{ticker:<5}] "
                f"Trades={result['trades']:<3} | "
                f"W={result['wins']:<3} | "
                f"L={result['losses']:<3} | "
                f"Win={result['win_rate']:>5.1f}% | "
                f"P&L=${result['pnl']:>8.2f} | "
                f"Exp=${result['expectancy']:>7.2f} | "
                f"PF={result['profit_factor_display']:>5} | "
                f"AvgR={result['avg_r']:>6.3f} | "
                f"DD={result['max_drawdown']:>6.2f}%"
            )

        except Exception as e:

            print(
                f"[{ticker:<5}] ERROR: {e}"
            )

    # ========================================================
    # OVERALL
    # ========================================================

    total_trades = sum(
        r["trades"]
        for r in results
    )

    total_wins = sum(
        r["wins"]
        for r in results
    )

    total_losses = sum(
        r["losses"]
        for r in results
    )

    total_pnl = sum(
        r["pnl"]
        for r in results
    )

    total_expectancy = (
        total_pnl / total_trades
        if total_trades
        else 0.0
    )

    gross_profit = sum(
        max(0, r["pnl"])
        for r in results
    )

    gross_loss = abs(
        sum(
            min(0, r["pnl"])
            for r in results
        )
    )

    overall_pf = (
        gross_profit / gross_loss
        if gross_loss > 0
        else (
            float("inf")
            if gross_profit > 0
            else 0
        )
    )

    overall_win_rate = (
        total_wins
        / total_trades
        * 100
        if total_trades
        else 0
    )

    print("-" * 78)

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
        f"${total_expectancy:.2f}"
    )

    print(
        f"PROFIT FACTOR: "
        f"{overall_pf:.3f}"
    )

    print(
        f"AMBIGUOUS TP/SL BARS: "
        f"{sum(r['ambiguous'] for r in results)}"
    )

    print(
        f"FORCED END-OF-DATA EXITS: "
        f"{sum(r['forced_exits'] for r in results)}"
    )

    print("=" * 78)

    if total_trades < 100:
        print(
            f"⚠️ Sample size is small "
            f"({total_trades} trades)."
        )
    else:
        print(
            "✅ Larger sample reached."
        )

    print(
        "IMPORTANT: Run a separate out-of-sample period "
        "before judging the strategy."
    )
