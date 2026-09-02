"""
BULL. BEAR AND BROKE - V13 BACKTEST
===================================

Tests the V13 strategy using:

    - 1-minute data
    - shared strategy.py signal engine
    - next-bar execution
    - ATR stop
    - ATR target
    - slippage
    - commissions
    - same-bar ambiguity handling
    - forced end-of-data exits
    - max drawdown
    - R statistics

IMPORTANT:
    This is NOT a guarantee of future performance.

Use the results to compare V13 against V12.
"""

from __future__ import annotations

import math
from typing import Any, Dict, List

import numpy as np
import pandas as pd
import yfinance as yf

from strategy import (
    calculate_indicators,
    evaluate_signal,
    calculate_exit_prices,
)


# ============================================================================
# CONFIG
# ============================================================================

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
INTERVAL = "1m"

STARTING_CAPITAL = 1000.0

# Realistic-ish baseline.
# Change these only when comparing all strategies equally.
SLIPPAGE_PCT = 0.0005

COMMISSION_PER_TRADE = 0.0

# Position sizing:
# risk a fixed fraction of account equity per trade.
RISK_PER_TRADE_PCT = 0.0075

MAX_POSITION_PCT = 0.25

# Minimum data before signal evaluation.
WARMUP_BARS = 500


# ============================================================================
# DATA
# ============================================================================

def download_data(symbol: str) -> pd.DataFrame:

    print(f"[{symbol:<5}] downloading...")

    df = yf.download(
        symbol,
        period=PERIOD,
        interval=INTERVAL,
        auto_adjust=False,
        progress=False,
    )

    if df.empty:
        return pd.DataFrame()

    # Flatten yfinance MultiIndex.
    if isinstance(df.columns, pd.MultiIndex):

        if symbol in df.columns.get_level_values(-1):
            df = df.xs(
                symbol,
                level=-1,
                axis=1,
            )

        elif symbol in df.columns.get_level_values(0):
            df = df.xs(
                symbol,
                level=0,
                axis=1,
            )

    required = [
        "Open",
        "High",
        "Low",
        "Close",
        "Volume",
    ]

    if not all(
        column in df.columns
        for column in required
    ):
        return pd.DataFrame()

    df = df[required].copy()

    for column in required:
        df[column] = pd.to_numeric(
            df[column],
            errors="coerce",
        )

    df.dropna(inplace=True)

    # Ensure chronological order.
    df.sort_index(inplace=True)

    # Remove duplicate timestamps.
    df = df[
        ~df.index.duplicated(
            keep="last"
        )
    ]

    return df


# ============================================================================
# FEATURE PREPARATION
# ============================================================================

def prepare_features(
    df: pd.DataFrame,
) -> pd.DataFrame:

    df = calculate_indicators(df)

    if df.empty:
        return df

    # Previous values used by signal engine.
    df["PREV_RSI_1M"] = (
        df["RSI_1M"].shift(1)
    )

    df["PREV_MACD_DIFF_15M"] = (
        df["MACD_DIFF_15M"].shift(1)
    )

    df["PREV_EMA20_4H"] = (
        df["EMA20_4H"].shift(1)
    )

    # Previous completed 5-bar high.
    df["PREV_5_HIGH"] = (
        df["High"]
        .shift(1)
        .rolling(5)
        .max()
    )

    return df


# ============================================================================
# POSITION SIZING
# ============================================================================

def calculate_position_size(
    equity: float,
    entry_price: float,
    stop_price: float,
) -> int:

    if (
        equity <= 0
        or entry_price <= 0
        or stop_price <= 0
    ):
        return 0

    risk_per_share = (
        entry_price - stop_price
    )

    if risk_per_share <= 0:
        return 0

    risk_dollars = (
        equity * RISK_PER_TRADE_PCT
    )

    qty_by_risk = (
        risk_dollars
        / risk_per_share
    )

    max_position_dollars = (
        equity * MAX_POSITION_PCT
    )

    qty_by_cap = (
        max_position_dollars
        / entry_price
    )

    qty = min(
        qty_by_risk,
        qty_by_cap,
    )

    return max(
        0,
        int(qty),
    )


# ============================================================================
# BACKTEST
# ============================================================================

def run_backtest(
    symbol: str,
    df: pd.DataFrame,
) -> Dict[str, Any]:

    if df.empty:
        return empty_result(symbol)

    equity = STARTING_CAPITAL

    trades: List[Dict[str, Any]] = []

    in_trade = False

    qty = 0
    entry_price = 0.0
    stop_price = 0.0
    target_price = 0.0
    risk_dollars = 0.0

    ambiguous_bars = 0
    forced_exits = 0

    equity_curve = [
        equity
    ]

    # ------------------------------------------------------------------
    # IMPORTANT:
    #
    # Signal at bar i is executed at bar i+1.
    # This avoids look-ahead from using the current closing price
    # as the fill.
    # ------------------------------------------------------------------

    for i in range(
        WARMUP_BARS,
        len(df) - 1,
    ):

        row = df.iloc[i]

        # ==============================================================
        # MANAGE OPEN POSITION
        # ==============================================================

        if in_trade:

            high = float(row["High"])
            low = float(row["Low"])

            hit_stop = (
                low <= stop_price
            )

            hit_target = (
                high >= target_price
            )

            # ----------------------------------------------------------
            # Same bar touched both.
            #
            # Conservative assumption:
            # stop gets hit first.
            # ----------------------------------------------------------

            if hit_stop and hit_target:

                ambiguous_bars += 1

                exit_price = (
                    stop_price
                    * (1 - SLIPPAGE_PCT)
                )

                pnl = (
                    qty
                    * (
                        exit_price
                        - entry_price
                    )
                    - COMMISSION_PER_TRADE
                )

                r_multiple = (
                    pnl / risk_dollars
                    if risk_dollars > 0
                    else 0.0
                )

                trades.append(
                    {
                        "symbol": symbol,
                        "entry": entry_price,
                        "exit": exit_price,
                        "qty": qty,
                        "pnl": pnl,
                        "r": r_multiple,
                        "outcome": "LOSS",
                    }
                )

                equity += pnl
                equity_curve.append(equity)

                in_trade = False
                continue

            # ----------------------------------------------------------
            # Target
            # ----------------------------------------------------------

            if hit_target:

                exit_price = (
                    target_price
                    * (1 - SLIPPAGE_PCT)
                )

                pnl = (
                    qty
                    * (
                        exit_price
                        - entry_price
                    )
                    - COMMISSION_PER_TRADE
                )

                r_multiple = (
                    pnl / risk_dollars
                    if risk_dollars > 0
                    else 0.0
                )

                trades.append(
                    {
                        "symbol": symbol,
                        "entry": entry_price,
                        "exit": exit_price,
                        "qty": qty,
                        "pnl": pnl,
                        "r": r_multiple,
                        "outcome": "WIN",
                    }
                )

                equity += pnl
                equity_curve.append(equity)

                in_trade = False
                continue

            # ----------------------------------------------------------
            # Stop
            # ----------------------------------------------------------

            if hit_stop:

                exit_price = (
                    stop_price
                    * (1 - SLIPPAGE_PCT)
                )

                pnl = (
                    qty
                    * (
                        exit_price
                        - entry_price
                    )
                    - COMMISSION_PER_TRADE
                )

                r_multiple = (
                    pnl / risk_dollars
                    if risk_dollars > 0
                    else 0.0
                )

                trades.append(
                    {
                        "symbol": symbol,
                        "entry": entry_price,
                        "exit": exit_price,
                        "qty": qty,
                        "pnl": pnl,
                        "r": r_multiple,
                        "outcome": "LOSS",
                    }
                )

                equity += pnl
                equity_curve.append(equity)

                in_trade = False
                continue

        # ==============================================================
        # LOOK FOR NEW ENTRY
        # ==============================================================

        if not in_trade:

            signal = evaluate_signal(row)

            if not signal.valid:
                equity_curve.append(equity)
                continue

            # ----------------------------------------------------------
            # Next-bar execution.
            # ----------------------------------------------------------

            next_row = df.iloc[i + 1]

            raw_entry = float(
                next_row["Open"]
            )

            if not np.isfinite(raw_entry):
                equity_curve.append(equity)
                continue

            # Slippage against buyer.
            entry = (
                raw_entry
                * (1 + SLIPPAGE_PCT)
            )

            atr = float(
                row["ATR_1M"]
            )

            if (
                not np.isfinite(atr)
                or atr <= 0
            ):
                equity_curve.append(equity)
                continue

            exits = calculate_exit_prices(
                entry,
                atr,
            )

            stop = exits["stop"]
            target = exits["target"]

            qty_candidate = (
                calculate_position_size(
                    equity=equity,
                    entry_price=entry,
                    stop_price=stop,
                )
            )

            if qty_candidate <= 0:
                equity_curve.append(equity)
                continue

            # Don't exceed available starting capital.
            max_cash_qty = int(
                equity / entry
            )

            qty_candidate = min(
                qty_candidate,
                max_cash_qty,
            )

            if qty_candidate <= 0:
                equity_curve.append(equity)
                continue

            qty = qty_candidate
            entry_price = entry
            stop_price = stop
            target_price = target

            risk_dollars = (
                qty
                * (
                    entry_price
                    - stop_price
                )
            )

            in_trade = True

    # ==================================================================
    # FORCE CLOSE REMAINING POSITION
    # ==================================================================

    if in_trade:

        final_close = float(
            df["Close"].iloc[-1]
        )

        exit_price = (
            final_close
            * (1 - SLIPPAGE_PCT)
        )

        pnl = (
            qty
            * (
                exit_price
                - entry_price
            )
            - COMMISSION_PER_TRADE
        )

        r_multiple = (
            pnl / risk_dollars
            if risk_dollars > 0
            else 0.0
        )

        trades.append(
            {
                "symbol": symbol,
                "entry": entry_price,
                "exit": exit_price,
                "qty": qty,
                "pnl": pnl,
                "r": r_multiple,
                "outcome": (
                    "WIN"
                    if pnl > 0
                    else "LOSS"
                ),
            }
        )

        equity += pnl

        forced_exits += 1
        equity_curve.append(equity)

    # ==================================================================
    # STATISTICS
    # ==================================================================

    return calculate_statistics(
        symbol,
        trades,
        equity_curve,
        ambiguous_bars,
        forced_exits,
    )


# ============================================================================
# STATISTICS
# ============================================================================

def calculate_statistics(
    symbol: str,
    trades: List[Dict[str, Any]],
    equity_curve: List[float],
    ambiguous_bars: int,
    forced_exits: int,
) -> Dict[str, Any]:

    total = len(trades)

    wins = [
        trade
        for trade in trades
        if trade["pnl"] > 0
    ]

    losses = [
        trade
        for trade in trades
        if trade["pnl"] <= 0
    ]

    win_rate = (
        len(wins) / total * 100
        if total
        else 0.0
    )

    net_pnl = sum(
        trade["pnl"]
        for trade in trades
    )

    expectancy = (
        net_pnl / total
        if total
        else 0.0
    )

    gross_profit = sum(
        trade["pnl"]
        for trade in wins
    )

    gross_loss = abs(
        sum(
            trade["pnl"]
            for trade in losses
        )
    )

    profit_factor = (
        gross_profit / gross_loss
        if gross_loss > 0
        else (
            float("inf")
            if gross_profit > 0
            else 0.0
        )
    )

    avg_r = (
        np.mean(
            [
                trade["r"]
                for trade in trades
            ]
        )
        if trades
        else 0.0
    )

    # ------------------------------------------------------------------
    # Maximum drawdown
    # ------------------------------------------------------------------

    peak = -math.inf
    max_drawdown = 0.0

    for value in equity_curve:

        peak = max(
            peak,
            value,
        )

        if peak > 0:

            drawdown = (
                value - peak
            ) / peak

            max_drawdown = min(
                max_drawdown,
                drawdown,
            )

    pf_display = (
        "inf"
        if profit_factor == float("inf")
        else f"{profit_factor:.2f}"
    )

    return {
        "symbol": symbol,
        "trades": total,
        "wins": len(wins),
        "losses": len(losses),
        "win_rate": win_rate,
        "pnl": net_pnl,
        "expectancy": expectancy,
        "profit_factor": profit_factor,
        "pf_display": pf_display,
        "avg_r": avg_r,
        "max_drawdown": max_drawdown * 100,
        "ambiguous_bars": ambiguous_bars,
        "forced_exits": forced_exits,
    }


def empty_result(
    symbol: str,
) -> Dict[str, Any]:

    return {
        "symbol": symbol,
        "trades": 0,
        "wins": 0,
        "losses": 0,
        "win_rate": 0.0,
        "pnl": 0.0,
        "expectancy": 0.0,
        "profit_factor": 0.0,
        "pf_display": "0.00",
        "avg_r": 0.0,
        "max_drawdown": 0.0,
        "ambiguous_bars": 0,
        "forced_exits": 0,
    }


# ============================================================================
# MAIN
# ============================================================================

def main() -> None:

    print("=" * 78)

    print(
        "  BULL. BEAR AND BROKE — UNIFIED STRATEGY BACKTEST v13"
    )

    print(
        "  4H regime | 15M confirmation | 1M execution | ATR exits"
    )

    print("=" * 78)

    results: List[Dict[str, Any]] = []

    for symbol in TICKERS:

        raw = download_data(symbol)

        if raw.empty:

            print(
                f"[{symbol:<5}] no usable data"
            )

            results.append(
                empty_result(symbol)
            )

            continue

        df = prepare_features(raw)

        if df.empty:

            print(
                f"[{symbol:<5}] insufficient features"
            )

            results.append(
                empty_result(symbol)
            )

            continue

        result = run_backtest(
            symbol,
            df,
        )

        results.append(result)

        print(
            f"[{symbol:<5}] "
            f"Trades={result['trades']:<3} | "
            f"W={result['wins']:<3} | "
            f"L={result['losses']:<3} | "
            f"Win={result['win_rate']:>5.1f}% | "
            f"P&L=${result['pnl']:>8.2f} | "
            f"Exp=${result['expectancy']:>7.2f} | "
            f"PF={result['pf_display']:>5} | "
            f"AvgR={result['avg_r']:>6.3f} | "
            f"DD={result['max_drawdown']:>6.2f}%"
        )

    # ==================================================================
    # PORTFOLIO SUMMARY
    # ==================================================================

    total_trades = sum(
        result["trades"]
        for result in results
    )

    total_wins = sum(
        result["wins"]
        for result in results
    )

    total_losses = sum(
        result["losses"]
        for result in results
    )

    total_pnl = sum(
        result["pnl"]
        for result in results
    )

    total_expectancy = (
        total_pnl / total_trades
        if total_trades
        else 0.0
    )

    gross_profit = sum(
        max(result["pnl"], 0)
        for result in results
    )

    gross_loss = abs(
        sum(
            min(result["pnl"], 0)
            for result in results
        )
    )

    portfolio_pf = (
        gross_profit / gross_loss
        if gross_loss > 0
        else (
            float("inf")
            if gross_profit > 0
            else 0.0
        )
    )

    portfolio_win_rate = (
        total_wins
        / total_trades
        * 100
        if total_trades
        else 0.0
    )

    ambiguous = sum(
        result["ambiguous_bars"]
        for result in results
    )

    forced = sum(
        result["forced_exits"]
        for result in results
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
        f"{portfolio_win_rate:.2f}%"
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
        f"{portfolio_pf:.3f}"
        if portfolio_pf != float("inf")
        else "PROFIT FACTOR: inf"
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
            f"⚠️ Sample size is small "
            f"({total_trades} trades)."
        )

    if total_expectancy <= 0:

        print(
            "❌ V13 is NOT profitable on this test."
        )

    elif portfolio_pf <= 1.0:

        print(
            "⚠️ Positive P&L but PF <= 1.0. "
            "Investigate before trusting it."
        )

    else:

        print(
            "✅ V13 shows positive historical expectancy."
        )

    print(
        "IMPORTANT: This is an in-sample/historical test."
    )

    print(
        "Run a separate out-of-sample period before judging V13."
    )

    print("=" * 78)


if __name__ == "__main__":
    main()
