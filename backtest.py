"""
BULL. BEAR AND BROKE - v14 BACKTESTER

Important:
This backtester imports calculate_indicators() and check_signal()
directly from strategy.py.

There is no duplicate signal logic here.

Signal:
    strategy.check_signal()

Position sizing:
    strategy.calculate_position_size()

Stop / target:
    strategy.calculate_exit_levels()
"""

from __future__ import annotations

import os
import math
from typing import Dict, Any, List

import numpy as np
import pandas as pd
import yfinance as yf

from strategy import (
    calculate_indicators,
    check_signal,
    calculate_position_size,
    calculate_exit_levels,
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

STARTING_EQUITY = 10_000.00

SLIPPAGE_PCT = 0.0005
COMMISSION_PER_SIDE = 0.00

DATA_PERIOD = "60d"
DATA_INTERVAL = "15m"


# ============================================================
# DATA
# ============================================================

def download_data(
    symbol: str,
    period: str = DATA_PERIOD,
    interval: str = DATA_INTERVAL,
) -> pd.DataFrame:

    df = yf.download(
        symbol,
        period=period,
        interval=interval,
        auto_adjust=False,
        progress=False,
    )

    if df is None or df.empty:
        return pd.DataFrame()

    return df


# ============================================================
# METRICS
# ============================================================

def calculate_max_drawdown(
    equity_curve: pd.Series,
) -> float:

    if equity_curve.empty:
        return 0.0

    running_max = equity_curve.cummax()

    drawdown = (
        equity_curve / running_max
    ) - 1.0

    return float(
        drawdown.min() * 100.0
    )


def calculate_sharpe(
    returns: pd.Series,
) -> float:

    returns = returns.dropna()

    if len(returns) < 2:
        return 0.0

    std = returns.std()

    if std == 0 or pd.isna(std):
        return 0.0

    return float(
        np.sqrt(252) * returns.mean() / std
    )


def calculate_sortino(
    returns: pd.Series,
) -> float:

    returns = returns.dropna()

    if len(returns) < 2:
        return 0.0

    downside = returns[returns < 0]

    if len(downside) == 0:
        return float("inf")

    downside_std = downside.std()

    if downside_std == 0 or pd.isna(downside_std):
        return 0.0

    return float(
        np.sqrt(252)
        * returns.mean()
        / downside_std
    )


# ============================================================
# SINGLE SYMBOL BACKTEST
# ============================================================

def run_backtest(
    symbol: str,
    period: str = DATA_PERIOD,
    interval: str = DATA_INTERVAL,
) -> Dict[str, Any]:

    raw = download_data(
        symbol,
        period,
        interval,
    )

    if raw.empty:
        return {
            "symbol": symbol,
            "trades": [],
            "total_trades": 0,
            "wins": 0,
            "losses": 0,
            "win_rate": 0.0,
            "net_pnl": 0.0,
            "expectancy": 0.0,
            "profit_factor": 0.0,
            "max_drawdown": 0.0,
            "sharpe": 0.0,
            "sortino": 0.0,
            "ambiguous_bars": 0,
            "open_at_end": False,
        }

    df = calculate_indicators(
        raw,
        symbol=symbol,
    )

    equity = STARTING_EQUITY

    trades: List[Dict[str, Any]] = []

    in_trade = False

    qty = 0
    entry_price = 0.0
    stop_price = 0.0
    target_price = 0.0

    entry_time = None
    signal_time = None

    ambiguous_bars = 0

    equity_points = []

    # We need a next bar, so stop at len(df)-2.
    for i in range(60, len(df) - 1):

        row = df.iloc[i]

        current_time = df.index[i]

        # ====================================================
        # MANAGE EXISTING TRADE
        # ====================================================

        if in_trade:

            high = float(row["High"])
            low = float(row["Low"])

            hit_stop = (
                low <= stop_price
            )

            hit_target = (
                high >= target_price
            )

            # Conservative:
            # If both are touched in the same bar,
            # assume stop was hit first.
            if hit_stop and hit_target:

                ambiguous_bars += 1

                exit_price = (
                    stop_price
                    * (1.0 - SLIPPAGE_PCT)
                )

                exit_reason = "STOP_AMBIGUOUS"

            elif hit_stop:

                exit_price = (
                    stop_price
                    * (1.0 - SLIPPAGE_PCT)
                )

                exit_reason = "STOP"

            elif hit_target:

                exit_price = (
                    target_price
                    * (1.0 - SLIPPAGE_PCT)
                )

                exit_reason = "TARGET"

            else:
                equity_points.append(
                    {
                        "timestamp": current_time,
                        "equity": equity,
                    }
                )

                continue

            gross_pnl = (
                qty
                * (exit_price - entry_price)
            )

            commissions = (
                2.0 * COMMISSION_PER_SIDE
            )

            net_pnl = (
                gross_pnl
                - commissions
            )

            equity += net_pnl

            risk_per_share = (
                entry_price - stop_price
            )

            initial_risk = (
                qty * risk_per_share
            )

            r_multiple = (
                net_pnl / initial_risk
                if initial_risk > 0
                else 0.0
            )

            return_pct = (
                (exit_price - entry_price)
                / entry_price
                * 100.0
            )

            hold_minutes = (
                (current_time - entry_time)
                .total_seconds()
                / 60.0
            )

            trades.append(
                {
                    "symbol": symbol,
                    "signal_time": signal_time,
                    "entry_time": entry_time,
                    "exit_time": current_time,
                    "entry_price": entry_price,
                    "exit_price": exit_price,
                    "qty": qty,
                    "stop_price": stop_price,
                    "target_price": target_price,
                    "pnl": net_pnl,
                    "return_pct": return_pct,
                    "r_multiple": r_multiple,
                    "hold_minutes": hold_minutes,
                    "outcome": (
                        "WIN"
                        if net_pnl > 0
                        else "LOSS"
                    ),
                    "exit_reason": exit_reason,
                }
            )

            in_trade = False
            qty = 0

            equity_points.append(
                {
                    "timestamp": current_time,
                    "equity": equity,
                }
            )

            continue

        # ====================================================
        # LOOK FOR NEW SIGNAL
        # ====================================================

        valid, details = check_signal(row)

        if not valid:
            equity_points.append(
                {
                    "timestamp": current_time,
                    "equity": equity,
                }
            )
            continue

        # ====================================================
        # ENTER ON NEXT BAR
        # ====================================================

        next_row = df.iloc[i + 1]

        next_open = float(
            next_row["Open"]
        )

        if next_open <= 0:
            continue

        atr = float(row["ATR"])

        if pd.isna(atr) or atr <= 0:
            continue

        # Slippage applied to actual next-bar entry.
        actual_entry = (
            next_open
            * (1.0 + SLIPPAGE_PCT)
        )

        position_qty = calculate_position_size(
            equity=equity,
            entry_price=actual_entry,
            atr=atr,
        )

        if position_qty <= 0:
            continue

        stop, target = calculate_exit_levels(
            entry_price=actual_entry,
            atr=atr,
        )

        if (
            stop <= 0
            or target <= actual_entry
        ):
            continue

        in_trade = True

        qty = position_qty

        entry_price = actual_entry

        stop_price = stop
        target_price = target

        signal_time = current_time
        entry_time = df.index[i + 1]

    # ========================================================
    # FORCE CLOSE OPEN TRADE AT END OF DATA
    # ========================================================

    open_at_end = False

    if in_trade:

        open_at_end = True

        final_row = df.iloc[-1]

        final_price = float(
            final_row["Close"]
        )

        exit_price = (
            final_price
            * (1.0 - SLIPPAGE_PCT)
        )

        gross_pnl = (
            qty
            * (exit_price - entry_price)
        )

        commissions = (
            2.0 * COMMISSION_PER_SIDE
        )

        net_pnl = (
            gross_pnl
            - commissions
        )

        equity += net_pnl

        risk_per_share = (
            entry_price - stop_price
        )

        initial_risk = (
            qty * risk_per_share
        )

        r_multiple = (
            net_pnl / initial_risk
            if initial_risk > 0
            else 0.0
        )

        return_pct = (
            (exit_price - entry_price)
            / entry_price
            * 100.0
        )

        hold_minutes = (
            (
                df.index[-1]
                - entry_time
            ).total_seconds()
            / 60.0
        )

        trades.append(
            {
                "symbol": symbol,
                "signal_time": signal_time,
                "entry_time": entry_time,
                "exit_time": df.index[-1],
                "entry_price": entry_price,
                "exit_price": exit_price,
                "qty": qty,
                "stop_price": stop_price,
                "target_price": target_price,
                "pnl": net_pnl,
                "return_pct": return_pct,
                "r_multiple": r_multiple,
                "hold_minutes": hold_minutes,
                "outcome": (
                    "WIN"
                    if net_pnl > 0
                    else "LOSS"
                ),
                "exit_reason": "END_OF_DATA",
            }
        )

    # ========================================================
    # STATISTICS
    # ========================================================

    total_trades = len(trades)

    wins = [
        t for t in trades
        if t["outcome"] == "WIN"
    ]

    losses = [
        t for t in trades
        if t["outcome"] == "LOSS"
    ]

    wins_count = len(wins)
    losses_count = len(losses)

    win_rate = (
        wins_count
        / total_trades
        * 100.0
        if total_trades
        else 0.0
    )

    net_pnl = sum(
        t["pnl"]
        for t in trades
    )

    expectancy = (
        net_pnl / total_trades
        if total_trades
        else 0.0
    )

    gross_profit = sum(
        max(0.0, t["pnl"])
        for t in trades
    )

    gross_loss = sum(
        abs(min(0.0, t["pnl"]))
        for t in trades
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

    r_values = [
        t["r_multiple"]
        for t in trades
    ]

    average_r = (
        float(np.mean(r_values))
        if r_values
        else 0.0
    )

    average_win = (
        float(np.mean([
            t["pnl"]
            for t in wins
        ]))
        if wins
        else 0.0
    )

    average_loss = (
        float(np.mean([
            t["pnl"]
            for t in losses
        ]))
        if losses
        else 0.0
    )

    hold_times = [
        t["hold_minutes"]
        for t in trades
    ]

    average_hold_minutes = (
        float(np.mean(hold_times))
        if hold_times
        else 0.0
    )

    # Build equity curve from completed trades.
    equity_curve = pd.Series(
        [STARTING_EQUITY]
        + [
            STARTING_EQUITY
            + sum(
                x["pnl"]
                for x in trades[:j + 1]
            )
            for j in range(len(trades))
        ]
    )

    max_drawdown = calculate_max_drawdown(
        equity_curve
    )

    trade_returns = pd.Series(
        [
            t["pnl"] / STARTING_EQUITY
            for t in trades
        ]
    )

    sharpe = calculate_sharpe(
        trade_returns
    )

    sortino = calculate_sortino(
        trade_returns
    )

    max_consecutive_losses = 0
    max_consecutive_wins = 0

    current_losses = 0
    current_wins = 0

    for trade in trades:

        if trade["outcome"] == "LOSS":
            current_losses += 1
            current_wins = 0

        else:
            current_wins += 1
            current_losses = 0

        max_consecutive_losses = max(
            max_consecutive_losses,
            current_losses,
        )

        max_consecutive_wins = max(
            max_consecutive_wins,
            current_wins,
        )

    return {
        "symbol": symbol,
        "total_trades": total_trades,
        "wins": wins_count,
        "losses": losses_count,
        "win_rate": round(win_rate, 2),
        "net_pnl": round(net_pnl, 2),
        "expectancy": round(expectancy, 2),
        "profit_factor": profit_factor,
        "average_win": round(average_win, 2),
        "average_loss": round(average_loss, 2),
        "average_r": round(average_r, 3),
        "max_drawdown": round(max_drawdown, 2),
        "sharpe": round(sharpe, 3),
        "sortino": round(sortino, 3),
        "average_hold_minutes": round(
            average_hold_minutes,
            1,
        ),
        "max_consecutive_losses": max_consecutive_losses,
        "max_consecutive_wins": max_consecutive_wins,
        "ambiguous_bars": ambiguous_bars,
        "open_at_end": open_at_end,
        "trades": trades,
    }


# ============================================================
# PORTFOLIO BACKTEST
# ============================================================

def main():

    print("=" * 78)
    print(" BULL. BEAR AND BROKE - v14 UNIFIED STRATEGY BACKTEST")
    print("=" * 78)

    print(
        f"Data: {DATA_INTERVAL} | "
        f"Period: {DATA_PERIOD}"
    )

    print(
        f"Starting equity per ticker: "
        f"${STARTING_EQUITY:,.2f}"
    )

    print(
        "Signal engine: strategy.check_signal()"
    )

    print("-" * 78)

    results = []

    for symbol in TICKERS:

        try:

            result = run_backtest(
                symbol,
                period=DATA_PERIOD,
                interval=DATA_INTERVAL,
            )

            results.append(result)

            pf = result["profit_factor"]

            if math.isinf(pf):
                pf_display = "inf"
            else:
                pf_display = f"{pf:.2f}"

            print(
                f"[{symbol:<5}] "
                f"Trades={result['total_trades']:<3} | "
                f"W={result['wins']:<3} | "
                f"L={result['losses']:<3} | "
                f"Win={result['win_rate']:>5.1f}% | "
                f"P&L=${result['net_pnl']:>9.2f} | "
                f"Exp=${result['expectancy']:>7.2f} | "
                f"PF={pf_display:>5} | "
                f"AvgR={result['average_r']:>6.3f} | "
                f"DD={result['max_drawdown']:>6.2f}%"
            )

        except Exception as e:

            print(
                f"[{symbol:<5}] ERROR: {e}"
            )

    print("-" * 78)

    all_trades = []

    for result in results:
        all_trades.extend(
            result["trades"]
        )

    total_trades = len(all_trades)

    total_wins = sum(
        1
        for t in all_trades
        if t["outcome"] == "WIN"
    )

    total_losses = (
        total_trades
        - total_wins
    )

    total_pnl = sum(
        t["pnl"]
        for t in all_trades
    )

    overall_win_rate = (
        total_wins
        / total_trades
        * 100.0
        if total_trades
        else 0.0
    )

    overall_expectancy = (
        total_pnl
        / total_trades
        if total_trades
        else 0.0
    )

    gross_profit = sum(
        max(0.0, t["pnl"])
        for t in all_trades
    )

    gross_loss = sum(
        abs(min(0.0, t["pnl"]))
        for t in all_trades
    )

    if gross_loss > 0:
        overall_pf = (
            gross_profit
            / gross_loss
        )
    elif gross_profit > 0:
        overall_pf = float("inf")
    else:
        overall_pf = 0.0

    average_r = (
        np.mean([
            t["r_multiple"]
            for t in all_trades
        ])
        if all_trades
        else 0.0
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
        f"${total_pnl:,.2f}"
    )

    print(
        f"EXPECTANCY / TRADE: "
        f"${overall_expectancy:,.2f}"
    )

    if math.isinf(overall_pf):
        print("PROFIT FACTOR: inf")
    else:
        print(
            f"PROFIT FACTOR: "
            f"{overall_pf:.3f}"
        )

    print(
        f"AVERAGE R: "
        f"{average_r:.3f}"
    )

    total_ambiguous = sum(
        r["ambiguous_bars"]
        for r in results
    )

    print(
        f"AMBIGUOUS TP/SL BARS: "
        f"{total_ambiguous}"
    )

    open_positions = sum(
        1
        for r in results
        if r["open_at_end"]
    )

    print(
        f"FORCED END-OF-DATA EXITS: "
        f"{open_positions}"
    )

    print("=" * 78)

    if total_trades < 100:
        print(
            f"⚠️ WARNING: Only {total_trades} trades."
        )
        print(
            "Do NOT treat this as evidence of a "
            "stable trading edge yet."
        )

    if total_trades >= 100:
        print(
            "✅ Larger sample reached, but "
            "out-of-sample testing is still required."
        )


if __name__ == "__main__":
    main()
