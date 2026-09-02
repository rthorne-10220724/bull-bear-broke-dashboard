"""
BULL. BEAR AND BROKE - V14 BACKTEST
===================================

V14:

    4H regime
        ->
    15M trend/momentum
        ->
    1M pullback/reclaim
        ->
    next-bar execution
        ->
    ATR risk

Important:
    This is a research backtest.
    It is not a prediction of future profitability.
"""

from __future__ import annotations

from typing import Any, Dict, List
import math

import numpy as np
import pandas as pd
import yfinance as yf

from strategy import (
    align_timeframes,
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

EXECUTION_PERIOD = "7d"
HIGHER_PERIOD = "180d"

STARTING_CAPITAL = 1000.0

SLIPPAGE_PCT = 0.0005
COMMISSION_PER_TRADE = 0.0

RISK_PER_TRADE_PCT = 0.0075
MAX_POSITION_PCT = 0.25

WARMUP_BARS = 100


# ============================================================================
# DATA
# ============================================================================

def download_data(
    symbol: str,
    period: str,
    interval: str,
) -> pd.DataFrame:

    df = yf.download(
        symbol,
        period=period,
        interval=interval,
        auto_adjust=False,
        progress=False,
        prepost=False,
    )

    if df.empty:
        return pd.DataFrame()

    if isinstance(
        df.columns,
        pd.MultiIndex,
    ):

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

        else:

            df.columns = [
                c[0]
                for c in df.columns
            ]

    required = [
        "Open",
        "High",
        "Low",
        "Close",
        "Volume",
    ]

    if not all(
        c in df.columns
        for c in required
    ):
        return pd.DataFrame()

    df = df[required].copy()

    for column in required:

        df[column] = pd.to_numeric(
            df[column],
            errors="coerce",
        )

    df.dropna(
        subset=[
            "Open",
            "High",
            "Low",
            "Close",
        ],
        inplace=True,
    )

    df.sort_index(
        inplace=True
    )

    df = df[
        ~df.index.duplicated(
            keep="last"
        )
    ]

    return df


# ============================================================================
# POSITION SIZE
# ============================================================================

def calculate_position_size(
    equity: float,
    entry: float,
    stop: float,
) -> int:

    if (
        equity <= 0
        or entry <= 0
        or stop <= 0
        or stop >= entry
    ):
        return 0

    risk_per_share = (
        entry - stop
    )

    risk_budget = (
        equity
        * RISK_PER_TRADE_PCT
    )

    qty_by_risk = (
        risk_budget
        / risk_per_share
    )

    max_position = (
        equity
        * MAX_POSITION_PCT
    )

    qty_by_cap = (
        max_position
        / entry
    )

    return max(
        0,
        int(
            min(
                qty_by_risk,
                qty_by_cap,
            )
        ),
    )


# ============================================================================
# RESULT
# ============================================================================

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
        "avg_r": 0.0,
        "max_drawdown": 0.0,
        "ambiguous_bars": 0,
        "forced_exits": 0,
        "valid_signals": 0,
        "invalid_signals": 0,
        "zero_qty": 0,
        "exit_errors": 0,
        "rejections": {},
    }


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

    equity_curve = [
        equity
    ]

    in_trade = False

    qty = 0
    entry_price = 0.0
    stop_price = 0.0
    target_price = 0.0
    risk_dollars = 0.0

    ambiguous_bars = 0
    forced_exits = 0
    valid_signals = 0
    invalid_signals = 0
    zero_qty = 0
    exit_errors = 0

    rejection_counts: Dict[str, int] = {}

    for i in range(
        WARMUP_BARS,
        len(df) - 1,
    ):

        row = df.iloc[i]

        # ====================================================================
        # OPEN TRADE MANAGEMENT
        # ====================================================================

        if in_trade:

            high = float(
                row["High"]
            )

            low = float(
                row["Low"]
            )

            hit_stop = (
                low <= stop_price
            )

            hit_target = (
                high >= target_price
            )

            # Conservative:
            # if both were touched, stop wins.

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

                r = (
                    pnl / risk_dollars
                    if risk_dollars > 0
                    else 0.0
                )

                trades.append({
                    "symbol": symbol,
                    "entry": entry_price,
                    "exit": exit_price,
                    "qty": qty,
                    "pnl": pnl,
                    "r": r,
                    "outcome": "LOSS",
                })

                equity += pnl
                equity_curve.append(
                    equity
                )

                in_trade = False
                continue

            # =================================================================
            # GAP-AWARE STOP
            # =================================================================

            if hit_stop:

                bar_open = float(
                    row["Open"]
                )

                if bar_open < stop_price:
                    exit_base = bar_open
                else:
                    exit_base = stop_price

                exit_price = (
                    exit_base
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

                r = (
                    pnl / risk_dollars
                    if risk_dollars > 0
                    else 0.0
                )

                trades.append({
                    "symbol": symbol,
                    "entry": entry_price,
                    "exit": exit_price,
                    "qty": qty,
                    "pnl": pnl,
                    "r": r,
                    "outcome": "LOSS",
                })

                equity += pnl
                equity_curve.append(
                    equity
                )

                in_trade = False
                continue

            # =================================================================
            # TARGET
            # =================================================================

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

                r = (
                    pnl / risk_dollars
                    if risk_dollars > 0
                    else 0.0
                )

                trades.append({
                    "symbol": symbol,
                    "entry": entry_price,
                    "exit": exit_price,
                    "qty": qty,
                    "pnl": pnl,
                    "r": r,
                    "outcome": "WIN",
                })

                equity += pnl
                equity_curve.append(
                    equity
                )

                in_trade = False
                continue

        # ====================================================================
        # ENTRY
        # ====================================================================

        if not in_trade:

            signal = evaluate_signal(
                row
            )

            if not signal.valid:

                invalid_signals += 1

                rejection_counts[
                    signal.reason
                ] = (
                    rejection_counts.get(
                        signal.reason,
                        0,
                    ) + 1
                )

                equity_curve.append(
                    equity
                )

                continue

            valid_signals += 1

            next_row = df.iloc[
                i + 1
            ]

            raw_entry = float(
                next_row["Open"]
            )

            if not np.isfinite(
                raw_entry
            ):

                invalid_signals += 1

                rejection_counts[
                    "invalid_next_open"
                ] = (
                    rejection_counts.get(
                        "invalid_next_open",
                        0,
                    ) + 1
                )

                equity_curve.append(
                    equity
                )

                continue

            # =================================================================
            # ENTRY SLIPPAGE
            # =================================================================

            entry = (
                raw_entry
                * (1 + SLIPPAGE_PCT)
            )

            atr = float(
                row["ATR_1M"]
            )

            try:

                exits = calculate_exit_prices(
                    entry,
                    atr,
                )

            except Exception:

                exit_errors += 1

                equity_curve.append(
                    equity
                )

                continue

            stop = exits["stop"]
            target = exits["target"]

            qty_candidate = (
                calculate_position_size(
                    equity,
                    entry,
                    stop,
                )
            )

            if qty_candidate <= 0:

                zero_qty += 1

                equity_curve.append(
                    equity
                )

                continue

            # Never borrow cash.

            max_cash_qty = int(
                equity / entry
            )

            qty_candidate = min(
                qty_candidate,
                max_cash_qty,
            )

            if qty_candidate <= 0:

                zero_qty += 1

                equity_curve.append(
                    equity
                )

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

    # =========================================================================
    # FORCE FINAL EXIT
    # =========================================================================

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

        r = (
            pnl / risk_dollars
            if risk_dollars > 0
            else 0.0
        )

        trades.append({
            "symbol": symbol,
            "entry": entry_price,
            "exit": exit_price,
            "qty": qty,
            "pnl": pnl,
            "r": r,
            "outcome": (
                "WIN"
                if pnl > 0
                else "LOSS"
            ),
        })

        equity += pnl
        equity_curve.append(
            equity
        )

        forced_exits += 1

    # =========================================================================
    # STATS
    # =========================================================================

    total = len(trades)

    wins = [
        t for t in trades
        if t["pnl"] > 0
    ]

    losses = [
        t for t in trades
        if t["pnl"] <= 0
    ]

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

    peak = -math.inf
    max_drawdown = 0.0

    for value in equity_curve:

        peak = max(
            peak,
            value,
        )

        if peak > 0:

            dd = (
                value - peak
            ) / peak

            max_drawdown = min(
                max_drawdown,
                dd,
            )

    return {
        "symbol": symbol,
        "trades": total,
        "wins": len(wins),
        "losses": len(losses),
        "win_rate": (
            len(wins)
            / total
            * 100
            if total
            else 0.0
        ),
        "pnl": sum(
            t["pnl"]
            for t in trades
        ),
        "expectancy": (
            sum(
                t["pnl"]
                for t in trades
            )
            / total
            if total
            else 0.0
        ),
        "profit_factor": profit_factor,
        "avg_r": (
            np.mean(
                [
                    t["r"]
                    for t in trades
                ]
            )
            if trades
            else 0.0
        ),
        "max_drawdown": (
            max_drawdown * 100
        ),
        "ambiguous_bars": ambiguous_bars,
        "forced_exits": forced_exits,
        "valid_signals": valid_signals,
        "invalid_signals": invalid_signals,
        "zero_qty": zero_qty,
        "exit_errors": exit_errors,
        "rejections": rejection_counts,
    }


# ============================================================================
# MAIN
# ============================================================================

def main() -> None:

    print("=" * 82)
    print(
        "  BULL. BEAR AND BROKE — V14 BACKTEST"
    )
    print(
        "  4H REGIME | 15M MOMENTUM | 1M PULLBACK/RECLAIM"
    )
    print("=" * 82)

    results = []

    for symbol in TICKERS:

        print(
            f"\n[{symbol:<5}] 1M execution data..."
        )

        one = download_data(
            symbol,
            EXECUTION_PERIOD,
            "1m",
        )

        if one.empty:

            print(
                f"[{symbol:<5}] NO 1M DATA"
            )

            results.append(
                empty_result(symbol)
            )

            continue

        print(
            f"          {len(one):,} 1M bars"
        )

        print(
            f"          15M confirmation..."
        )

        fifteen = download_data(
            symbol,
            HIGHER_PERIOD,
            "15m",
        )

        if fifteen.empty:

            print(
                f"[{symbol:<5}] NO 15M DATA"
            )

            results.append(
                empty_result(symbol)
            )

            continue

        print(
            f"          {len(fifteen):,} 15M bars"
        )

        print(
            f"          1H data for 4H regime..."
        )

        hourly = download_data(
            symbol,
            HIGHER_PERIOD,
            "1h",
        )

        if hourly.empty:

            print(
                f"[{symbol:<5}] NO 1H DATA"
            )

            results.append(
                empty_result(symbol)
            )

            continue

        # =====================================================================
        # BUILD 4H BARS
        # =====================================================================

        four = (
            hourly
            .resample("4h")
            .agg({
                "Open": "first",
                "High": "max",
                "Low": "min",
                "Close": "last",
                "Volume": "sum",
            })
            .dropna(
                subset=[
                    "Open",
                    "High",
                    "Low",
                    "Close",
                ]
            )
        )

        print(
            f"          {len(four):,} 4H bars"
        )

        # =====================================================================
        # ALIGN
        # =====================================================================

        try:

            df = align_timeframes(
                one,
                fifteen,
                four,
            )

        except Exception as exc:

            print(
                f"[{symbol:<5}] ALIGN ERROR: {exc}"
            )

            results.append(
                empty_result(symbol)
            )

            continue

        # =====================================================================
        # DIAGNOSTICS
        # =====================================================================

        print(
            f"          DATA QUALITY"
        )

        for column in [
            "ATR_1M",
            "RSI_1M",
            "MACD_DIFF_15M",
            "EMA20_4H",
        ]:

            valid = int(
                df[column]
                .notna()
                .sum()
            )

            total = len(df)

            pct = (
                valid / total * 100
                if total
                else 0.0
            )

            print(
                f"          {column:<18}"
                f"{valid:>6}/{total:<6}"
                f" {pct:5.1f}%"
            )

        # =====================================================================
        # RUN
        # =====================================================================

        result = run_backtest(
            symbol,
            df,
        )

        results.append(
            result
        )

        pf = result[
            "profit_factor"
        ]

        pf_text = (
            "inf"
            if pf == float("inf")
            else f"{pf:.2f}"
        )

        print(
            f"[{symbol:<5}] "
            f"Trades={result['trades']:<3} "
            f"W={result['wins']:<3} "
            f"L={result['losses']:<3} "
            f"Win={result['win_rate']:>5.1f}% "
            f"P&L=${result['pnl']:>8.2f} "
            f"Exp=${result['expectancy']:>7.2f} "
            f"PF={pf_text:>5} "
            f"AvgR={result['avg_r']:>6.3f} "
            f"DD={result['max_drawdown']:>6.2f}%"
        )

        print(
            f"          Valid signals: "
            f"{result['valid_signals']}"
        )

        print(
            f"          Zero qty: "
            f"{result['zero_qty']}"
        )

        print(
            f"          Ambiguous: "
            f"{result['ambiguous_bars']}"
        )

        print(
            f"          Forced exits: "
            f"{result['forced_exits']}"
        )

        # =====================================================================
        # REJECTION BREAKDOWN
        # =====================================================================

        if result["rejections"]:

            print(
                "          Rejection breakdown:"
            )

            ordered = sorted(
                result["rejections"].items(),
                key=lambda x: x[1],
                reverse=True,
            )

            for reason, count in ordered[:10]:

                print(
                    f"            "
                    f"{reason:<30}"
                    f"{count}"
                )

    # =========================================================================
    # PORTFOLIO SUMMARY
    # =========================================================================

    print("\n" + "=" * 82)
    print("  PORTFOLIO SUMMARY")
    print("=" * 82)

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

    gross_profit = sum(
        max(r["pnl"], 0)
        for r in results
    )

    gross_loss = abs(
        sum(
            min(r["pnl"], 0)
            for r in results
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

    win_rate = (
        total_wins
        / total_trades
        * 100
        if total_trades
        else 0.0
    )

    expectancy = (
        total_pnl
        / total_trades
        if total_trades
        else 0.0
    )

    print(
        f"Trades:              {total_trades}"
    )

    print(
        f"Wins / Losses:       "
        f"{total_wins} / {total_losses}"
    )

    print(
        f"Win rate:             "
        f"{win_rate:.2f}%"
    )

    print(
        f"Net P&L:              "
        f"${total_pnl:.2f}"
    )

    print(
        f"Expectancy/trade:     "
        f"${expectancy:.2f}"
    )

    if portfolio_pf == float("inf"):

        print(
            "Profit factor:       inf"
        )

    else:

        print(
            f"Profit factor:        "
            f"{portfolio_pf:.3f}"
        )

    print(
        "------------------------------------------"
    )

    if total_trades == 0:

        print(
            "❌ NO TRADES."
        )

    elif expectancy <= 0:

        print(
            "❌ NEGATIVE EXPECTANCY."
        )

    elif portfolio_pf <= 1:

        print(
            "⚠️ Positive expectancy but PF <= 1."
        )

    elif total_trades < 100:

        print(
            "⚠️ POSITIVE RESULT — BUT SAMPLE IS SMALL."
        )

    else:

        print(
            "✅ POSITIVE HISTORICAL RESULT."
        )

    print(
        "Do NOT optimize parameters from this "
        "single test window."
    )

    print(
        "Validate on a separate out-of-sample period."
    )

    print("=" * 82)


if __name__ == "__main__":
    main()
