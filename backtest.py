"""
BULL. BEAR AND BROKE - V13.1 BACKTEST
=====================================

V13 diagnostic backtest.

Purpose:
    Fix the Yahoo 1-minute data-window issue and diagnose why V13
    may generate zero trades.

Tests:
    - 1-minute Yahoo data
    - shared strategy.py signal engine
    - next-bar execution
    - ATR stop
    - ATR target
    - slippage
    - commissions
    - same-bar ambiguity handling
    - forced end-of-data exits
    - position sizing
    - maximum drawdown
    - R statistics
    - signal diagnostics

IMPORTANT:
    This file does NOT intentionally change the V13 trading rules.

    It is designed to determine whether:
        1. indicators are not being populated,
        2. V13 signal conditions are too restrictive,
        3. signals exist but position sizing rejects them,
        4. or the data window is insufficient.

Yahoo currently limits 1-minute historical data to short windows.
Therefore this backtest uses 7 days and diagnoses the higher-timeframe
indicator availability rather than pretending 60 days of 1-minute data
is available.
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

# Yahoo 1m historical data is limited.
PERIOD = "7d"
INTERVAL = "1m"

STARTING_CAPITAL = 1000.0

SLIPPAGE_PCT = 0.0005
COMMISSION_PER_TRADE = 0.0

RISK_PER_TRADE_PCT = 0.0075
MAX_POSITION_PCT = 0.25

# Keep this deliberately modest for a 7-day 1m test.
WARMUP_BARS = 500


# ============================================================================
# DATA
# ============================================================================

def download_data(symbol: str) -> pd.DataFrame:

    print(f"[{symbol:<5}] downloading {PERIOD} of 1m data...")

    try:
        df = yf.download(
            symbol,
            period=PERIOD,
            interval=INTERVAL,
            auto_adjust=False,
            progress=False,
            threads=False,
        )
    except Exception as exc:
        print(f"[{symbol:<5}] download error: {exc}")
        return pd.DataFrame()

    if df is None or df.empty:
        return pd.DataFrame()

    # ------------------------------------------------------------------
    # Flatten yfinance MultiIndex.
    # ------------------------------------------------------------------

    if isinstance(df.columns, pd.MultiIndex):

        # Typical yfinance format:
        # ('Open', 'SPY'), ('High', 'SPY'), ...
        if symbol in df.columns.get_level_values(-1):

            try:
                df = df.xs(
                    symbol,
                    level=-1,
                    axis=1,
                )
            except Exception:
                pass

        elif symbol in df.columns.get_level_values(0):

            try:
                df = df.xs(
                    symbol,
                    level=0,
                    axis=1,
                )
            except Exception:
                pass

        # If still MultiIndex, flatten names.
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = [
                str(column[0])
                for column in df.columns
            ]

    required = [
        "Open",
        "High",
        "Low",
        "Close",
        "Volume",
    ]

    missing = [
        column
        for column in required
        if column not in df.columns
    ]

    if missing:
        print(
            f"[{symbol:<5}] missing columns: {missing}"
        )
        return pd.DataFrame()

    df = df[required].copy()

    for column in required:

        df[column] = pd.to_numeric(
            df[column],
            errors="coerce",
        )

    df.dropna(inplace=True)

    df.sort_index(inplace=True)

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

    print("          calculating V13 indicators...")

    try:
        df = calculate_indicators(df.copy())
    except Exception as exc:
        print(
            f"          indicator error: {exc}"
        )
        return pd.DataFrame()

    if df.empty:
        return df

    # ------------------------------------------------------------------
    # Diagnostic / previous-value columns.
    #
    # These are only added if the underlying columns exist.
    # ------------------------------------------------------------------

    if "RSI_1M" in df.columns:
        df["PREV_RSI_1M"] = (
            df["RSI_1M"].shift(1)
        )

    if "MACD_DIFF_15M" in df.columns:
        df["PREV_MACD_DIFF_15M"] = (
            df["MACD_DIFF_15M"].shift(1)
        )

    if "EMA20_4H" in df.columns:
        df["PREV_EMA20_4H"] = (
            df["EMA20_4H"].shift(1)
        )

    df["PREV_5_HIGH"] = (
        df["High"]
        .shift(1)
        .rolling(5)
        .max()
    )

    return df


# ============================================================================
# INDICATOR DIAGNOSTICS
# ============================================================================

def indicator_diagnostics(
    symbol: str,
    df: pd.DataFrame,
) -> None:

    print()
    print(
        f"          [{symbol}] INDICATOR DIAGNOSTICS"
    )

    expected = [
        "ATR_1M",
        "RSI_1M",
        "MACD_DIFF_15M",
        "EMA20_4H",
    ]

    for column in expected:

        if column not in df.columns:

            print(
                f"          {column:<18} MISSING"
            )

            continue

        valid = (
            df[column]
            .replace(
                [np.inf, -np.inf],
                np.nan,
            )
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
            f"          {column:<18} "
            f"{valid:>5}/{total:<5} valid "
            f"({pct:>5.1f}%)"
        )


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
# RESULT FACTORY
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
        "pf_display": "0.00",
        "avg_r": 0.0,
        "max_drawdown": 0.0,
        "ambiguous_bars": 0,
        "forced_exits": 0,

        # Diagnostics.
        "rows_checked": 0,
        "valid_atr": 0,
        "signals_valid": 0,
        "signals_invalid": 0,
        "exit_errors": 0,
        "zero_qty": 0,
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

    in_trade = False

    qty = 0
    entry_price = 0.0
    stop_price = 0.0
    target_price = 0.0
    risk_dollars = 0.0

    ambiguous_bars = 0
    forced_exits = 0

    rows_checked = 0
    valid_atr = 0
    signals_valid = 0
    signals_invalid = 0
    exit_errors = 0
    zero_qty = 0

    equity_curve = [
        equity
    ]

    # ==================================================================
    # MAIN LOOP
    # ==================================================================

    for i in range(
        WARMUP_BARS,
        len(df) - 1,
    ):

        row = df.iloc[i]

        rows_checked += 1

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
            # Ambiguous bar.
            # Conservative assumption:
            # stop first.
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
            # Target.
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
            # Stop.
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
        # NEW ENTRY
        # ==============================================================

        if not in_trade:

            # ----------------------------------------------------------
            # ATR diagnostic.
            # ----------------------------------------------------------

            if "ATR_1M" in df.columns:

                atr_value = row["ATR_1M"]

                if pd.notna(atr_value):

                    try:
                        atr_float = float(
                            atr_value
                        )

                        if (
                            np.isfinite(atr_float)
                            and atr_float > 0
                        ):
                            valid_atr += 1

                    except Exception:
                        pass

            # ----------------------------------------------------------
            # Evaluate V13 signal.
            # ----------------------------------------------------------

            try:
                signal = evaluate_signal(row)
            except Exception as exc:

                signals_invalid += 1

                if signals_invalid <= 3:

                    print(
                        f"          SIGNAL ERROR "
                        f"at row {i}: {exc}"
                    )

                equity_curve.append(equity)

                continue

            if not signal.valid:

                signals_invalid += 1

                equity_curve.append(equity)

                continue

            # ----------------------------------------------------------
            # V13 generated a valid signal.
            # ----------------------------------------------------------

            signals_valid += 1

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

            entry = (
                raw_entry
                * (1 + SLIPPAGE_PCT)
            )

            # ----------------------------------------------------------
            # ATR.
            # ----------------------------------------------------------

            try:
                atr = float(
                    row["ATR_1M"]
                )
            except Exception:

                exit_errors += 1

                equity_curve.append(equity)

                continue

            if (
                not np.isfinite(atr)
                or atr <= 0
            ):

                exit_errors += 1

                equity_curve.append(equity)

                continue

            # ----------------------------------------------------------
            # Exit prices.
            # ----------------------------------------------------------

            try:

                exits = calculate_exit_prices(
                    entry,
                    atr,
                )

                stop = float(
                    exits["stop"]
                )

                target = float(
                    exits["target"]
                )

            except Exception as exc:

                exit_errors += 1

                if exit_errors <= 3:

                    print(
                        f"          EXIT ERROR "
                        f"at row {i}: {exc}"
                    )

                equity_curve.append(equity)

                continue

            if (
                not np.isfinite(stop)
                or not np.isfinite(target)
                or stop <= 0
                or target <= entry
            ):

                exit_errors += 1

                equity_curve.append(equity)

                continue

            # ----------------------------------------------------------
            # Position size.
            # ----------------------------------------------------------

            qty_candidate = (
                calculate_position_size(
                    equity=equity,
                    entry_price=entry,
                    stop_price=stop,
                )
            )

            if qty_candidate <= 0:

                zero_qty += 1

                equity_curve.append(equity)

                continue

            # ----------------------------------------------------------
            # Never exceed available cash.
            # ----------------------------------------------------------

            max_cash_qty = int(
                equity / entry
            )

            qty_candidate = min(
                qty_candidate,
                max_cash_qty,
            )

            if qty_candidate <= 0:

                zero_qty += 1

                equity_curve.append(equity)

                continue

            # ----------------------------------------------------------
            # Open trade.
            # ----------------------------------------------------------

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

    result = calculate_statistics(
        symbol,
        trades,
        equity_curve,
        ambiguous_bars,
        forced_exits,
    )

    # Diagnostics.
    result["rows_checked"] = rows_checked
    result["valid_atr"] = valid_atr
    result["signals_valid"] = signals_valid
    result["signals_invalid"] = signals_invalid
    result["exit_errors"] = exit_errors
    result["zero_qty"] = zero_qty

    return result


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
            [
                trade["r"]
                for trade in trades
            ]
        )
        if trades
        else 0.0
    )

    # ------------------------------------------------------------------
    # Maximum drawdown.
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

    if profit_factor == float("inf"):

        pf_display = "inf"

    else:

        pf_display = (
            f"{profit_factor:.2f}"
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
        "max_drawdown": (
            max_drawdown * 100
        ),
        "ambiguous_bars": ambiguous_bars,
        "forced_exits": forced_exits,
    }


# ============================================================================
# MAIN
# ============================================================================

def main() -> None:

    print("=" * 78)

    print(
        "  BULL. BEAR AND BROKE — UNIFIED STRATEGY BACKTEST v13.1"
    )

    print(
        "  4H regime | 15M confirmation | 1M execution | ATR exits"
    )

    print(
        f"  Yahoo test window: {PERIOD}"
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

        print(
            f"[{symbol:<5}] "
            f"{len(raw):,} bars | "
            f"{raw.index[0]} -> {raw.index[-1]}"
        )

        df = prepare_features(raw)

        if df.empty:

            print(
                f"[{symbol:<5}] insufficient features"
            )

            results.append(
                empty_result(symbol)
            )

            continue

        indicator_diagnostics(
            symbol,
            df,
        )

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

        print(
            f"          DIAGNOSTICS: "
            f"rows={result['rows_checked']} | "
            f"ATR-valid={result['valid_atr']} | "
            f"valid-signals={result['signals_valid']} | "
            f"invalid-signals={result['signals_invalid']} | "
            f"exit-errors={result['exit_errors']} | "
            f"zero-qty={result['zero_qty']}"
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

    if gross_loss > 0:

        portfolio_pf = (
            gross_profit
            / gross_loss
        )

    elif gross_profit > 0:

        portfolio_pf = float("inf")

    else:

        portfolio_pf = 0.0

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

    valid_signals = sum(
        result["signals_valid"]
        for result in results
    )

    invalid_signals = sum(
        result["signals_invalid"]
        for result in results
    )

    zero_qty = sum(
        result["zero_qty"]
        for result in results
    )

    exit_errors = sum(
        result["exit_errors"]
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

    if portfolio_pf == float("inf"):

        print(
            "PROFIT FACTOR: inf"
        )

    else:

        print(
            f"PROFIT FACTOR: "
            f"{portfolio_pf:.3f}"
        )

    print(
        f"VALID V13 SIGNALS: "
        f"{valid_signals}"
    )

    print(
        f"INVALID SIGNAL CHECKS: "
        f"{invalid_signals}"
    )

    print(
        f"ZERO POSITION-SIZE SIGNALS: "
        f"{zero_qty}"
    )

    print(
        f"EXIT CALCULATION ERRORS: "
        f"{exit_errors}"
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

    # ==================================================================
    # DIAGNOSIS
    # ==================================================================

    if total_trades == 0:

        print(
            "❌ NO TRADES WERE GENERATED."
        )

        if valid_signals == 0:

            print(
                "   V13 generated ZERO valid signals."
            )

            print(
                "   This means the problem is upstream of execution."
            )

            print(
                "   Inspect the V13 indicator/signal conditions."
            )

        elif zero_qty > 0:

            print(
                "   V13 generated valid signals, "
                "but position sizing rejected them."
            )

            print(
                "   The likely issue is the $1,000 account "
                "combined with risk/position constraints."
            )

        elif exit_errors > 0:

            print(
                "   V13 generated signals, "
                "but exit prices could not be calculated."
            )

        else:

            print(
                "   Signals existed but were not converted "
                "into completed trades."
            )

    elif total_expectancy > 0 and portfolio_pf > 1:

        print(
            "✅ V13.1 produced positive historical expectancy."
        )

        print(
            "   This is NOT proof of future profitability."
        )

    else:

        print(
            "⚠️ V13 generated trades, "
            "but the tested result is not profitable."
        )

    print(
        "IMPORTANT: This is a short historical 1-minute test."
    )

    print(
        "Do NOT compare it directly with V12's larger sample "
        "until the testing windows are made comparable."
    )

    print("=" * 78)


if __name__ == "__main__":
    main()
