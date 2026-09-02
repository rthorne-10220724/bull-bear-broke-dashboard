"""
BULL. BEAR AND BROKE - V13.2 BACKTEST
=====================================

V13.2 DATA FIX:

    4H regime  -> downloaded/calculated on real 4H candles
    15M signal -> downloaded/calculated on real 15M candles
    1M entry   -> downloaded separately

This avoids trying to create a 20-period 4H EMA from only seven days
of 1-minute data.

Execution:
    - Signal on completed 1M bar
    - Entry on next 1M bar open
    - ATR stop
    - ATR target
    - Slippage
    - Commission
    - Conservative same-bar ambiguity
    - Forced end-of-data exit
    - Risk-based position sizing

IMPORTANT:
    Historical performance is not a guarantee of future results.
"""

from __future__ import annotations

import math
from typing import Any, Dict, List

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

# Yahoo generally limits 1m historical data.
EXECUTION_PERIOD = "7d"

# Higher timeframes can go much farther back.
HIGHER_PERIOD = "180d"

STARTING_CAPITAL = 1000.0

SLIPPAGE_PCT = 0.0005

COMMISSION_PER_TRADE = 0.0

RISK_PER_TRADE_PCT = 0.0075

MAX_POSITION_PCT = 0.25

WARMUP_BARS = 100


# ============================================================================
# DOWNLOAD
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

        else:

            # Fallback to first ticker level.
            df.columns = [
                column[0]
                for column in df.columns
            ]

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
        entry_price
        - stop_price
    )

    if risk_per_share <= 0:
        return 0

    risk_dollars = (
        equity
        * RISK_PER_TRADE_PCT
    )

    qty_by_risk = (
        risk_dollars
        / risk_per_share
    )

    max_position_dollars = (
        equity
        * MAX_POSITION_PCT
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
# EMPTY RESULT
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
        "signal_valid": 0,
        "signal_invalid": 0,
        "zero_qty": 0,
        "exit_errors": 0,
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

    signal_valid = 0
    signal_invalid = 0
    zero_qty = 0
    exit_errors = 0

    for i in range(
        WARMUP_BARS,
        len(df) - 1,
    ):

        row = df.iloc[i]

        # ====================================================================
        # MANAGE OPEN POSITION
        # ====================================================================

        if in_trade:

            high = float(row["High"])
            low = float(row["Low"])

            hit_stop = (
                low <= stop_price
            )

            hit_target = (
                high >= target_price
            )

            # ---------------------------------------------------------------
            # Conservative ambiguity handling:
            # If both stop and target were touched in one candle,
            # assume stop happened first.
            # ---------------------------------------------------------------

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
                equity_curve.append(
                    equity
                )

                in_trade = False
                continue

            # ---------------------------------------------------------------
            # TARGET
            # ---------------------------------------------------------------

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
                equity_curve.append(
                    equity
                )

                in_trade = False
                continue

            # ---------------------------------------------------------------
            # STOP
            # ---------------------------------------------------------------

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
                equity_curve.append(
                    equity
                )

                in_trade = False
                continue

        # ====================================================================
        # NEW ENTRY
        # ====================================================================

        if not in_trade:

            signal = evaluate_signal(
                row
            )

            if not signal.valid:

                signal_invalid += 1

                equity_curve.append(
                    equity
                )

                continue

            signal_valid += 1

            # ---------------------------------------------------------------
            # NEXT-BAR EXECUTION
            # ---------------------------------------------------------------

            next_row = df.iloc[i + 1]

            raw_entry = float(
                next_row["Open"]
            )

            if not np.isfinite(
                raw_entry
            ):

                equity_curve.append(
                    equity
                )

                continue

            # Buyer pays adverse slippage.
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

                signal_invalid += 1

                equity_curve.append(
                    equity
                )

                continue

            # ---------------------------------------------------------------
            # EXITS
            # ---------------------------------------------------------------

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

            # ---------------------------------------------------------------
            # SIZE
            # ---------------------------------------------------------------

            qty_candidate = (
                calculate_position_size(
                    equity=equity,
                    entry_price=entry,
                    stop_price=stop,
                )
            )

            if qty_candidate <= 0:

                zero_qty += 1

                equity_curve.append(
                    equity
                )

                continue

            # Never exceed available cash.
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

    # ========================================================================
    # FORCE END-OF-DATA EXIT
    # ========================================================================

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

        equity_curve.append(
            equity
        )

        forced_exits += 1

    # ========================================================================
    # STATISTICS
    # ========================================================================

    return calculate_statistics(
        symbol=symbol,
        trades=trades,
        equity_curve=equity_curve,
        ambiguous_bars=ambiguous_bars,
        forced_exits=forced_exits,
        signal_valid=signal_valid,
        signal_invalid=signal_invalid,
        zero_qty=zero_qty,
        exit_errors=exit_errors,
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
    signal_valid: int,
    signal_invalid: int,
    zero_qty: int,
    exit_errors: int,
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

    # ------------------------------------------------------------------------
    # Drawdown
    # ------------------------------------------------------------------------

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
        "signal_valid": signal_valid,
        "signal_invalid": signal_invalid,
        "zero_qty": zero_qty,
        "exit_errors": exit_errors,
    }


# ============================================================================
# MAIN
# ============================================================================

def main() -> None:

    print("=" * 78)

    print(
        "  BULL. BEAR AND BROKE — UNIFIED STRATEGY BACKTEST v13.2"
    )

    print(
        "  4H regime | 15M confirmation | 1M execution | ATR exits"
    )

    print(
        f"  4H/15M history: {HIGHER_PERIOD}"
    )

    print(
        f"  1M execution window: {EXECUTION_PERIOD}"
    )

    print("=" * 78)

    results: List[Dict[str, Any]] = []

    for symbol in TICKERS:

        print(
            f"[{symbol:<5}] downloading 1M execution data..."
        )

        one_minute = download_data(
            symbol,
            EXECUTION_PERIOD,
            "1m",
        )

        if one_minute.empty:

            print(
                f"[{symbol:<5}] no 1M data"
            )

            results.append(
                empty_result(symbol)
            )

            continue

        print(
            f"[{symbol:<5}] "
            f"{len(one_minute):,} 1M bars | "
            f"{one_minute.index[0]} -> "
            f"{one_minute.index[-1]}"
        )

        print(
            f"          downloading 15M confirmation..."
        )

        fifteen_minute = download_data(
            symbol,
            HIGHER_PERIOD,
            "15m",
        )

        if fifteen_minute.empty:

            print(
                f"[{symbol:<5}] no 15M data"
            )

            results.append(
                empty_result(symbol)
            )

            continue

        print(
            f"          {len(fifteen_minute):,} 15M bars"
        )

        print(
            f"          downloading 4H regime..."
        )

        four_hour = download_data(
            symbol,
            HIGHER_PERIOD,
            "1h",
        )

        if four_hour.empty:

            print(
                f"[{symbol:<5}] no 1H data for 4H construction"
            )

            results.append(
                empty_result(symbol)
            )

            continue

        # --------------------------------------------------------------------
        # Build actual 4H candles from 1H candles.
        #
        # This gives us enough history for EMA20_4H.
        # --------------------------------------------------------------------

        four_hour = (
            four_hour
            .resample("4h")
            .agg(
                {
                    "Open": "first",
                    "High": "max",
                    "Low": "min",
                    "Close": "last",
                    "Volume": "sum",
                }
            )
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
            f"          {len(four_hour):,} 4H bars"
        )

        # --------------------------------------------------------------------
        # Align all timeframes.
        # --------------------------------------------------------------------

        print(
            f"          calculating V13.2 indicators..."
        )

        try:

            df = align_timeframes(
                df_1m=one_minute,
                df_15m=fifteen_minute,
                df_4h=four_hour,
            )

        except Exception as exc:

            print(
                f"[{symbol:<5}] indicator error: {exc}"
            )

            results.append(
                empty_result(symbol)
            )

            continue

        # --------------------------------------------------------------------
        # Diagnostics
        # --------------------------------------------------------------------

        total_rows = len(df)

        atr_valid = int(
            df["ATR_1M"]
            .notna()
            .sum()
        )

        rsi_valid = int(
            df["RSI_1M"]
            .notna()
            .sum()
        )

        macd_valid = int(
            df["MACD_DIFF_15M"]
            .notna()
            .sum()
        )

        ema_valid = int(
            df["EMA20_4H"]
            .notna()
            .sum()
        )

        print(
            f"          [{symbol}] INDICATOR DIAGNOSTICS"
        )

        print(
            f"          ATR_1M        "
            f"{atr_valid}/{total_rows} "
            f"valid "
            f"({atr_valid / total_rows * 100:5.1f}%)"
        )

        print(
            f"          RSI_1M        "
            f"{rsi_valid}/{total_rows} "
            f"valid "
            f"({rsi_valid / total_rows * 100:5.1f}%)"
        )

        print(
            f"          MACD_DIFF_15M "
            f"{macd_valid}/{total_rows} "
            f"valid "
            f"({macd_valid / total_rows * 100:5.1f}%)"
        )

        print(
            f"          EMA20_4H      "
            f"{ema_valid}/{total_rows} "
            f"valid "
            f"({ema_valid / total_rows * 100:5.1f}%)"
        )

        result = run_backtest(
            symbol,
            df,
        )

        results.append(
            result
        )

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
            f"valid-signals={result['signal_valid']} | "
            f"invalid-signals={result['signal_invalid']} | "
            f"zero-qty={result['zero_qty']} | "
            f"exit-errors={result['exit_errors']}"
        )

    # =========================================================================
    # PORTFOLIO SUMMARY
    # =========================================================================

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
        max(
            result["pnl"],
            0,
        )
        for result in results
    )

    gross_loss = abs(
        sum(
            min(
                result["pnl"],
                0,
            )
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
        result["signal_valid"]
        for result in results
    )

    invalid_signals = sum(
        result["signal_invalid"]
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

    print(
        f"PROFIT FACTOR: "
        f"{portfolio_pf:.3f}"
        if portfolio_pf != float("inf")
        else "PROFIT FACTOR: inf"
    )

    print(
        f"VALID V13.2 SIGNALS: "
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

    if total_trades == 0:

        print(
            "❌ NO TRADES WERE GENERATED."
        )

        print(
            "This time the diagnostics should tell us "
            "whether the problem is data or signal logic."
        )

    elif total_expectancy <= 0:

        print(
            "❌ V13.2 is NOT profitable on this test."
        )

    elif portfolio_pf <= 1.0:

        print(
            "⚠️ Positive P&L but PF <= 1.0."
        )

    else:

        print(
            "✅ V13.2 shows positive historical expectancy."
        )

    if total_trades < 100:

        print(
            f"⚠️ Sample size is small "
            f"({total_trades} trades)."
        )

    print(
        "IMPORTANT: Historical performance "
        "does not guarantee future results."
    )

    print(
        "Use a separate out-of-sample period "
        "before judging V13.2."
    )

    print("=" * 78)


if __name__ == "__main__":
    main()
