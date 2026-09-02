"""
BULL. BEAR AND BROKE - V13 BACKTEST
===================================

V13 BACKTESTER

Tests the existing V13 strategy.py signal engine using:

    - 1-minute execution data
    - 4H regime
    - 15M confirmation
    - shared strategy.py signal engine
    - next-bar execution
    - ATR stop
    - ATR target
    - slippage
    - commissions
    - conservative same-bar ambiguity handling
    - forced end-of-data exits
    - fixed-risk position sizing
    - maximum drawdown
    - R statistics

IMPORTANT:
    Yahoo Finance has strict limits on historical 1-minute data.
    Do NOT request 60d of 1m data.

    This backtester therefore uses a short 1-minute window that
    Yahoo can actually supply.

    This is a historical test, NOT a guarantee of future performance.
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

# IMPORTANT:
# Yahoo limits 1-minute historical data.
#
# Use a maximum of 7 days so we stay safely inside the
# commonly enforced 8-day-per-request limitation.
PERIOD = "7d"
INTERVAL = "1m"

STARTING_CAPITAL = 1000.0

# Buyer pays slightly more on entry.
# Seller receives slightly less on exit.
SLIPPAGE_PCT = 0.0005

COMMISSION_PER_TRADE = 0.0

# Risk 0.75% of current equity per trade.
RISK_PER_TRADE_PCT = 0.0075

# Never allocate more than 25% of equity to one position.
MAX_POSITION_PCT = 0.25

# Minimum number of bars before signals are considered.
WARMUP_BARS = 500


# ============================================================================
# DATA DOWNLOAD
# ============================================================================

def download_data(symbol: str) -> pd.DataFrame:
    """
    Download 1-minute OHLCV data from Yahoo Finance.

    IMPORTANT:
        Do not change PERIOD to 60d for 1-minute data.
        Yahoo does not provide that much 1-minute history in a
        normal request.
    """

    print(f"[{symbol:<5}] downloading {PERIOD} of {INTERVAL} data...")

    try:
        df = yf.download(
            symbol,
            period=PERIOD,
            interval=INTERVAL,
            auto_adjust=False,
            progress=False,
            prepost=False,
            threads=False,
        )
    except Exception as exc:
        print(f"[{symbol:<5}] download error: {exc}")
        return pd.DataFrame()

    if df is None or df.empty:
        print(f"[{symbol:<5}] no data returned")
        return pd.DataFrame()

    # ------------------------------------------------------------------------
    # Flatten yfinance MultiIndex columns.
    # ------------------------------------------------------------------------

    if isinstance(df.columns, pd.MultiIndex):

        # Typical yfinance layout:
        #
        # Open / High / Low / Close / Volume
        # with ticker as another level.
        #
        # Try to extract this symbol cleanly.

        try:

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

        except Exception:
            pass

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

    # ------------------------------------------------------------------------
    # Numeric conversion.
    # ------------------------------------------------------------------------

    for column in required:

        df[column] = pd.to_numeric(
            df[column],
            errors="coerce",
        )

    df.dropna(
        subset=required,
        inplace=True,
    )

    if df.empty:
        print(f"[{symbol:<5}] all rows invalid")
        return pd.DataFrame()

    # ------------------------------------------------------------------------
    # Timestamp cleanup.
    # ------------------------------------------------------------------------

    df.sort_index(
        inplace=True
    )

    df = df[
        ~df.index.duplicated(
            keep="last"
        )
    ]

    # Remove impossible prices.
    df = df[
        (df["Open"] > 0)
        & (df["High"] > 0)
        & (df["Low"] > 0)
        & (df["Close"] > 0)
    ]

    if df.empty:
        print(f"[{symbol:<5}] no valid price data")
        return pd.DataFrame()

    print(
        f"[{symbol:<5}] "
        f"{len(df):,} bars | "
        f"{df.index[0]} -> {df.index[-1]}"
    )

    return df


# ============================================================================
# FEATURE PREPARATION
# ============================================================================

def prepare_features(
    df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Run the exact shared V13 indicator engine.

    No strategy logic is duplicated here.
    """

    if df.empty:
        return df

    try:

        df = calculate_indicators(
            df.copy()
        )

    except Exception as exc:

        print(
            f"feature calculation error: {exc}"
        )

        return pd.DataFrame()

    if df.empty:
        return df

    # ------------------------------------------------------------------------
    # Previous values used by the signal engine.
    #
    # These are shifted one bar so the signal does not accidentally
    # use future information.
    # ------------------------------------------------------------------------

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

    # ------------------------------------------------------------------------
    # Previous completed 5-bar high.
    # ------------------------------------------------------------------------

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
    """
    Position size based on fixed account risk.

    Example:
        Equity = $1,000
        Risk = 0.75%
        Risk budget = $7.50

    The position is also capped at MAX_POSITION_PCT of equity.
    """

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
# TRADE EXIT HELPER
# ============================================================================

def close_trade(
    trades: List[Dict[str, Any]],
    symbol: str,
    qty: int,
    entry_price: float,
    exit_price: float,
    risk_dollars: float,
    outcome: str,
) -> float:
    """
    Record a completed trade and return its P&L.
    """

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
            "outcome": outcome,
        }
    )

    return pnl


# ============================================================================
# BACKTEST
# ============================================================================

def run_backtest(
    symbol: str,
    df: pd.DataFrame,
) -> Dict[str, Any]:

    if df.empty:
        return empty_result(symbol)

    if len(df) <= WARMUP_BARS + 1:

        print(
            f"[{symbol:<5}] "
            f"not enough bars "
            f"({len(df)} available)"
        )

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

    # =========================================================================
    # MAIN LOOP
    # =========================================================================

    for i in range(
        WARMUP_BARS,
        len(df) - 1,
    ):

        row = df.iloc[i]

        # =====================================================================
        # MANAGE OPEN POSITION
        # =====================================================================

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

            # -----------------------------------------------------------------
            # BOTH STOP AND TARGET TOUCHED.
            #
            # We do not know which occurred first inside a 1-minute OHLC bar.
            #
            # Conservative assumption:
            # STOP FIRST.
            # -----------------------------------------------------------------

            if hit_stop and hit_target:

                ambiguous_bars += 1

                exit_price = (
                    stop_price
                    * (
                        1
                        - SLIPPAGE_PCT
                    )
                )

                pnl = close_trade(
                    trades=trades,
                    symbol=symbol,
                    qty=qty,
                    entry_price=entry_price,
                    exit_price=exit_price,
                    risk_dollars=risk_dollars,
                    outcome="LOSS",
                )

                equity += pnl

                equity_curve.append(
                    equity
                )

                in_trade = False

                continue

            # -----------------------------------------------------------------
            # TARGET
            # -----------------------------------------------------------------

            if hit_target:

                exit_price = (
                    target_price
                    * (
                        1
                        - SLIPPAGE_PCT
                    )
                )

                pnl = close_trade(
                    trades=trades,
                    symbol=symbol,
                    qty=qty,
                    entry_price=entry_price,
                    exit_price=exit_price,
                    risk_dollars=risk_dollars,
                    outcome="WIN",
                )

                equity += pnl

                equity_curve.append(
                    equity
                )

                in_trade = False

                continue

            # -----------------------------------------------------------------
            # STOP
            # -----------------------------------------------------------------

            if hit_stop:

                exit_price = (
                    stop_price
                    * (
                        1
                        - SLIPPAGE_PCT
                    )
                )

                pnl = close_trade(
                    trades=trades,
                    symbol=symbol,
                    qty=qty,
                    entry_price=entry_price,
                    exit_price=exit_price,
                    risk_dollars=risk_dollars,
                    outcome="LOSS",
                )

                equity += pnl

                equity_curve.append(
                    equity
                )

                in_trade = False

                continue

        # =====================================================================
        # NEW ENTRY
        # =====================================================================

        if not in_trade:

            try:

                signal = evaluate_signal(
                    row
                )

            except Exception as exc:

                print(
                    f"[{symbol:<5}] "
                    f"signal error at "
                    f"{df.index[i]}: {exc}"
                )

                equity_curve.append(
                    equity
                )

                continue

            if not signal.valid:

                equity_curve.append(
                    equity
                )

                continue

            # -----------------------------------------------------------------
            # NEXT-BAR EXECUTION
            # -----------------------------------------------------------------

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

            # Buyer receives adverse slippage.
            entry = (
                raw_entry
                * (
                    1
                    + SLIPPAGE_PCT
                )
            )

            # -----------------------------------------------------------------
            # ATR
            # -----------------------------------------------------------------

            atr = float(
                row["ATR_1M"]
            )

            if (
                not np.isfinite(atr)
                or atr <= 0
            ):

                equity_curve.append(
                    equity
                )

                continue

            # -----------------------------------------------------------------
            # ATR STOP / TARGET
            # -----------------------------------------------------------------

            try:

                exits = calculate_exit_prices(
                    entry,
                    atr,
                )

            except Exception as exc:

                print(
                    f"[{symbol:<5}] "
                    f"exit calculation error: {exc}"
                )

                equity_curve.append(
                    equity
                )

                continue

            stop = float(
                exits["stop"]
            )

            target = float(
                exits["target"]
            )

            # -----------------------------------------------------------------
            # POSITION SIZE
            # -----------------------------------------------------------------

            qty_candidate = (
                calculate_position_size(
                    equity=equity,
                    entry_price=entry,
                    stop_price=stop,
                )
            )

            if qty_candidate <= 0:

                equity_curve.append(
                    equity
                )

                continue

            # -----------------------------------------------------------------
            # CASH LIMIT
            # -----------------------------------------------------------------

            max_cash_qty = int(
                equity / entry
            )

            qty_candidate = min(
                qty_candidate,
                max_cash_qty,
            )

            if qty_candidate <= 0:

                equity_curve.append(
                    equity
                )

                continue

            # -----------------------------------------------------------------
            # OPEN TRADE
            # -----------------------------------------------------------------

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

            if risk_dollars <= 0:

                equity_curve.append(
                    equity
                )

                continue

            in_trade = True

    # =========================================================================
    # FORCE CLOSE OPEN POSITION
    # =========================================================================

    if in_trade:

        final_close = float(
            df["Close"].iloc[-1]
        )

        exit_price = (
            final_close
            * (
                1
                - SLIPPAGE_PCT
            )
        )

        outcome = (
            "WIN"
            if exit_price > entry_price
            else "LOSS"
        )

        pnl = close_trade(
            trades=trades,
            symbol=symbol,
            qty=qty,
            entry_price=entry_price,
            exit_price=exit_price,
            risk_dollars=risk_dollars,
            outcome=outcome,
        )

        equity += pnl

        forced_exits += 1

        equity_curve.append(
            equity
        )

    # =========================================================================
    # STATISTICS
    # =========================================================================

    return calculate_statistics(
        symbol=symbol,
        trades=trades,
        equity_curve=equity_curve,
        ambiguous_bars=ambiguous_bars,
        forced_exits=forced_exits,
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
        len(wins)
        / total
        * 100
        if total
        else 0.0
    )

    net_pnl = sum(
        trade["pnl"]
        for trade in trades
    )

    expectancy = (
        net_pnl
        / total
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

        profit_factor = float(
            "inf"
        )

    else:

        profit_factor = 0.0

    avg_r = (
        float(
            np.mean(
                [
                    trade["r"]
                    for trade in trades
                ]
            )
        )
        if trades
        else 0.0
    )

    # =========================================================================
    # MAX DRAWDOWN
    # =========================================================================

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

    print(
        f"  Yahoo test window: {PERIOD}"
    )

    print("=" * 78)

    results: List[Dict[str, Any]] = []

    for symbol in TICKERS:

        raw = download_data(
            symbol
        )

        if raw.empty:

            results.append(
                empty_result(symbol)
            )

            continue

        df = prepare_features(
            raw
        )

        if df.empty:

            print(
                f"[{symbol:<5}] "
                f"insufficient features"
            )

            results.append(
                empty_result(symbol)
            )

            continue

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
        total_pnl
        / total_trades
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

        portfolio_pf = float(
            "inf"
        )

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

    print("-" * 78)

    print(
        f"TOTAL TRADES: "
        f"{total_trades}"
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
        f"AMBIGUOUS TP/SL BARS: "
        f"{ambiguous}"
    )

    print(
        f"FORCED END-OF-DATA EXITS: "
        f"{forced}"
    )

    print("=" * 78)

    # =========================================================================
    # INTERPRETATION
    # =========================================================================

    if total_trades == 0:

        print(
            "❌ NO TRADES WERE GENERATED."
        )

        print(
            "This is NOT evidence that V13 loses."
        )

        print(
            "Check strategy.py indicators/signals and the available "
            "Yahoo data window."
        )

    elif total_trades < 50:

        print(
            f"⚠️ Sample size is very small "
            f"({total_trades} trades)."
        )

        print(
            "Do NOT conclude that V13 beats or loses to V12 "
            "from this sample alone."
        )

    elif total_expectancy <= 0:

        print(
            "❌ V13 is negative expectancy "
            "on this test window."
        )

    elif portfolio_pf <= 1.0:

        print(
            "⚠️ V13 has positive P&L but "
            "profit factor <= 1.0."
        )

    else:

        print(
            "✅ V13 shows positive historical "
            "expectancy on this test window."
        )

    print(
        "IMPORTANT: Historical performance does not "
        "guarantee future performance."
    )

    print(
        "Use a separate out-of-sample test before "
        "judging V13."
    )

    print("=" * 78)


if __name__ == "__main__":
    main()
