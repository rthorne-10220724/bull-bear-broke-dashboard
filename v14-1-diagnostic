"""
BULL. BEAR AND BROKE - V14.1 DIAGNOSTIC BACKTEST

4H REGIME | 15M MOMENTUM | 1M PULLBACK/RECLAIM

V14.1 purpose:
- Preserve the V14 strategy rules.
- Diagnose position sizing instead of silently losing valid signals.
- Add fractional-share diagnostic mode so $1,000 accounts can actually
  express risk on higher-priced symbols.
- Report the difference between signal generation, sizing rejection,
  and executed trades.
- Keep the original risk and max-position percentages unchanged.
- Do NOT change evaluate_signal(), calculate_exit_prices(), timeframe
  construction, entry timing, ATR exits, or signal thresholds.

IMPORTANT:
This is a diagnostic backtest, not a claim of live-trading suitability.
"""

from __future__ import annotations

import math
from typing import Any, Dict, List

import numpy as np
import pandas as pd
import yfinance as yf

from strategy import (
    align_timeframes,
    calculate_exit_prices,
    evaluate_signal,
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

# Yahoo intraday-safe windows.
EXECUTION_PERIOD = "7d"
CONFIRMATION_PERIOD = "59d"
REGIME_PERIOD = "180d"

STARTING_CAPITAL = 1000.0

SLIPPAGE_PCT = 0.0005
COMMISSION_PER_TRADE = 0.0

# ORIGINAL V14 VALUES — UNCHANGED.
RISK_PER_TRADE_PCT = 0.0075
MAX_POSITION_PCT = 0.25

WARMUP_BARS = 100

# ---------------------------------------------------------------------------
# V14.1 DIAGNOSTIC SIZING
#
# Fractional shares are used ONLY so the backtest can test the signal/exit
# logic with a small account. The risk budget and 25% position cap remain
# exactly the same.
#
# Set False to reproduce original whole-share sizing.
# ---------------------------------------------------------------------------
DIAGNOSTIC_FRACTIONAL_SHARES = True

# If False, the backtest will use original integer-share sizing.
# If True, quantity is allowed to be fractional to 4 decimal places.
SHARE_DECIMALS = 4


# ============================================================================
# DOWNLOAD
# ============================================================================

def download_data(
    symbol: str,
    period: str,
    interval: str,
) -> pd.DataFrame:

    try:
        df = yf.download(
            symbol,
            period=period,
            interval=interval,
            auto_adjust=False,
            progress=False,
            prepost=False,
            threads=False,
        )
    except Exception as exc:
        print(
            f"[{symbol:<5}] DOWNLOAD ERROR "
            f"{interval}: {exc}"
        )
        return pd.DataFrame()

    if df is None or df.empty:
        return pd.DataFrame()

    # Flatten yfinance MultiIndex.
    if isinstance(df.columns, pd.MultiIndex):
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
            else:
                df.columns = [column[0] for column in df.columns]
        except Exception as exc:
            print(
                f"[{symbol:<5}] "
                f"MultiIndex error: {exc}"
            )
            return pd.DataFrame()

    required = [
        "Open",
        "High",
        "Low",
        "Close",
        "Volume",
    ]

    if not all(column in df.columns for column in required):
        print(
            f"[{symbol:<5}] "
            f"{interval} missing OHLCV columns"
        )
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

    if df.empty:
        return df

    index = pd.DatetimeIndex(df.index)

    if index.tz is None:
        index = index.tz_localize("America/New_York")
    else:
        index = index.tz_convert("America/New_York")

    df.index = index
    df.sort_index(inplace=True)

    df = df[
        ~df.index.duplicated(keep="last")
    ]

    return df


# ============================================================================
# 4H CONSTRUCTION
# ============================================================================

def build_4h_from_1h(
    df_1h: pd.DataFrame,
) -> pd.DataFrame:

    if df_1h.empty:
        return pd.DataFrame()

    df = df_1h.copy()

    # Regular market hours only.
    df = df.between_time(
        "09:30",
        "16:00",
        inclusive="both",
    )

    if df.empty:
        return pd.DataFrame()

    # Anchor 4H buckets to 09:30 ET.
    result = (
        df.resample(
            "4h",
            origin="start_day",
            offset="9h30min",
            label="left",
            closed="left",
        )
        .agg(
            {
                "Open": "first",
                "High": "max",
                "Low": "min",
                "Close": "last",
                "Volume": "sum",
            }
        )
    )

    result = result.dropna(
        subset=[
            "Open",
            "High",
            "Low",
            "Close",
        ]
    )

    result = result[
        result.index.time
        >= pd.Timestamp("09:30").time()
    ]

    return result


# ============================================================================
# POSITION SIZE — V14 ORIGINAL
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

    risk_per_share = entry_price - stop_price

    if risk_per_share <= 0:
        return 0

    risk_dollars = equity * RISK_PER_TRADE_PCT

    qty_by_risk = risk_dollars / risk_per_share

    max_position_dollars = equity * MAX_POSITION_PCT

    qty_by_cap = max_position_dollars / entry_price

    qty = min(
        qty_by_risk,
        qty_by_cap,
    )

    return max(
        0,
        int(qty),
    )


# ============================================================================
# POSITION SIZE — V14.1 DIAGNOSTIC
# ============================================================================

def calculate_diagnostic_position_size(
    equity: float,
    entry_price: float,
    stop_price: float,
) -> Dict[str, float]:

    result = {
        "risk_per_share": 0.0,
        "risk_dollars": 0.0,
        "qty_by_risk": 0.0,
        "max_position_dollars": 0.0,
        "qty_by_cap": 0.0,
        "qty_original_integer": 0.0,
        "qty_diagnostic": 0.0,
        "position_dollars": 0.0,
        "risk_used": 0.0,
    }

    if (
        equity <= 0
        or entry_price <= 0
        or stop_price <= 0
    ):
        return result

    risk_per_share = entry_price - stop_price

    if risk_per_share <= 0:
        return result

    risk_dollars = equity * RISK_PER_TRADE_PCT
    qty_by_risk = risk_dollars / risk_per_share

    max_position_dollars = equity * MAX_POSITION_PCT
    qty_by_cap = max_position_dollars / entry_price

    raw_qty = min(
        qty_by_risk,
        qty_by_cap,
    )

    original_qty = max(0, int(raw_qty))

    if DIAGNOSTIC_FRACTIONAL_SHARES:
        diagnostic_qty = round(
            max(0.0, raw_qty),
            SHARE_DECIMALS,
        )
    else:
        diagnostic_qty = float(original_qty)

    result.update(
        {
            "risk_per_share": risk_per_share,
            "risk_dollars": risk_dollars,
            "qty_by_risk": qty_by_risk,
            "max_position_dollars": max_position_dollars,
            "qty_by_cap": qty_by_cap,
            "qty_original_integer": float(original_qty),
            "qty_diagnostic": diagnostic_qty,
            "position_dollars": diagnostic_qty * entry_price,
            "risk_used": diagnostic_qty * risk_per_share,
        }
    )

    return result


# ============================================================================
# EMPTY RESULT
# ============================================================================

def empty_result(
    symbol: str,
    status: str = "NO_DATA",
) -> Dict[str, Any]:

    return {
        "symbol": symbol,
        "status": status,
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
        "sizing_rejected": 0,
        "diagnostic_eligible": 0,
        "original_integer_qty_zero": 0,
        "cash_rejected": 0,
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
        return empty_result(
            symbol,
            "NO_ALIGNED_DATA",
        )

    if len(df) <= WARMUP_BARS + 1:
        return empty_result(
            symbol,
            "INSUFFICIENT_BARS",
        )

    equity = STARTING_CAPITAL

    trades: List[Dict[str, Any]] = []
    equity_curve = [equity]

    in_trade = False

    qty = 0.0

    entry_price = 0.0
    stop_price = 0.0
    target_price = 0.0
    risk_dollars = 0.0

    ambiguous_bars = 0
    forced_exits = 0

    signal_valid = 0
    signal_invalid = 0

    # Original V14 whole-share zero quantity.
    original_integer_qty_zero = 0

    # V14.1 sizing diagnostics.
    sizing_rejected = 0
    diagnostic_eligible = 0
    cash_rejected = 0

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

            hit_stop = low <= stop_price
            hit_target = high >= target_price

            # Conservative ambiguity: stop first.
            if hit_stop and hit_target:

                ambiguous_bars += 1

                exit_price = stop_price * (
                    1 - SLIPPAGE_PCT
                )

                pnl = (
                    qty
                    * (exit_price - entry_price)
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

            # TARGET
            if hit_target:

                exit_price = target_price * (
                    1 - SLIPPAGE_PCT
                )

                pnl = (
                    qty
                    * (exit_price - entry_price)
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

            # STOP
            if hit_stop:

                exit_price = stop_price * (
                    1 - SLIPPAGE_PCT
                )

                pnl = (
                    qty
                    * (exit_price - entry_price)
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

        # ====================================================================
        # NEW ENTRY
        # ====================================================================

        if not in_trade:

            # IMPORTANT: EXACT SAME SIGNAL FUNCTION AS V14.
            signal = evaluate_signal(row)

            if not signal.valid:

                signal_invalid += 1
                equity_curve.append(equity)
                continue

            signal_valid += 1

            # Same next-1M-open execution as V14.
            next_row = df.iloc[i + 1]

            raw_entry = float(next_row["Open"])

            if not np.isfinite(raw_entry):
                equity_curve.append(equity)
                continue

            entry = raw_entry * (
                1 + SLIPPAGE_PCT
            )

            atr = float(row["ATR_1M"])

            if (
                not np.isfinite(atr)
                or atr <= 0
            ):
                signal_invalid += 1
                equity_curve.append(equity)
                continue

            # EXACT SAME EXIT FUNCTION AS V14.
            try:
                exits = calculate_exit_prices(
                    entry,
                    atr,
                )
            except Exception:
                exit_errors += 1
                equity_curve.append(equity)
                continue

            stop = exits["stop"]
            target = exits["target"]

            # ----------------------------------------------------------------
            # DIAGNOSTIC SIZING
            # ----------------------------------------------------------------

            sizing = calculate_diagnostic_position_size(
                equity=equity,
                entry_price=entry,
                stop_price=stop,
            )

            original_qty = int(
                sizing["qty_original_integer"]
            )

            diagnostic_qty = sizing["qty_diagnostic"]

            if original_qty <= 0:
                original_integer_qty_zero += 1

            # A signal is only a sizing rejection if the mathematical
            # position size is actually zero, not merely because the
            # original integer conversion rounded it down.
            if diagnostic_qty <= 0:
                sizing_rejected += 1
                equity_curve.append(equity)
                continue

            diagnostic_eligible += 1

            # Never exceed available cash.
            max_cash_qty = (
                equity / entry
            )

            if DIAGNOSTIC_FRACTIONAL_SHARES:
                qty_candidate = min(
                    diagnostic_qty,
                    max_cash_qty,
                )
                qty_candidate = round(
                    qty_candidate,
                    SHARE_DECIMALS,
                )
            else:
                qty_candidate = min(
                    float(original_qty),
                    float(int(max_cash_qty)),
                )

            if qty_candidate <= 0:
                cash_rejected += 1
                equity_curve.append(equity)
                continue

            qty = qty_candidate

            entry_price = entry
            stop_price = stop
            target_price = target

            # Actual risk of the executed position.
            risk_dollars = (
                qty
                * (entry_price - stop_price)
            )

            in_trade = True

    # =========================================================================
    # FORCE END-OF-DATA EXIT
    # =========================================================================

    if in_trade:

        final_close = float(
            df["Close"].iloc[-1]
        )

        exit_price = final_close * (
            1 - SLIPPAGE_PCT
        )

        pnl = (
            qty
            * (exit_price - entry_price)
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
                    "WIN" if pnl > 0 else "LOSS"
                ),
            }
        )

        equity += pnl
        equity_curve.append(equity)
        forced_exits += 1

    return calculate_statistics(
        symbol=symbol,
        trades=trades,
        equity_curve=equity_curve,
        ambiguous_bars=ambiguous_bars,
        forced_exits=forced_exits,
        signal_valid=signal_valid,
        signal_invalid=signal_invalid,
        zero_qty=sizing_rejected,
        sizing_rejected=sizing_rejected,
        diagnostic_eligible=diagnostic_eligible,
        original_integer_qty_zero=original_integer_qty_zero,
        cash_rejected=cash_rejected,
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
    sizing_rejected: int,
    diagnostic_eligible: int,
    original_integer_qty_zero: int,
    cash_rejected: int,
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
            gross_profit / gross_loss
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

    # Drawdown.
    peak = -math.inf
    max_drawdown = 0.0

    for value in equity_curve:

        peak = max(peak, value)

        if peak > 0:
            drawdown = (
                (value - peak) / peak
            )
        else:
            drawdown = 0.0

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
        "status": "OK",
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
        "sizing_rejected": sizing_rejected,
        "diagnostic_eligible": diagnostic_eligible,
        "original_integer_qty_zero": original_integer_qty_zero,
        "cash_rejected": cash_rejected,
        "exit_errors": exit_errors,
    }


# ============================================================================
# MAIN
# ============================================================================

def main() -> None:

    print("=" * 82)
    print("  BULL. BEAR AND BROKE — V14.1 DIAGNOSTIC BACKTEST")
    print("  4H REGIME | 15M MOMENTUM | 1M PULLBACK/RECLAIM")
    print("=" * 82)

    print(
        f"1M execution window: {EXECUTION_PERIOD}"
    )
    print(
        f"15M confirmation window: {CONFIRMATION_PERIOD}"
    )
    print(
        f"1H regime source window: {REGIME_PERIOD}"
    )
    print("Yahoo intraday-safe architecture enabled.")

    print("-" * 82)
    print("V14 RULES: UNCHANGED")
    print(
        f"Starting capital:       ${STARTING_CAPITAL:.2f}"
    )
    print(
        f"Risk per trade:         {RISK_PER_TRADE_PCT:.2%}"
    )
    print(
        f"Max position:           {MAX_POSITION_PCT:.2%}"
    )
    print(
        f"Fractional diagnostics: "
        f"{'ON' if DIAGNOSTIC_FRACTIONAL_SHARES else 'OFF'}"
    )
    print("=" * 82)

    results: List[Dict[str, Any]] = []

    for symbol in TICKERS:

        # ====================================================================
        # 1M
        # ====================================================================

        print(
            f"[{symbol:<5}] 1M execution data..."
        )

        one_minute = download_data(
            symbol,
            EXECUTION_PERIOD,
            "1m",
        )

        if one_minute.empty:

            print(
                f"[{symbol:<5}] ❌ NO 1M DATA"
            )

            results.append(
                empty_result(
                    symbol,
                    "NO_1M_DATA",
                )
            )

            continue

        print(
            f"          {len(one_minute):,} 1M bars"
        )

        # ====================================================================
        # 15M
        # ====================================================================

        print(
            "          15M confirmation data..."
        )

        fifteen_minute = download_data(
            symbol,
            CONFIRMATION_PERIOD,
            "15m",
        )

        if fifteen_minute.empty:

            print(
                f"[{symbol:<5}] ❌ NO 15M DATA"
            )

            results.append(
                empty_result(
                    symbol,
                    "NO_15M_DATA",
                )
            )

            continue

        print(
            f"          {len(fifteen_minute):,} 15M bars"
        )

        # ====================================================================
        # 1H
        # ====================================================================

        print(
            "          1H data for 4H regime..."
        )

        one_hour = download_data(
            symbol,
            REGIME_PERIOD,
            "1h",
        )

        if one_hour.empty:

            print(
                f"[{symbol:<5}] ❌ NO 1H DATA"
            )

            results.append(
                empty_result(
                    symbol,
                    "NO_1H_DATA",
                )
            )

            continue

        print(
            f"          {len(one_hour):,} 1H bars"
        )

        # ====================================================================
        # BUILD 4H
        # ====================================================================

        four_hour = build_4h_from_1h(
            one_hour
        )

        if four_hour.empty:

            print(
                f"[{symbol:<5}] "
                f"❌ COULD NOT BUILD 4H DATA"
            )

            results.append(
                empty_result(
                    symbol,
                    "NO_4H_DATA",
                )
            )

            continue

        print(
            f"          {len(four_hour):,} "
            f"constructed 4H bars"
        )

        # ====================================================================
        # ALIGN
        # ====================================================================

        print(
            "          Aligning timeframes..."
        )

        try:

            df = align_timeframes(
                df_1m=one_minute,
                df_15m=fifteen_minute,
                df_4h=four_hour,
            )

        except Exception as exc:

            print(
                f"[{symbol:<5}] "
                f"❌ ALIGNMENT ERROR: {exc}"
            )

            results.append(
                empty_result(
                    symbol,
                    "ALIGNMENT_ERROR",
                )
            )

            continue

        if df.empty:

            print(
                f"[{symbol:<5}] "
                f"❌ EMPTY ALIGNED DATA"
            )

            results.append(
                empty_result(
                    symbol,
                    "EMPTY_ALIGNED_DATA",
                )
            )

            continue

        # ====================================================================
        # DIAGNOSTICS
        # ====================================================================

        total_rows = len(df)

        required_columns = [
            "ATR_1M",
            "RSI_1M",
            "MACD_DIFF_15M",
            "EMA20_4H",
        ]

        print("          INDICATOR DIAGNOSTICS")

        for column in required_columns:

            valid = int(
                df[column].notna().sum()
            )

            percentage = (
                valid / total_rows * 100
            )

            print(
                f"          {column:<18}"
                f"{valid:>5}/{total_rows:<5} "
                f"({percentage:5.1f}%)"
            )

        # ====================================================================
        # BACKTEST
        # ====================================================================

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
            f"PF={result['pf_display']:>5} | "
            f"AvgR={result['avg_r']:>6.3f} | "
            f"DD={result['max_drawdown']:>6.2f}%"
        )

        print(
            f"          "
            f"signals={result['signal_valid']} "
            f"| rejected={result['signal_invalid']} "
            f"| original-zero-qty="
            f"{result['original_integer_qty_zero']} "
            f"| diagnostic-trades="
            f"{result['diagnostic_eligible']} "
            f"| cash-rejected="
            f"{result['cash_rejected']} "
            f"| exit-errors="
            f"{result['exit_errors']}"
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
            gross_profit / gross_loss
        )
    elif gross_profit > 0:
        portfolio_pf = float("inf")
    else:
        portfolio_pf = 0.0

    win_rate = (
        total_wins / total_trades * 100
        if total_trades
        else 0.0
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

    original_zero_qty = sum(
        result["original_integer_qty_zero"]
        for result in results
    )

    diagnostic_eligible = sum(
        result["diagnostic_eligible"]
        for result in results
    )

    cash_rejected = sum(
        result["cash_rejected"]
        for result in results
    )

    exit_errors = sum(
        result["exit_errors"]
        for result in results
    )

    ambiguous = sum(
        result["ambiguous_bars"]
        for result in results
    )

    forced = sum(
        result["forced_exits"]
        for result in results
    )

    data_errors = [
        result
        for result in results
        if result["status"] != "OK"
    ]

    print("=" * 82)
    print("  V14.1 DIAGNOSTIC PORTFOLIO SUMMARY")
    print("=" * 82)

    print(
        f"Trades:                    {total_trades}"
    )
    print(
        f"Wins / Losses:             "
        f"{total_wins} / {total_losses}"
    )
    print(
        f"Win rate:                  {win_rate:.2f}%"
    )
    print(
        f"Net P&L:                   ${total_pnl:.2f}"
    )
    print(
        f"Expectancy/trade:          "
        f"${total_expectancy:.2f}"
    )

    if portfolio_pf == float("inf"):
        print("Profit factor:             inf")
    else:
        print(
            f"Profit factor:             "
            f"{portfolio_pf:.3f}"
        )

    print("-" * 42)

    print(
        f"Valid signals:              {valid_signals}"
    )
    print(
        f"Rejected signals:           {invalid_signals}"
    )
    print(
        f"Original integer zero-qty:  {original_zero_qty}"
    )
    print(
        f"Diagnostic executable:      "
        f"{diagnostic_eligible}"
    )
    print(
        f"Cash-rejected:              {cash_rejected}"
    )
    print(
        f"Exit errors:                {exit_errors}"
    )
    print(
        f"Ambiguous TP/SL:            {ambiguous}"
    )
    print(
        f"Forced exits:               {forced}"
    )
    print(
        f"Data-error symbols:         {len(data_errors)}"
    )

    print("-" * 82)
    print(
        "DIAGNOSTIC INTERPRETATION:"
    )

    if valid_signals > 0:
        original_pct = (
            original_zero_qty
            / valid_signals
            * 100
        )
        diagnostic_pct = (
            diagnostic_eligible
            / valid_signals
            * 100
        )

        print(
            f"Original whole-share sizing would "
            f"discard {original_pct:.1f}% of valid signals."
        )
        print(
            f"V14.1 diagnostic sizing can execute "
            f"{diagnostic_pct:.1f}% of valid signals."
        )

    if total_trades == 0:

        print("❌ No executable trades.")
        print(
            "The diagnostic did not find a sizing "
            "path that could express the signals."
        )

    elif total_expectancy <= 0:

        print(
            "❌ V14 signal/exit logic is negative "
            "on this historical window under "
            "diagnostic sizing."
        )
        print(
            "This is now a much cleaner strategy "
            "diagnostic because the integer-share "
            "rounding problem has been isolated."
        )

    elif portfolio_pf <= 1.0:

        print(
            "⚠️ Positive P&L but profit factor "
            "<= 1.0."
        )

    else:

        print(
            "✅ Positive historical expectancy "
            "under diagnostic sizing."
        )

    if total_trades < 100:
        print(
            f"⚠️ Small sample: {total_trades} trades."
        )

    print("-" * 82)
    print("IMPORTANT:")
    print(
        "V14.1 does NOT change the signal rules "
        "or exit rules."
    )
    print(
        "Historical performance does not guarantee "
        "future results."
    )
    print(
        "Run an out-of-sample test before changing "
        "strategy parameters."
    )
    print("=" * 82)


if __name__ == "__main__":
    main()
