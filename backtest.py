"""
BULL. BEAR AND BROKE - V14 BACKTEST
===================================

DATA ARCHITECTURE:

    1M  -> execution
    15M -> momentum confirmation
    1H  -> source data for 4H regime

IMPORTANT:

Yahoo restricts historical intraday ranges.

Therefore:

    1M  = 7 days
    15M = 59 days
    1H  = 180 days

The 4H regime is constructed from actual 1H candles.

The backtester refuses to treat missing higher-timeframe data as
"zero trades".

Execution:

    Signal on completed 1M candle
    Entry on next 1M open
    Slippage
    ATR stop
    ATR target
    Conservative same-bar ambiguity
    Risk-based position sizing
    Forced end-of-data exit
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

# ---------------------------------------------------------------------------
# Yahoo intraday-safe windows.
# ---------------------------------------------------------------------------

EXECUTION_PERIOD = "7d"

# IMPORTANT:
# Do NOT change this to 180d for 15m.
# Yahoo rejected that request in the previous run.
CONFIRMATION_PERIOD = "59d"

# 1H has a longer usable history and is used to build 4H.
REGIME_PERIOD = "180d"


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

    # ------------------------------------------------------------------------
    # Flatten yfinance MultiIndex.
    # ------------------------------------------------------------------------

    if isinstance(
        df.columns,
        pd.MultiIndex,
    ):

        try:

            if symbol in (
                df.columns
                .get_level_values(-1)
            ):

                df = df.xs(
                    symbol,
                    level=-1,
                    axis=1,
                )

            elif symbol in (
                df.columns
                .get_level_values(0)
            ):

                df = df.xs(
                    symbol,
                    level=0,
                    axis=1,
                )

            else:

                df.columns = [
                    column[0]
                    for column in df.columns
                ]

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

    if not all(
        column in df.columns
        for column in required
    ):

        print(
            f"[{symbol:<5}] "
            f"{interval} missing OHLCV columns"
        )

        return pd.DataFrame()

    df = df[
        required
    ].copy()

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

    # ------------------------------------------------------------------------
    # Normalize timestamps.
    #
    # yfinance can return timezone-aware indexes.
    # Keep everything consistently timezone-aware.
    # ------------------------------------------------------------------------

    index = pd.DatetimeIndex(
        df.index
    )

    if index.tz is None:

        index = index.tz_localize(
            "America/New_York"
        )

    else:

        index = index.tz_convert(
            "America/New_York"
        )

    df.index = index

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
# 4H CONSTRUCTION
# ============================================================================

def build_4h_from_1h(
    df_1h: pd.DataFrame,
) -> pd.DataFrame:

    if df_1h.empty:
        return pd.DataFrame()

    df = df_1h.copy()

    # ------------------------------------------------------------------------
    # Regular market hours only.
    #
    # This prevents premarket/after-hours bars from contaminating the
    # 4H regime.
    # ------------------------------------------------------------------------

    df = df.between_time(
        "09:30",
        "16:00",
        inclusive="both",
    )

    if df.empty:
        return pd.DataFrame()

    # ------------------------------------------------------------------------
    # Anchor the 4H buckets to 09:30 ET.
    #
    # First bucket:
    #     09:30 -> 13:30
    #
    # Second bucket:
    #     13:30 -> 16:00
    #
    # The second bucket is shorter because the regular session ends.
    # It is retained as a legitimate completed session regime candle.
    # ------------------------------------------------------------------------

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

    # Remove buckets outside regular session.
    result = result[
        result.index.time
        >= pd.Timestamp(
            "09:30"
        ).time()
    ]

    return result


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

    if len(df) <= (
        WARMUP_BARS + 1
    ):

        return empty_result(
            symbol,
            "INSUFFICIENT_BARS",
        )

    equity = STARTING_CAPITAL

    trades: List[
        Dict[str, Any]
    ] = []

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

            # ----------------------------------------------------------------
            # Conservative ambiguity:
            # stop first if both are touched.
            # ----------------------------------------------------------------

            if (
                hit_stop
                and hit_target
            ):

                ambiguous_bars += 1

                exit_price = (
                    stop_price
                    * (
                        1
                        - SLIPPAGE_PCT
                    )
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

            # ----------------------------------------------------------------
            # TARGET
            # ----------------------------------------------------------------

            if hit_target:

                exit_price = (
                    target_price
                    * (
                        1
                        - SLIPPAGE_PCT
                    )
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

            # ----------------------------------------------------------------
            # STOP
            # ----------------------------------------------------------------

            if hit_stop:

                exit_price = (
                    stop_price
                    * (
                        1
                        - SLIPPAGE_PCT
                    )
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

            # ----------------------------------------------------------------
            # NEXT 1M BAR OPEN
            # ----------------------------------------------------------------

            next_row = df.iloc[
                i + 1
            ]

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

            entry = (
                raw_entry
                * (
                    1
                    + SLIPPAGE_PCT
                )
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

            # ----------------------------------------------------------------
            # EXITS
            # ----------------------------------------------------------------

            try:

                exits = (
                    calculate_exit_prices(
                        entry,
                        atr,
                    )
                )

            except Exception:

                exit_errors += 1

                equity_curve.append(
                    equity
                )

                continue

            stop = exits["stop"]
            target = exits["target"]

            # ----------------------------------------------------------------
            # POSITION SIZE
            # ----------------------------------------------------------------

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

            # Never buy more than available cash.
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
    # FORCE END-OF-DATA EXIT
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
    trades: List[
        Dict[str, Any]
    ],
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

        profit_factor = float(
            "inf"
        )

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
        else:

            drawdown = 0.0

        max_drawdown = min(
            max_drawdown,
            drawdown,
        )

    pf_display = (
        "inf"
        if profit_factor == float(
            "inf"
        )
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
        "max_drawdown": (
            max_drawdown * 100
        ),
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

    print("=" * 82)

    print(
        "  BULL. BEAR AND BROKE — V14 BACKTEST"
    )

    print(
        "  4H REGIME | 15M MOMENTUM | 1M PULLBACK/RECLAIM"
    )

    print("=" * 82)

    print(
        f"1M execution window: "
        f"{EXECUTION_PERIOD}"
    )

    print(
        f"15M confirmation window: "
        f"{CONFIRMATION_PERIOD}"
    )

    print(
        f"1H regime source window: "
        f"{REGIME_PERIOD}"
    )

    print(
        "Yahoo intraday-safe architecture enabled."
    )

    print("=" * 82)

    results: List[
        Dict[str, Any]
    ] = []

    for symbol in TICKERS:

        # ====================================================================
        # 1M
        # ====================================================================

        print(
            f"[{symbol:<5}] "
            f"1M execution data..."
        )

        one_minute = download_data(
            symbol,
            EXECUTION_PERIOD,
            "1m",
        )

        if one_minute.empty:

            print(
                f"[{symbol:<5}] "
                f"❌ NO 1M DATA"
            )

            results.append(
                empty_result(
                    symbol,
                    "NO_1M_DATA",
                )
            )

            continue

        print(
            f"          "
            f"{len(one_minute):,} 1M bars"
        )

        # ====================================================================
        # 15M
        # ====================================================================

        print(
            f"          "
            f"15M confirmation data..."
        )

        fifteen_minute = download_data(
            symbol,
            CONFIRMATION_PERIOD,
            "15m",
        )

        if fifteen_minute.empty:

            print(
                f"[{symbol:<5}] "
                f"❌ NO 15M DATA"
            )

            results.append(
                empty_result(
                    symbol,
                    "NO_15M_DATA",
                )
            )

            continue

        print(
            f"          "
            f"{len(fifteen_minute):,} "
            f"15M bars"
        )

        # ====================================================================
        # 1H
        # ====================================================================

        print(
            f"          "
            f"1H data for 4H regime..."
        )

        one_hour = download_data(
            symbol,
            REGIME_PERIOD,
            "1h",
        )

        if one_hour.empty:

            print(
                f"[{symbol:<5}] "
                f"❌ NO 1H DATA"
            )

            results.append(
                empty_result(
                    symbol,
                    "NO_1H_DATA",
                )
            )

            continue

        print(
            f"          "
            f"{len(one_hour):,} 1H bars"
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
            f"          "
            f"{len(four_hour):,} "
            f"constructed 4H bars"
        )

        # ====================================================================
        # ALIGN
        # ====================================================================

        print(
            f"          "
            f"Aligning timeframes..."
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
                f"❌ ALIGNMENT ERROR: "
                f"{exc}"
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

        print(
            f"          "
            f"INDICATOR DIAGNOSTICS"
        )

        for column in required_columns:

            valid = int(
                df[column]
                .notna()
                .sum()
            )

            percentage = (
                valid
                / total_rows
                * 100
            )

            print(
                f"          "
                f"{column:<18}"
                f"{valid:>5}/"
                f"{total_rows:<5} "
                f"({percentage:5.1f}%)"
            )

        # ====================================================================
        # BACKTEST
        # ====================================================================

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
            f"PF={result['pf_display']:>5} | "
            f"AvgR={result['avg_r']:>6.3f} | "
            f"DD={result['max_drawdown']:>6.2f}%"
        )

        print(
            f"          "
            f"signals={result['signal_valid']} "
            f"| rejected={result['signal_invalid']} "
            f"| zero-qty={result['zero_qty']} "
            f"| exit-errors={result['exit_errors']}"
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

    win_rate = (
        total_wins
        / total_trades
        * 100
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

    print(
        "  PORTFOLIO SUMMARY"
    )

    print("=" * 82)

    print(
        f"Trades:              "
        f"{total_trades}"
    )

    print(
        f"Wins / Losses:       "
        f"{total_wins} / {total_losses}"
    )

    print(
        f"Win rate:            "
        f"{win_rate:.2f}%"
    )

    print(
        f"Net P&L:             "
        f"${total_pnl:.2f}"
    )

    print(
        f"Expectancy/trade:    "
        f"${total_expectancy:.2f}"
    )

    if portfolio_pf == float(
        "inf"
    ):

        print(
            "Profit factor:      inf"
        )

    else:

        print(
            f"Profit factor:      "
            f"{portfolio_pf:.3f}"
        )

    print(
        "-" * 42
    )

    print(
        f"Valid signals:       "
        f"{valid_signals}"
    )

    print(
        f"Rejected signals:    "
        f"{invalid_signals}"
    )

    print(
        f"Zero position size:  "
        f"{zero_qty}"
    )

    print(
        f"Exit errors:         "
        f"{exit_errors}"
    )

    print(
        f"Ambiguous TP/SL:     "
        f"{ambiguous}"
    )

    print(
        f"Forced exits:        "
        f"{forced}"
    )

    print(
        f"Data-error symbols:  "
        f"{len(data_errors)}"
    )

    if data_errors:

        print(
            "Data statuses:"
        )

        for result in data_errors:

            print(
                f"  {result['symbol']}: "
                f"{result['status']}"
            )

    print("=" * 82)

    if total_trades == 0:

        print(
            "❌ NO TRADES."
        )

        print(
            "This is now a meaningful diagnostic:"
        )

        print(
            "If indicators are valid but signals=0, "
            "the strategy is too restrictive."
        )

    elif total_expectancy <= 0:

        print(
            "❌ V14 is NOT profitable "
            "on this historical window."
        )

    elif portfolio_pf <= 1.0:

        print(
            "⚠️ Positive P&L but "
            "profit factor <= 1.0."
        )

    else:

        print(
            "✅ V14 shows positive "
            "historical expectancy."
        )

    if total_trades < 100:

        print(
            f"⚠️ Small sample: "
            f"{total_trades} trades."
        )

    print(
        "IMPORTANT:"
    )

    print(
        "Historical performance does not "
        "guarantee future results."
    )

    print(
        "Run an out-of-sample test before "
        "changing parameters."
    )

    print("=" * 82)


if __name__ == "__main__":
    main()
