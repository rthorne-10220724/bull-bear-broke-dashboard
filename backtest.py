"""
BULL. BEAR AND BROKE - BACKTEST v12
===================================

Backtests the SAME strategy functions used by the v12 engine.

Core flow:

    historical 1m data
          |
          v
    completed 4H trend
          |
          v
    v12 indicator analysis
          |
          v
    v12 signal scoring
          |
          v
    NEXT-BAR execution
          |
          v
    ATR position sizing
          |
          v
    ATR stop / target
          |
          v
    realistic slippage
          |
          v
    performance statistics

IMPORTANT:
-----------
This is research code.

A positive backtest does not guarantee future profitability.

Run:
    - in-sample
    - out-of-sample
    - walk-forward
    - paper trading
    - realistic costs
"""

from __future__ import annotations

import os
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import yfinance as yf

from strategy import (
    normalize_ohlcv,
    analyze_indicators,
    passes_data_quality,
    evaluate_signal,
    calculate_position_size,
    calculate_entry_price,
    calculate_exit_prices,
    RISK_PER_TRADE_PCT,
)


# ============================================================================
# CONFIGURATION
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

PERIOD = os.getenv(
    "BACKTEST_PERIOD",
    "60d",
)

INTERVAL = os.getenv(
    "BACKTEST_INTERVAL",
    "15m",
)

STARTING_EQUITY = float(
    os.getenv(
        "BACKTEST_STARTING_EQUITY",
        "10000",
    )
)

SLIPPAGE_PCT = float(
    os.getenv(
        "BACKTEST_SLIPPAGE_PCT",
        "0.0005",
    )
)

COMMISSION_PER_TRADE = float(
    os.getenv(
        "BACKTEST_COMMISSION",
        "0.00",
    )
)

MAX_OPEN_POSITIONS = int(
    os.getenv(
        "BACKTEST_MAX_POSITIONS",
        "3",
    )
)


# ============================================================================
# DATA
# ============================================================================

def download_data(
    symbol: str,
    period: str = PERIOD,
    interval: str = INTERVAL,
) -> Optional[pd.DataFrame]:

    try:

        raw = yf.download(
            symbol,
            period=period,
            interval=interval,
            progress=False,
            auto_adjust=False,
        )

        if raw is None or raw.empty:
            return None

        df = normalize_ohlcv(raw)

        if df is None:
            return None

        return df

    except Exception as exc:

        print(
            f"[{symbol}] data error: {exc}"
        )

        return None


# ============================================================================
# 4H RESAMPLING
# ============================================================================

def build_4h_data(
    df: pd.DataFrame,
) -> pd.DataFrame:

    four_hour = (
        df
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
        .dropna()
    )

    # Remove currently-forming candle.
    if len(four_hour) > 1:
        four_hour = four_hour.iloc[:-1]

    return four_hour


# ============================================================================
# MAX DRAW DOWN
# ============================================================================

def calculate_max_drawdown(
    equity_curve: List[float],
) -> float:

    if not equity_curve:
        return 0.0

    curve = np.asarray(
        equity_curve,
        dtype=float,
    )

    peaks = np.maximum.accumulate(curve)

    drawdowns = (
        curve - peaks
    ) / peaks

    return float(
        drawdowns.min() * 100
    )


# ============================================================================
# TRADE
# ============================================================================

def close_trade(
    trade: Dict[str, Any],
    exit_price: float,
    exit_reason: str,
    equity: float,
) -> Dict[str, Any]:

    entry_price = trade["entry_price"]
    qty = trade["qty"]

    pnl = (
        (exit_price - entry_price)
        * qty
        - COMMISSION_PER_TRADE
    )

    risk_dollars = (
        trade["initial_risk_per_share"]
        * qty
    )

    if risk_dollars > 0:
        r_multiple = (
            pnl / risk_dollars
        )
    else:
        r_multiple = 0.0

    result = {
        **trade,
        "exit_price": exit_price,
        "exit_reason": exit_reason,
        "pnl": pnl,
        "r_multiple": r_multiple,
        "equity_after": equity + pnl,
    }

    return result


# ============================================================================
# SINGLE SYMBOL BACKTEST
# ============================================================================

def run_backtest(
    symbol: str,
    df: pd.DataFrame,
) -> Dict[str, Any]:

    df = normalize_ohlcv(df)

    if df is None or len(df) < 100:
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
            "avg_r": 0.0,
            "max_drawdown": 0.0,
            "ambiguous_bars": 0,
            "forced_exits": 0,
        }

    four_hour = build_4h_data(df)

    equity = STARTING_EQUITY

    trades: List[Dict[str, Any]] = []

    equity_curve: List[float] = [
        equity
    ]

    ambiguous_bars = 0
    forced_exits = 0

    in_trade = False
    pending_entry = False

    entry_price = 0.0
    qty = 0
    stop_price = 0.0
    target_price = 0.0
    initial_risk_per_share = 0.0

    entry_signal_index = None

    # ------------------------------------------------------------------
    # Walk forward.
    #
    # The signal is evaluated using bar i.
    # Entry occurs on bar i+1.
    # ------------------------------------------------------------------

    for i in range(60, len(df) - 1):

        current_bar = df.iloc[i]
        next_bar = df.iloc[i + 1]

        current_time = df.index[i]
        next_time = df.index[i + 1]

        # ==============================================================
        # MANAGE EXISTING POSITION
        # ==============================================================

        if in_trade:

            high = float(current_bar["High"])
            low = float(current_bar["Low"])

            hit_target = (
                high >= target_price
            )

            hit_stop = (
                low <= stop_price
            )

            if hit_target and hit_stop:

                # Conservative assumption:
                # stop occurs before target when both are
                # touched inside the same OHLC bar.
                ambiguous_bars += 1

                exit_price = (
                    stop_price
                    * (1 - SLIPPAGE_PCT)
                )

                result = close_trade(
                    {
                        "symbol": symbol,
                        "entry_time": entry_signal_index,
                        "entry_price": entry_price,
                        "qty": qty,
                        "stop_price": stop_price,
                        "target_price": target_price,
                        "initial_risk_per_share":
                            initial_risk_per_share,
                    },
                    exit_price,
                    "STOP_AMBIGUOUS",
                    equity,
                )

                equity += result["pnl"]

                trades.append(result)

                equity_curve.append(equity)

                in_trade = False

                continue

            if hit_target:

                exit_price = (
                    target_price
                    * (1 - SLIPPAGE_PCT)
                )

                result = close_trade(
                    {
                        "symbol": symbol,
                        "entry_time": entry_signal_index,
                        "entry_price": entry_price,
                        "qty": qty,
                        "stop_price": stop_price,
                        "target_price": target_price,
                        "initial_risk_per_share":
                            initial_risk_per_share,
                    },
                    exit_price,
                    "TARGET",
                    equity,
                )

                equity += result["pnl"]

                trades.append(result)

                equity_curve.append(equity)

                in_trade = False

                continue

            if hit_stop:

                exit_price = (
                    stop_price
                    * (1 - SLIPPAGE_PCT)
                )

                result = close_trade(
                    {
                        "symbol": symbol,
                        "entry_time": entry_signal_index,
                        "entry_price": entry_price,
                        "qty": qty,
                        "stop_price": stop_price,
                        "target_price": target_price,
                        "initial_risk_per_share":
                            initial_risk_per_share,
                    },
                    exit_price,
                    "STOP",
                    equity,
                )

                equity += result["pnl"]

                trades.append(result)

                equity_curve.append(equity)

                in_trade = False

                continue

        # ==============================================================
        # NEW SIGNAL
        # ==============================================================

        if in_trade:
            continue

        # --------------------------------------------------------------
        # Need enough completed 4H history available at this timestamp.
        # --------------------------------------------------------------

        completed_4h = four_hour[
            four_hour.index <= current_time
        ]

        if len(completed_4h) < 54:
            continue

        # --------------------------------------------------------------
        # Only use data available through current bar.
        # --------------------------------------------------------------

        df_1m = df.iloc[
            : i + 1
        ].copy()

        indicators = analyze_indicators(
            df_1m,
            completed_4h,
        )

        quality_ok, _ = passes_data_quality(
            df_1m,
            indicators,
        )

        if not quality_ok:
            continue

        signal = evaluate_signal(
            indicators
        )

        if not signal.valid:
            continue

        # ==============================================================
        # NEXT BAR EXECUTION
        # ==============================================================

        next_open = float(
            next_bar["Open"]
        )

        if next_open <= 0:
            continue

        # We model a limit-style entry as the greater of:
        # current calculated entry and next bar opening price
        # only when the next bar actually reaches the limit.
        #
        # For a conservative research test, require the next bar
        # to trade through the intended limit.
        intended_entry = calculate_entry_price(
            indicators["price"]
        )

        next_high = float(
            next_bar["High"]
        )

        next_low = float(
            next_bar["Low"]
        )

        if not (
            next_low
            <= intended_entry
            <= next_high
        ):
            continue

        # Entry fill.
        fill_price = (
            intended_entry
            * (1 + SLIPPAGE_PCT)
        )

        atr = indicators["atr"]

        qty = calculate_position_size(
            equity=equity,
            buying_power=equity,
            price=fill_price,
            atr=atr,
        )

        if qty <= 0:
            continue

        stop_price, target_price = (
            calculate_exit_prices(
                fill_price,
                atr,
            )
        )

        if (
            stop_price <= 0
            or target_price <= fill_price
        ):
            continue

        # Position cap.
        if (
            qty * fill_price
            > equity * 0.15
        ):
            qty = int(
                (
                    equity * 0.15
                ) / fill_price
            )

        if qty <= 0:
            continue

        # Enter.
        in_trade = True
        entry_price = fill_price
        initial_risk_per_share = (
            fill_price - stop_price
        )

        entry_signal_index = next_time

    # ==================================================================
    # FORCE END-OF-DATA EXIT
    # ==================================================================

    if in_trade:

        final_close = float(
            df["Close"].iloc[-1]
        )

        exit_price = (
            final_close
            * (1 - SLIPPAGE_PCT)
        )

        result = close_trade(
            {
                "symbol": symbol,
                "entry_time": entry_signal_index,
                "entry_price": entry_price,
                "qty": qty,
                "stop_price": stop_price,
                "target_price": target_price,
                "initial_risk_per_share":
                    initial_risk_per_share,
            },
            exit_price,
            "END_OF_DATA",
            equity,
        )

        equity += result["pnl"]

        trades.append(result)

        equity_curve.append(equity)

        forced_exits += 1

    # ==================================================================
    # STATISTICS
    # ==================================================================

    total_trades = len(trades)

    wins = [
        t for t in trades
        if t["pnl"] > 0
    ]

    losses = [
        t for t in trades
        if t["pnl"] <= 0
    ]

    win_rate = (
        len(wins)
        / total_trades
        * 100
        if total_trades
        else 0.0
    )

    net_pnl = sum(
        t["pnl"]
        for t in trades
    )

    expectancy = (
        net_pnl
        / total_trades
        if total_trades
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
            [
                t["r_multiple"]
                for t in trades
            ]
        )
        if trades
        else 0.0
    )

    max_dd = calculate_max_drawdown(
        equity_curve
    )

    return {
        "symbol": symbol,
        "trades": trades,
        "total_trades": total_trades,
        "wins": len(wins),
        "losses": len(losses),
        "win_rate": win_rate,
        "net_pnl": net_pnl,
        "expectancy": expectancy,
        "profit_factor": profit_factor,
        "avg_r": float(avg_r),
        "max_drawdown": max_dd,
        "ambiguous_bars": ambiguous_bars,
        "forced_exits": forced_exits,
        "ending_equity": equity,
    }


# ============================================================================
# FORMAT PF
# ============================================================================

def format_pf(
    pf: float,
) -> str:

    if pf == float("inf"):
        return "inf"

    return f"{pf:.2f}"


# ============================================================================
# MAIN
# ============================================================================

def main() -> None:

    print("=" * 78)
    print(
        "  BULL. BEAR AND BROKE — UNIFIED STRATEGY BACKTEST v12"
    )
    print(
        "  Shared signal engine | Next-bar execution | ATR exits"
    )
    print("=" * 78)

    results: List[Dict[str, Any]] = []

    for symbol in TICKERS:

        print(
            f"[{symbol:<5}] downloading..."
        )

        df = download_data(
            symbol
        )

        if df is None:

            print(
                f"[{symbol:<5}] no usable data"
            )

            results.append(
                {
                    "symbol": symbol,
                    "total_trades": 0,
                    "wins": 0,
                    "losses": 0,
                    "win_rate": 0.0,
                    "net_pnl": 0.0,
                    "expectancy": 0.0,
                    "profit_factor": 0.0,
                    "avg_r": 0.0,
                    "max_drawdown": 0.0,
                    "ambiguous_bars": 0,
                    "forced_exits": 0,
                }
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
            f"Trades={result['total_trades']:<3} | "
            f"W={result['wins']:<3} | "
            f"L={result['losses']:<3} | "
            f"Win={result['win_rate']:>5.1f}% | "
            f"P&L=${result['net_pnl']:>8.2f} | "
            f"Exp=${result['expectancy']:>7.2f} | "
            f"PF={format_pf(result['profit_factor']):>5} | "
            f"AvgR={result['avg_r']:>6.3f} | "
            f"DD={result['max_drawdown']:>6.2f}%"
        )

    # ==================================================================
    # SUMMARY
    # ==================================================================

    total_trades = sum(
        r["total_trades"]
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
        r["net_pnl"]
        for r in results
    )

    total_ambiguous = sum(
        r["ambiguous_bars"]
        for r in results
    )

    total_forced = sum(
        r["forced_exits"]
        for r in results
    )

    overall_win_rate = (
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

    gross_profit = sum(
        sum(
            t["pnl"]
            for t in r["trades"]
            if t["pnl"] > 0
        )
        for r in results
    )

    gross_loss = abs(
        sum(
            sum(
                t["pnl"]
                for t in r["trades"]
                if t["pnl"] <= 0
            )
            for r in results
        )
    )

    overall_pf = (
        gross_profit / gross_loss
        if gross_loss > 0
        else (
            float("inf")
            if gross_profit > 0
            else 0.0
        )
    )

    all_r = [
        t["r_multiple"]
        for r in results
        for t in r["trades"]
    ]

    average_r = (
        float(np.mean(all_r))
        if all_r
        else 0.0
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
        f"${expectancy:.2f}"
    )

    print(
        f"PROFIT FACTOR: "
        f"{format_pf(overall_pf)}"
    )

    print(
        f"AVERAGE R: "
        f"{average_r:.3f}"
    )

    print(
        f"AMBIGUOUS TP/SL BARS: "
        f"{total_ambiguous}"
    )

    print(
        f"FORCED END-OF-DATA EXITS: "
        f"{total_forced}"
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

    if total_pnl > 0 and overall_pf > 1:

        print(
            "📈 Historical result is positive — "
            "OUT-OF-SAMPLE TESTING STILL REQUIRED."
        )

    else:

        print(
            "⚠️ Historical result is not yet "
            "convincingly profitable."
        )

    print("=" * 78)


if __name__ == "__main__":
    main()
