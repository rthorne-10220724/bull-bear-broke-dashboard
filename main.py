"""
BULL. BEAR AND BROKE — MAIN SUPERVISOR
======================================

Clean Render entrypoint for the existing v13 trading engine.

Fixes/safeguards provided here:
- reconcile Alpaca positions against open protective orders every cycle
- repair orphaned exits automatically in PAPER mode by default
- block new entries while an existing position is unprotected
- count only BUY orders as pending entries
- use a configurable 1-minute ATR-percent quality floor
- fetch enough 1-minute history for the engine's 15-minute MACD calculation
- preserve the existing main_v6.py discovery, scoring, sizing, and entry logic

Recommended Render start command:
    python main.py
"""

from __future__ import annotations

import datetime as dt
import os
import time
from typing import Any, Dict, Iterable, List, Optional, Set, Tuple

import numpy as np
import pandas as pd
import ta

import main_v6 as engine

from alpaca.trading.requests import (
    LimitOrderRequest,
    TakeProfitRequest,
    StopLossRequest,
)
from alpaca.trading.enums import (
    OrderSide,
    TimeInForce,
    OrderClass,
)


# ============================================================================
# SUPERVISOR CONFIG
# ============================================================================

AUTO_REPAIR_ORPHANED_EXITS = (
    os.getenv(
        "AUTO_REPAIR_ORPHANED_EXITS",
        "true" if engine.PAPER_TRADING else "false",
    ).strip().lower()
    == "true"
)

BLOCK_NEW_ENTRIES_IF_UNPROTECTED = (
    os.getenv("BLOCK_NEW_ENTRIES_IF_UNPROTECTED", "true")
    .strip()
    .lower()
    == "true"
)

# The original v13 hard-coded 0.35% minimum 1-minute ATR. That rejected many
# normal liquid stocks before they ever reached scoring. Keep a floor, but use
# 0.08% by default and allow Render to override it.
MIN_1M_ATR_PCT = float(os.getenv("MIN_1M_ATR_PCT", "0.08"))

# MACD bug fix:
# main_v6.py resamples 1-minute closes into 15-minute bars and requires >=40
# completed 15-minute bars. Its old fetch returned only 180 one-minute bars,
# which could produce only ~12 15-minute bars. Fetch multiple calendar days so
# the existing MACD logic can actually run.
MACD_HISTORY_DAYS = int(os.getenv("MACD_HISTORY_DAYS", "7"))
MACD_MIN_1M_BARS = int(os.getenv("MACD_MIN_1M_BARS", "900"))

REPAIR_STOP_ATR_MULTIPLIER = engine.ATR_MULTIPLIER_STOP
REPAIR_TARGET_ATR_MULTIPLIER = engine.ATR_MULTIPLIER_TARGET
REPAIR_SETTLE_SECONDS = 2


# ============================================================================
# GENERIC HELPERS
# ============================================================================

def _enum_text(value: Any) -> str:
    if value is None:
        return ""
    raw = getattr(value, "value", value)
    return str(raw).strip().lower()


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        number = float(value)
        return number if np.isfinite(number) else default
    except (TypeError, ValueError):
        return default


def _iter_order_tree(orders: Iterable[Any]) -> Iterable[Any]:
    for order in orders or []:
        yield order
        for leg in getattr(order, "legs", None) or []:
            yield from _iter_order_tree([leg])


# ============================================================================
# FIX 1 — PENDING ENTRIES MUST MEAN BUY ORDERS ONLY
# ============================================================================

def get_pending_entry_symbols(orders: List[Any]) -> Set[str]:
    symbols: Set[str] = set()

    for order in _iter_order_tree(orders):
        symbol = str(getattr(order, "symbol", "") or "").upper()
        side = _enum_text(getattr(order, "side", None))

        if symbol and side == "buy":
            symbols.add(symbol)

    return symbols


# ============================================================================
# FIX 2 — CALIBRATED 1-MINUTE DATA QUALITY GATE
# ============================================================================

def passes_data_quality(
    df_1m: pd.DataFrame,
    indicators: Dict[str, Any],
) -> Tuple[bool, str]:
    if len(df_1m) < 60:
        return False, f"insufficient 1m history ({len(df_1m)} < 60)"

    required_columns = {"Open", "High", "Low", "Close", "Volume"}

    if not required_columns.issubset(df_1m.columns):
        return False, "missing OHLCV columns"

    if not np.isfinite(
        df_1m[list(required_columns)].tail(30).to_numpy()
    ).all():
        return False, "non-finite market data"

    price = _safe_float(indicators.get("price"))
    atr = _safe_float(indicators.get("atr"))
    rvol = _safe_float(indicators.get("rvol"))
    atr_pct = _safe_float(indicators.get("atr_pct"))

    if price <= 0:
        return False, "invalid price"

    if atr <= 0:
        return False, "invalid ATR"

    if rvol < engine.MIN_RVOL:
        return False, (
            f"insufficient RVOL ({rvol:.2f} < {engine.MIN_RVOL:.2f})"
        )

    if atr_pct < MIN_1M_ATR_PCT:
        return False, (
            f"volatility too low (1m ATR%={atr_pct:.3f} < {MIN_1M_ATR_PCT:.3f})"
        )

    return True, (
        f"data/volatility valid (RVOL={rvol:.2f}, 1m ATR%={atr_pct:.3f})"
    )


# ============================================================================
# FIX 3 — ENOUGH 1-MINUTE HISTORY FOR 15-MINUTE MACD
# ============================================================================

def fetch_1m_bars_with_macd_history(
    ticker: str,
    limit: int = 180,
    retries: int = 3,
) -> Optional[pd.DataFrame]:
    """
    Fetch a multi-day 1-minute history.

    The v13 indicator function already resamples this data to 15-minute closes.
    Returning >=900 recent 1-minute bars gives it enough completed 15-minute
    candles to satisfy its >=40-bar MACD requirement while preserving the same
    RSI/EMA/ATR/RVOL calculations on the most recent data.
    """
    end = dt.datetime.now(dt.timezone.utc)
    start = end - dt.timedelta(days=max(3, MACD_HISTORY_DAYS))
    is_crypto = "/" in ticker

    for attempt in range(retries):
        try:
            if is_crypto:
                request = engine.CryptoBarsRequest(
                    symbol_or_symbols=ticker,
                    timeframe=engine.TimeFrame.Minute,
                    start=start,
                    end=end,
                )
                response = engine.crypto_data_client.get_crypto_bars(request)
            else:
                request = engine.StockBarsRequest(
                    symbol_or_symbols=ticker,
                    timeframe=engine.TimeFrame.Minute,
                    start=start,
                    end=end,
                    feed=engine.DataFeed.IEX,
                )
                response = engine.stock_data_client.get_stock_bars(request)

            df = response.df

            if isinstance(df.index, pd.MultiIndex):
                try:
                    df = df.xs(ticker, level=0)
                except Exception:
                    return None

            df = engine.normalize_ohlcv(df)

            if df is None:
                raise ValueError("Invalid/empty OHLCV data")

            desired = max(limit, MACD_MIN_1M_BARS)
            df = df.tail(desired)

            # Do not reject here if a thin/new symbol has fewer bars. The
            # existing quality gate will log the exact reason downstream.
            return df

        except Exception as exc:
            engine.logger.warning(
                "[%s] extended 1m data attempt %s/%s failed: %s",
                ticker,
                attempt + 1,
                retries,
                exc,
            )

            if attempt < retries - 1:
                time.sleep(1.5)

    return None


# Apply the targeted patches. main_v6.py remains the strategy source of truth.
engine.get_pending_symbols = get_pending_entry_symbols
engine.passes_data_quality = passes_data_quality
engine.fetch_1m_bars = fetch_1m_bars_with_macd_history


# ============================================================================
# POSITION PROTECTION / RECONCILIATION
# ============================================================================

def _sell_orders_by_symbol(orders: List[Any]) -> Dict[str, List[Any]]:
    grouped: Dict[str, List[Any]] = {}

    for order in _iter_order_tree(orders):
        symbol = str(getattr(order, "symbol", "") or "").upper()
        side = _enum_text(getattr(order, "side", None))

        if symbol and side == "sell":
            grouped.setdefault(symbol, []).append(order)

    return grouped


def _classify_protection(orders: List[Any]) -> Tuple[bool, bool]:
    has_stop = False
    has_target = False

    for order in orders:
        order_type = _enum_text(
            getattr(order, "type", None)
            or getattr(order, "order_type", None)
        )
        stop_price = getattr(order, "stop_price", None)
        limit_price = getattr(order, "limit_price", None)

        if stop_price not in (None, "", "0", 0):
            has_stop = True

        if order_type in {"stop", "stop_limit", "trailing_stop"}:
            has_stop = True

        if (
            limit_price not in (None, "", "0", 0)
            and order_type != "stop_limit"
        ):
            has_target = True

    return has_stop, has_target


def _latest_atr(symbol: str) -> float:
    df = engine.fetch_1m_bars(symbol, limit=180)

    if df is None or df.empty or len(df) < engine.ATR_WINDOW + 5:
        return 0.0

    required = {"High", "Low", "Close"}
    if not required.issubset(df.columns):
        return 0.0

    try:
        atr_series = ta.volatility.AverageTrueRange(
            high=pd.to_numeric(df["High"], errors="coerce"),
            low=pd.to_numeric(df["Low"], errors="coerce"),
            close=pd.to_numeric(df["Close"], errors="coerce"),
            window=engine.ATR_WINDOW,
        ).average_true_range()

        return max(_safe_float(atr_series.iloc[-1]), 0.0)

    except Exception as exc:
        engine.logger.error("[%s] ATR repair calculation failed: %s", symbol, exc)
        return 0.0


def _repair_oco_for_position(position: Any) -> bool:
    symbol = str(getattr(position, "symbol", "") or "").upper()
    qty = _safe_float(getattr(position, "qty", 0))
    current_price = _safe_float(getattr(position, "current_price", 0))

    if not symbol or "/" in symbol:
        engine.logger.warning(
            "[%s] OCO repair skipped: unsupported/non-stock symbol.",
            symbol or "UNKNOWN",
        )
        return False

    if qty <= 0 or current_price <= 0:
        engine.logger.error(
            "[%s] OCO repair skipped: invalid qty/current_price qty=%s price=%s",
            symbol,
            qty,
            current_price,
        )
        return False

    rounded_qty = round(qty)
    if abs(qty - rounded_qty) > 1e-8:
        engine.logger.error(
            "[%s] OCO repair skipped: fractional qty %.8f requires manual handling.",
            symbol,
            qty,
        )
        return False

    atr = _latest_atr(symbol)
    if atr <= 0:
        engine.logger.error(
            "[%s] OCO repair skipped: no valid ATR available.",
            symbol,
        )
        return False

    stop_price = round(current_price - (atr * REPAIR_STOP_ATR_MULTIPLIER), 2)
    target_price = round(current_price + (atr * REPAIR_TARGET_ATR_MULTIPLIER), 2)

    stop_price = max(0.01, min(stop_price, round(current_price - 0.01, 2)))
    target_price = max(round(current_price + 0.01, 2), target_price)

    if stop_price >= current_price or target_price <= current_price:
        engine.logger.error(
            "[%s] OCO repair rejected locally: current=%.2f stop=%.2f target=%.2f",
            symbol,
            current_price,
            stop_price,
            target_price,
        )
        return False

    try:
        request = LimitOrderRequest(
            symbol=symbol,
            qty=int(rounded_qty),
            side=OrderSide.SELL,
            limit_price=target_price,
            time_in_force=TimeInForce.GTC,
            order_class=OrderClass.OCO,
            take_profit=TakeProfitRequest(limit_price=target_price),
            stop_loss=StopLossRequest(stop_price=stop_price),
        )

        response = engine.trading_client.submit_order(request)

        engine.logger.warning(
            "[%s] REPAIRED OCO PROTECTION | qty=%s current=%.2f "
            "| stop=%.2f | target=%.2f | order_id=%s",
            symbol,
            int(rounded_qty),
            current_price,
            stop_price,
            target_price,
            getattr(response, "id", "unknown"),
        )

        engine.log_decision(
            f"{symbol} orphaned-position protection repaired | "
            f"qty={int(rounded_qty)} stop={stop_price:.2f} target={target_price:.2f}"
        )
        return True

    except Exception as exc:
        engine.logger.exception(
            "[%s] OCO protection repair FAILED: %s",
            symbol,
            exc,
        )
        return False


def reconcile_portfolio() -> Tuple[bool, Set[str]]:
    positions = engine.get_open_positions()
    orders = engine.get_open_orders()
    sell_orders = _sell_orders_by_symbol(orders)

    engine.logger.info("========== PORTFOLIO RECONCILIATION ==========")

    if not positions:
        engine.logger.info("No open positions. Portfolio protection status: CLEAR")
        return True, set()

    unprotected: Set[str] = set()

    for symbol, position in positions.items():
        symbol = str(symbol).upper()
        has_stop, has_target = _classify_protection(
            sell_orders.get(symbol, [])
        )

        qty = _safe_float(getattr(position, "qty", 0))
        avg_entry = _safe_float(getattr(position, "avg_entry_price", 0))
        current = _safe_float(getattr(position, "current_price", 0))
        unrealized_plpc = _safe_float(getattr(position, "unrealized_plpc", 0))

        protected = has_stop and has_target

        engine.logger.info(
            "[%s] qty=%.4f avg=%.2f current=%.2f unrealized=%+.2f%% "
            "| stop=%s target=%s | protected=%s",
            symbol,
            qty,
            avg_entry,
            current,
            unrealized_plpc * 100,
            "YES" if has_stop else "NO",
            "YES" if has_target else "NO",
            "YES" if protected else "NO",
        )

        if protected:
            continue

        unprotected.add(symbol)

        engine.logger.warning(
            "[%s] UNPROTECTED POSITION DETECTED | missing=%s%s",
            symbol,
            "STOP " if not has_stop else "",
            "TARGET" if not has_target else "",
        )

        if AUTO_REPAIR_ORPHANED_EXITS and _repair_oco_for_position(position):
            unprotected.discard(symbol)

    if AUTO_REPAIR_ORPHANED_EXITS:
        time.sleep(REPAIR_SETTLE_SECONDS)
        fresh_orders = engine.get_open_orders()
        fresh_sell_orders = _sell_orders_by_symbol(fresh_orders)
        final_unprotected: Set[str] = set()

        for symbol in positions:
            symbol = str(symbol).upper()
            has_stop, has_target = _classify_protection(
                fresh_sell_orders.get(symbol, [])
            )
            if not (has_stop and has_target):
                final_unprotected.add(symbol)

        unprotected = final_unprotected

    if unprotected:
        engine.logger.error(
            "PORTFOLIO RECONCILIATION FAILED | unprotected=%s",
            ", ".join(sorted(unprotected)),
        )
        return False, unprotected

    engine.logger.info(
        "PORTFOLIO RECONCILIATION PASSED | all %s position(s) protected",
        len(positions),
    )
    return True, set()


# ============================================================================
# MAIN LOOP
# ============================================================================

def run_supervised_cycle() -> None:
    protected, unprotected = reconcile_portfolio()

    if BLOCK_NEW_ENTRIES_IF_UNPROTECTED and not protected:
        engine.logger.error(
            "NEW ENTRIES BLOCKED: existing position(s) lack complete protection: %s",
            ", ".join(sorted(unprotected)),
        )
        return

    engine.run_cycle()


def main() -> None:
    engine.init_db()

    engine.logger.info("==========================================")
    engine.logger.info("BULL. BEAR AND BROKE — MAIN SUPERVISOR")
    engine.logger.info("Engine module: main_v6.py (v13 strategy engine)")
    engine.logger.info("Paper trading: %s", engine.PAPER_TRADING)
    engine.logger.info(
        "Auto-repair orphaned exits: %s",
        AUTO_REPAIR_ORPHANED_EXITS,
    )
    engine.logger.info(
        "Block entries if unprotected: %s",
        BLOCK_NEW_ENTRIES_IF_UNPROTECTED,
    )
    engine.logger.info("Pending-entry accounting: BUY orders only")
    engine.logger.info("1m ATR quality floor: %.3f%%", MIN_1M_ATR_PCT)
    engine.logger.info(
        "MACD history: %s calendar days / up to %s 1m bars",
        MACD_HISTORY_DAYS,
        MACD_MIN_1M_BARS,
    )
    engine.logger.info("Cycle seconds: %s", engine.CYCLE_SECONDS)
    engine.logger.info("==========================================")

    while True:
        try:
            run_supervised_cycle()

        except KeyboardInterrupt:
            engine.logger.info("Shutdown requested.")
            break

        except Exception as exc:
            engine.logger.exception("Supervisor cycle failure: %s", exc)

        engine.logger.info(
            "Sleeping %s seconds...",
            engine.CYCLE_SECONDS,
        )
        time.sleep(engine.CYCLE_SECONDS)


if __name__ == "__main__":
    main()
