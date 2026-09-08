"""
BULL. BEAR AND BROKE — MAIN SUPERVISOR
======================================

Clean Render entrypoint for the existing trading engine.

Responsibilities added here:
- reconcile Alpaca positions against open protective orders every cycle
- identify missing stop-loss / take-profit protection
- optionally repair orphaned exits with an OCO order
- default automatic repair ON in paper trading, OFF in live trading
- block new entries if any open position remains unprotected
- preserve the existing main_v6.py signal, sizing, discovery, and entry rules

Recommended Render start command:
    python main.py
"""

from __future__ import annotations

import os
import sys
import time
from typing import Any, Dict, Iterable, List, Set, Tuple

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

# In PAPER mode, repair orphaned exits by default so the bot can prove the
# full buy -> protect -> exit lifecycle. In LIVE mode, default to warning only.
AUTO_REPAIR_ORPHANED_EXITS = (
    os.getenv(
        "AUTO_REPAIR_ORPHANED_EXITS",
        "true" if engine.PAPER_TRADING else "false",
    ).strip().lower()
    == "true"
)

# Never add new risk while an existing position is missing protection.
BLOCK_NEW_ENTRIES_IF_UNPROTECTED = (
    os.getenv("BLOCK_NEW_ENTRIES_IF_UNPROTECTED", "true")
    .strip()
    .lower()
    == "true"
)

# ATR-based repair uses the same stop/target multipliers as the entry engine.
REPAIR_STOP_ATR_MULTIPLIER = engine.ATR_MULTIPLIER_STOP
REPAIR_TARGET_ATR_MULTIPLIER = engine.ATR_MULTIPLIER_TARGET

# Give broker state a moment to settle after submitting repaired OCO exits.
REPAIR_SETTLE_SECONDS = 2


# ============================================================================
# ORDER / POSITION HELPERS
# ============================================================================

def _enum_text(value: Any) -> str:
    """Normalize Alpaca enum/string values for logging and comparisons."""
    if value is None:
        return ""
    raw = getattr(value, "value", value)
    return str(raw).strip().lower()


def _iter_order_tree(orders: Iterable[Any]) -> Iterable[Any]:
    """Yield parent orders and nested legs returned by nested=True."""
    for order in orders or []:
        yield order
        legs = getattr(order, "legs", None) or []
        for leg in _iter_order_tree(legs):
            yield leg


def _sell_orders_by_symbol(orders: List[Any]) -> Dict[str, List[Any]]:
    grouped: Dict[str, List[Any]] = {}

    for order in _iter_order_tree(orders):
        symbol = str(getattr(order, "symbol", "") or "").upper()
        side = _enum_text(getattr(order, "side", None))

        if not symbol or side != "sell":
            continue

        grouped.setdefault(symbol, []).append(order)

    return grouped


def _classify_protection(orders: List[Any]) -> Tuple[bool, bool]:
    """
    Return (has_stop, has_target).

    Alpaca may expose an OCO take-profit as the parent and the stop as a child
    when orders are requested with nested=True.
    """
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
            and order_type not in {"stop_limit"}
        ):
            has_target = True

    return has_stop, has_target


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        number = float(value)
        return number if np.isfinite(number) else default
    except (TypeError, ValueError):
        return default


def _latest_atr(symbol: str) -> float:
    """
    Calculate current 1-minute ATR using the engine's own Alpaca data helper.
    Returns 0 when data is unavailable.
    """
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

        atr = _safe_float(atr_series.iloc[-1])
        return max(atr, 0.0)

    except Exception as exc:
        engine.logger.error("[%s] ATR repair calculation failed: %s", symbol, exc)
        return 0.0


def _repair_oco_for_position(position: Any) -> bool:
    """
    Submit a fresh OCO exit for an already-open long stock position.

    Repair prices are anchored to CURRENT market price, not a stale historical
    target. This guarantees the stop remains below current market and the
    take-profit above it.

    Automatic repair is intended for PAPER trading by default.
    """
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

    # Alpaca GTC limit/OCO equity orders generally need whole-share qty.
    # Do not guess how to protect a fractional remainder.
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

    # Alpaca requires a valid positive stop and at least $0.01 separation.
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


# ============================================================================
# RECONCILIATION
# ============================================================================

def reconcile_portfolio() -> Tuple[bool, Set[str]]:
    """
    Reconcile positions with protective sell orders.

    Returns:
        all_protected: True only when every open position has both stop + target.
        unprotected_symbols: symbols still missing complete protection.
    """
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
        protective_orders = sell_orders.get(symbol, [])
        has_stop, has_target = _classify_protection(protective_orders)

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

        if AUTO_REPAIR_ORPHANED_EXITS:
            repaired = _repair_oco_for_position(position)

            if repaired:
                # Remove provisionally; we verify broker state after all repairs.
                unprotected.discard(symbol)

    # Verify repairs using fresh broker state.
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

    if (
        BLOCK_NEW_ENTRIES_IF_UNPROTECTED
        and not protected
    ):
        engine.logger.error(
            "NEW ENTRIES BLOCKED: existing position(s) lack complete protection: %s",
            ", ".join(sorted(unprotected)),
        )
        return

    # Existing engine remains the single source of truth for:
    # market hours, circuit breaker, active capacity, discovery, signal score,
    # position sizing, cooldowns, duplicate prevention, and order submission.
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
