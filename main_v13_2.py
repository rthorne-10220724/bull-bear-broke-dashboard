"""
BULL. BEAR AND BROKE — V13.2 ADAPTIVE ENTRY CONTROLLER
======================================================

Thin safety/learning layer on top of the working V13.1 supervisor in main.py.

Purpose:
- Keep normal daily entry behavior capped at 5.
- Permit one 6th entry only when the setup is materially stronger than normal.
- Never permit more than 6 new entries in a day.
- Stop opening new positions after 3 filled stop-loss exits in the same day.
- After the stop-out pause, continue scanning in diagnostic-only mode so good
  setups are visible without sending any new broker entry orders.
- Preserve all V13.1 protections: per-symbol cap, cooldown, fresh re-entry,
  stop-distance floor, risk sizing, portfolio reconciliation and OCO repair.

Recommended Render start command:
    python main_v13_2.py

PAPER TRADING FIRST. This controller is intended to generate more useful
research data without reopening the rapid-fire churn behavior V13.1 fixed.
"""

from __future__ import annotations

import datetime as dt
import os
from typing import Any, Dict, Iterable, List, Tuple

import main as supervisor

engine = supervisor.engine


# ---------------------------------------------------------------------------
# Adaptive daily-entry policy
# ---------------------------------------------------------------------------

SOFT_DAILY_ENTRY_CAP = int(os.getenv("SOFT_DAILY_ENTRY_CAP", "5"))
HARD_DAILY_ENTRY_CAP = int(os.getenv("HARD_DAILY_ENTRY_CAP", "6"))
MAX_DAILY_STOP_OUTS = int(os.getenv("MAX_DAILY_STOP_OUTS", "3"))
MAX_DAILY_REALIZED_LOSS_DOLLARS = float(
    os.getenv("MAX_DAILY_REALIZED_LOSS_DOLLARS", "750")
)

# The ordinary V13 engine accepts score >= 6. The single exception entry must
# be clearly better and must also show renewed momentum/reclaim structure.
EXCEPTIONAL_MIN_SIGNAL_SCORE = int(os.getenv("EXCEPTIONAL_MIN_SIGNAL_SCORE", "8"))
EXCEPTIONAL_MIN_RVOL = float(os.getenv("EXCEPTIONAL_MIN_RVOL", "1.0"))

# Disable the old 5-entry hard stop inside main_v6.run_cycle. V13.2 enforces
# soft-5 / exceptional-6 / hard-6 through the patched signal evaluator below.
engine.MAX_NEW_ENTRIES_PER_DAY = HARD_DAILY_ENTRY_CAP

# True only while the post-stop-out research scan is running. This bypasses
# only the total daily entry cap so we can see what the strategy would find.
# V13.1 signal quality, per-symbol entry limits, cooldowns, data quality,
# portfolio checks and sizing still run normally.
_DIAGNOSTIC_ONLY_MODE = False
_DIAGNOSTIC_WOULD_ENTRIES = []


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _enum_text(value: Any) -> str:
    if value is None:
        return ""
    raw = getattr(value, "value", value)
    return str(raw).strip().lower()


def _iter_order_tree(orders: Iterable[Any]) -> Iterable[Any]:
    for order in orders or []:
        yield order
        for leg in getattr(order, "legs", None) or []:
            yield from _iter_order_tree([leg])


def _today_start_utc() -> dt.datetime:
    """Midnight Eastern converted to UTC, matching the user's trading day."""
    try:
        from zoneinfo import ZoneInfo
        eastern = dt.datetime.now(ZoneInfo("America/New_York"))
    except Exception:
        eastern = dt.datetime.now(dt.timezone.utc)
    start_et = eastern.replace(hour=0, minute=0, second=0, microsecond=0)
    return start_et.astimezone(dt.timezone.utc)


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _filled_at(order: Any) -> dt.datetime:
    value = getattr(order, "filled_at", None) or getattr(order, "updated_at", None)
    if isinstance(value, dt.datetime):
        if value.tzinfo is None:
            return value.replace(tzinfo=dt.timezone.utc)
        return value.astimezone(dt.timezone.utc)
    return dt.datetime.min.replace(tzinfo=dt.timezone.utc)


def _is_stop_order(order: Any) -> bool:
    order_type = _enum_text(
        getattr(order, "type", None) or getattr(order, "order_type", None)
    )
    stop_price = getattr(order, "stop_price", None)
    return (
        order_type in {"stop", "stop_limit", "trailing_stop"}
        or stop_price not in (None, "", "0", 0)
    )


def get_today_trade_summary() -> Dict[str, Any]:
    """
    Reconstruct today's closed long trades from broker fills.

    A filled sell-stop is counted as a loss only when its matched realized P/L
    is negative. Profitable trailing-stop exits are winners, not stop-outs.
    FIFO lot matching also supports multiple entries in the same symbol.
    """
    summary: Dict[str, Any] = {
        "available": False,
        "losses": 0,
        "wins": 0,
        "breakeven": 0,
        "stop_exits": 0,
        "realized_pnl": 0.0,
        "trades": [],
        "unmatched_sell_qty": 0.0,
    }

    try:
        request = supervisor.GetOrdersRequest(
            status=supervisor.QueryOrderStatus.CLOSED,
            after=_today_start_utc(),
            nested=True,
        )
        roots: List[Any] = list(
            engine.trading_client.get_orders(filter=request) or []
        )
    except Exception as exc:
        engine.logger.error(
            "V13.2 realized-P/L audit failed; new entries will be paused: %s",
            exc,
        )
        return summary

    unique: Dict[str, Any] = {}
    anonymous = 0
    for order in _iter_order_tree(roots):
        status = _enum_text(getattr(order, "status", None))
        if status != "filled":
            continue

        order_id = str(getattr(order, "id", "") or "")
        if not order_id:
            anonymous += 1
            order_id = f"anonymous:{anonymous}:{id(order)}"
        unique.setdefault(order_id, order)

    filled = sorted(unique.values(), key=_filled_at)
    lots: Dict[str, List[List[float]]] = {}

    for order in filled:
        symbol = str(getattr(order, "symbol", "") or "").upper()
        side = _enum_text(getattr(order, "side", None))
        qty = _safe_float(
            getattr(order, "filled_qty", None) or getattr(order, "qty", None)
        )
        price = _safe_float(getattr(order, "filled_avg_price", None))

        if not symbol or qty <= 0 or price <= 0:
            continue

        if side == "buy":
            lots.setdefault(symbol, []).append([qty, price])
            continue
        if side != "sell":
            continue

        remaining = qty
        cost = 0.0
        matched = 0.0
        symbol_lots = lots.setdefault(symbol, [])

        while remaining > 1e-9 and symbol_lots:
            lot_qty, lot_price = symbol_lots[0]
            used = min(remaining, lot_qty)
            cost += used * lot_price
            matched += used
            remaining -= used
            lot_qty -= used
            if lot_qty <= 1e-9:
                symbol_lots.pop(0)
            else:
                symbol_lots[0][0] = lot_qty

        if matched <= 0:
            summary["unmatched_sell_qty"] += qty
            continue

        proceeds = matched * price
        pnl = proceeds - cost
        average_entry = cost / matched
        is_stop = _is_stop_order(order)
        if is_stop:
            summary["stop_exits"] += 1

        # One cent avoids classifying tiny floating-point residue as a loss.
        if pnl < -0.01:
            classification = "LOSS"
            summary["losses"] += 1
        elif pnl > 0.01:
            classification = "WIN"
            summary["wins"] += 1
        else:
            classification = "BREAKEVEN"
            summary["breakeven"] += 1

        summary["realized_pnl"] += pnl
        trade = {
            "symbol": symbol,
            "qty": matched,
            "entry": average_entry,
            "exit": price,
            "pnl": pnl,
            "classification": classification,
            "exit_kind": "STOP" if is_stop else "OTHER",
            "filled_at": _filled_at(order),
        }
        summary["trades"].append(trade)

    summary["available"] = True
    return summary


def count_today_stop_outs() -> int:
    """Compatibility wrapper: count genuine losing closed trades."""
    return int(get_today_trade_summary()["losses"])


# ---------------------------------------------------------------------------
# Signal gate: normal first five, exceptional sixth only
# ---------------------------------------------------------------------------

_v13_1_evaluate_signal = engine.evaluate_signal


def evaluate_signal_v13_2(indicators):
    result = _v13_1_evaluate_signal(indicators)

    if not result.valid:
        return result

    # Post-stop-out diagnostic mode asks: "Would the underlying V13.1 setup
    # have qualified right now?" It therefore ignores only the aggregate
    # daily-entry cap. It does not weaken the actual signal rules.
    if _DIAGNOSTIC_ONLY_MODE:
        return result

    entries_today = engine.get_daily_entry_count()

    if entries_today >= HARD_DAILY_ENTRY_CAP:
        return engine.SignalResult(
            valid=False,
            score=result.score,
            reasons=result.reasons,
            rejection=(
                f"V13.2 hard daily entry cap reached "
                f"({entries_today}/{HARD_DAILY_ENTRY_CAP})"
            ),
        )

    # Entries 1-5 use the existing V13.1 gates unchanged.
    if entries_today < SOFT_DAILY_ENTRY_CAP:
        return result

    # Entry #6 is the one research exception. Require a materially stronger
    # signal plus confirmation that the setup is actively reclaiming/moving.
    exceptional = (
        result.score >= EXCEPTIONAL_MIN_SIGNAL_SCORE
        and bool(indicators.get("breakout"))
        and bool(indicators.get("macd_improving"))
        and bool(indicators.get("short_term_bullish"))
        and float(indicators.get("rvol") or 0.0) >= EXCEPTIONAL_MIN_RVOL
    )

    if not exceptional:
        return engine.SignalResult(
            valid=False,
            score=result.score,
            reasons=result.reasons,
            rejection=(
                "V13.2 soft daily cap reached: sixth entry requires "
                f"score>={EXCEPTIONAL_MIN_SIGNAL_SCORE}, breakout, improving "
                f"MACD, bullish 1m EMA structure and RVOL>={EXCEPTIONAL_MIN_RVOL:.2f}"
            ),
        )

    engine.logger.warning(
        "V13.2 EXCEPTIONAL SIXTH ENTRY QUALIFIED | score=%s rvol=%.2f",
        result.score,
        float(indicators.get("rvol") or 0.0),
    )
    return result


engine.evaluate_signal = evaluate_signal_v13_2


# ---------------------------------------------------------------------------
# Diagnostic-only broker shim
# ---------------------------------------------------------------------------

_real_place_stock_bracket = engine.place_stock_bracket


def diagnostic_place_stock_bracket(*, symbol, qty, price, atr, score, reasons):
    """Record a would-be entry without sending anything to Alpaca."""
    record = {
        "symbol": symbol,
        "qty": qty,
        "price": price,
        "atr": atr,
        "score": score,
        "reasons": list(reasons or []),
    }
    _DIAGNOSTIC_WOULD_ENTRIES.append(record)

    engine.logger.warning(
        "[%s] DIAGNOSTIC WOULD ENTER — ORDER BLOCKED | qty=%s price=%.2f "
        "score=%s reasons=%s",
        symbol,
        qty,
        float(price),
        score,
        " | ".join(reasons or []),
    )

    # Return a harmless sentinel so run_cycle continues normally. No broker
    # method is called. Its generic 'order accepted' line is reinterpreted by
    # the explicit diagnostic summary printed immediately after the scan.
    return object()


def run_diagnostic_only_scan() -> None:
    """Run the real scanner/sizer while replacing only final order submission."""
    global _DIAGNOSTIC_ONLY_MODE

    _DIAGNOSTIC_WOULD_ENTRIES.clear()
    _DIAGNOSTIC_ONLY_MODE = True

    original_place = engine.place_stock_bracket
    original_daily_cap = engine.MAX_NEW_ENTRIES_PER_DAY

    # Ensure run_cycle reaches the scanner even if the hard daily cap has
    # already been reached. No entry can reach Alpaca because submission is
    # replaced below.
    engine.place_stock_bracket = diagnostic_place_stock_bracket
    engine.MAX_NEW_ENTRIES_PER_DAY = 999999

    engine.logger.warning(
        "V13.2 DIAGNOSTIC-ONLY SCAN STARTED | new broker entries disabled"
    )

    try:
        engine.run_cycle()
    finally:
        engine.place_stock_bracket = original_place
        engine.MAX_NEW_ENTRIES_PER_DAY = original_daily_cap
        _DIAGNOSTIC_ONLY_MODE = False

    if _DIAGNOSTIC_WOULD_ENTRIES:
        ranked = sorted(
            _DIAGNOSTIC_WOULD_ENTRIES,
            key=lambda x: (x["score"], x["price"]),
            reverse=True,
        )
        best = ranked[0]
        engine.logger.warning(
            "V13.2 DIAGNOSTIC SUMMARY | would_enter=%s | best=%s score=%s "
            "price=%.2f | NO ORDERS SENT",
            len(ranked),
            best["symbol"],
            best["score"],
            float(best["price"]),
        )
    else:
        engine.logger.warning(
            "V13.2 DIAGNOSTIC SUMMARY | would_enter=0 | no setup cleared "
            "the existing V13.1 strategy rules | NO ORDERS SENT"
        )

    engine.logger.info(
        "V13.2 diagnostic note: any V13 discovery-summary 'submitted' count "
        "during diagnostic-only mode means WOULD-ENTER simulations, not broker orders."
    )


# ---------------------------------------------------------------------------
# Daily stop-out circuit breaker
# ---------------------------------------------------------------------------

_v13_1_run_supervised_cycle = supervisor.run_supervised_cycle


def run_supervised_cycle_v13_2() -> None:
    trade_summary = get_today_trade_summary()
    entries_today = engine.get_daily_entry_count()

    if not trade_summary["available"]:
        engine.logger.error(
            "V13.2 NEW ENTRIES PAUSED | realized-P/L audit unavailable. "
            "Existing positions remain protected and managed."
        )
        protected, unprotected = supervisor.reconcile_portfolio()
        if not protected:
            engine.logger.error(
                "V13.2 pause reconciliation found unprotected positions: %s",
                ", ".join(sorted(unprotected)),
            )
        return

    losses = int(trade_summary["losses"])
    realized_pnl = float(trade_summary["realized_pnl"])
    daily_loss_hit = realized_pnl <= -MAX_DAILY_REALIZED_LOSS_DOLLARS

    engine.logger.info(
        "V13.2 DAILY CONTROL | entries=%s soft_cap=%s hard_cap=%s "
        "losses=%s/%s wins=%s breakeven=%s stop_exits=%s "
        "realized_pnl=$%.2f loss_limit=-$%.2f",
        entries_today,
        SOFT_DAILY_ENTRY_CAP,
        HARD_DAILY_ENTRY_CAP,
        losses,
        MAX_DAILY_STOP_OUTS,
        trade_summary["wins"],
        trade_summary["breakeven"],
        trade_summary["stop_exits"],
        realized_pnl,
        MAX_DAILY_REALIZED_LOSS_DOLLARS,
    )

    for trade in trade_summary["trades"]:
        engine.logger.info(
            "V13.2 CLOSED TRADE | symbol=%s qty=%.8g entry=%.4f exit=%.4f "
            "gross_pnl=$%.2f classification=%s exit_kind=%s "
            "loss_counted=%s filled_at=%s",
            trade["symbol"],
            trade["qty"],
            trade["entry"],
            trade["exit"],
            trade["pnl"],
            trade["classification"],
            trade["exit_kind"],
            trade["classification"] == "LOSS",
            trade["filled_at"].isoformat(),
        )

    if trade_summary["unmatched_sell_qty"] > 0:
        engine.logger.warning(
            "V13.2 P/L AUDIT | unmatched sell quantity=%.8g; "
            "likely position opened before today's session",
            trade_summary["unmatched_sell_qty"],
        )

    if losses >= MAX_DAILY_STOP_OUTS or daily_loss_hit:
        reason = (
            f"daily realized-loss limit reached "
            f"(${realized_pnl:.2f} <= -${MAX_DAILY_REALIZED_LOSS_DOLLARS:.2f})"
            if daily_loss_hit
            else f"daily losing-trade limit reached "
                 f"({losses}/{MAX_DAILY_STOP_OUTS})"
        )
        engine.logger.warning(
            "V13.2 NEW ENTRIES PAUSED | %s. "
            "Existing positions remain protected and managed.",
            reason,
        )

        # Always protect/manage existing positions first.
        protected, unprotected = supervisor.reconcile_portfolio()
        if not protected:
            engine.logger.error(
                "V13.2 pause reconciliation found unprotected positions: %s",
                ", ".join(sorted(unprotected)),
            )
            return

        # Continue learning without creating new orders.
        run_diagnostic_only_scan()
        return

    _v13_1_run_supervised_cycle()


supervisor.run_supervised_cycle = run_supervised_cycle_v13_2


# ---------------------------------------------------------------------------
# Startup
# ---------------------------------------------------------------------------

def main() -> None:
    engine.logger.info("==========================================")
    engine.logger.info("BULL. BEAR AND BROKE — V13.2 ADAPTIVE ENTRY CONTROLLER")
    engine.logger.info(
        "Daily entries: soft=%s exceptional hard=%s | losing-trade pause=%s",
        SOFT_DAILY_ENTRY_CAP,
        HARD_DAILY_ENTRY_CAP,
        MAX_DAILY_STOP_OUTS,
    )
    engine.logger.info(
        "Daily realized-loss limit: $%.2f",
        MAX_DAILY_REALIZED_LOSS_DOLLARS,
    )
    engine.logger.info(
        "Exceptional sixth trade: score>=%s RVOL>=%.2f + breakout + "
        "improving MACD + bullish 1m EMA",
        EXCEPTIONAL_MIN_SIGNAL_SCORE,
        EXCEPTIONAL_MIN_RVOL,
    )
    engine.logger.info(
        "Post-stop-out behavior: diagnostic-only scanning ON; broker entries OFF"
    )
    engine.logger.info("==========================================")

    supervisor.main()


if __name__ == "__main__":
    main()
