"""
BULL. BEAR AND BROKE — V13.1 SAFETY SUPERVISOR
===============================================

Render entrypoint for the existing main_v6.py strategy engine.

V13.1 keeps the discovery/scoring architecture but hardens live paper execution
against the churn seen in HOOD/MARA/SOFI:
- 30-minute symbol cooldown
- maximum 2 entries per symbol per UTC day
- second entry requires a fresh breakout + improving 15m MACD + bullish 1m EMA
- minimum stop distance of 0.35% even when 1m ATR is microscopic
- target remains 2.5R from the effective stop distance
- risk/trade reduced to 0.50%, max position to 10%, daily breaker to 2%
- BUY-only pending-entry accounting
- extended 1m history so 15m MACD has enough bars
- orphaned-position OCO repair uses the same V13.1 stop/target logic

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
    GetOrdersRequest,
    LimitOrderRequest,
    TakeProfitRequest,
    StopLossRequest,
)
from alpaca.trading.enums import (
    OrderSide,
    TimeInForce,
    OrderClass,
    QueryOrderStatus,
)


# ---------------------------------------------------------------------------
# V13.1 safety calibration (all values remain overrideable in Render env)
# ---------------------------------------------------------------------------

engine.RISK_PER_TRADE_PCT = float(os.getenv("RISK_PER_TRADE_PCT", "0.005"))
engine.MAX_POSITION_PCT = float(os.getenv("MAX_POSITION_PCT", "0.10"))
engine.MAX_DAILY_DRAWDOWN_PCT = float(os.getenv("MAX_DAILY_DRAWDOWN_PCT", "0.02"))

ENTRY_COOLDOWN_SECONDS = int(os.getenv("ENTRY_COOLDOWN_SECONDS", "1800"))
MAX_SYMBOL_ENTRIES_PER_DAY = int(os.getenv("MAX_SYMBOL_ENTRIES_PER_DAY", "2"))
MIN_STOP_DISTANCE_PCT = float(os.getenv("MIN_STOP_DISTANCE_PCT", "0.0035"))
TARGET_R_MULTIPLE = float(
    os.getenv(
        "TARGET_R_MULTIPLE",
        str(engine.ATR_MULTIPLIER_TARGET / engine.ATR_MULTIPLIER_STOP),
    )
)

MIN_1M_ATR_PCT = float(os.getenv("MIN_1M_ATR_PCT", "0.08"))
MACD_HISTORY_DAYS = int(os.getenv("MACD_HISTORY_DAYS", "7"))
MACD_MIN_1M_BARS = int(os.getenv("MACD_MIN_1M_BARS", "900"))

AUTO_REPAIR_ORPHANED_EXITS = (
    os.getenv(
        "AUTO_REPAIR_ORPHANED_EXITS",
        "true" if engine.PAPER_TRADING else "false",
    ).strip().lower() == "true"
)
BLOCK_NEW_ENTRIES_IF_UNPROTECTED = (
    os.getenv("BLOCK_NEW_ENTRIES_IF_UNPROTECTED", "true").strip().lower() == "true"
)
REPAIR_SETTLE_SECONDS = 2
REPAIR_CANCEL_TIMEOUT_SECONDS = int(
    os.getenv("REPAIR_CANCEL_TIMEOUT_SECONDS", "15")
)
REPAIR_CANCEL_POLL_SECONDS = float(
    os.getenv("REPAIR_CANCEL_POLL_SECONDS", "0.5")
)


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        number = float(value)
        return number if np.isfinite(number) else default
    except (TypeError, ValueError):
        return default


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


def _get_open_orders_nested() -> List[Any]:
    """Fetch open orders with OCO/bracket child legs attached."""
    request = GetOrdersRequest(
        status=QueryOrderStatus.OPEN,
        nested=True,
    )
    return list(engine.trading_client.get_orders(filter=request) or [])


def _cancelable_sell_roots(order: Any, symbol: str) -> List[Any]:
    """Find the highest sell nodes, never a filled BUY bracket parent."""
    node_symbol = str(getattr(order, "symbol", "") or "").upper()
    node_side = _enum_text(getattr(order, "side", None))
    if node_symbol == symbol and node_side == "sell":
        return [order]

    roots: List[Any] = []
    for leg in getattr(order, "legs", None) or []:
        roots.extend(_cancelable_sell_roots(leg, symbol))
    return roots


def _open_sell_order_roots(symbol: str) -> List[Any]:
    """Return cancelable open orders whose trees reserve this position."""
    roots: List[Any] = []
    for order in _get_open_orders_nested():
        roots.extend(_cancelable_sell_roots(order, symbol))
    return roots


def _cancel_existing_exits_and_wait(symbol: str) -> bool:
    """Release shares held by orphaned exits before creating one OCO group."""
    roots = _open_sell_order_roots(symbol)
    if not roots:
        return True

    cancel_ids = []
    for order in roots:
        order_id = getattr(order, "id", None)
        if order_id is None or order_id in cancel_ids:
            continue
        cancel_ids.append(order_id)

    for order_id in cancel_ids:
        try:
            engine.trading_client.cancel_order_by_id(order_id)
            engine.logger.warning(
                "[%s] canceled orphaned exit before OCO repair | order_id=%s",
                symbol,
                order_id,
            )
        except Exception as exc:
            # Cancellation can race an order-state update. The polling check
            # below is authoritative.
            engine.logger.warning(
                "[%s] exit cancel returned an error; verifying state | "
                "order_id=%s error=%s",
                symbol,
                order_id,
                exc,
            )

    deadline = time.monotonic() + REPAIR_CANCEL_TIMEOUT_SECONDS
    while time.monotonic() < deadline:
        if not _open_sell_order_roots(symbol):
            return True
        time.sleep(REPAIR_CANCEL_POLL_SECONDS)

    remaining = [
        str(getattr(order, "id", "unknown"))
        for order in _open_sell_order_roots(symbol)
    ]
    engine.logger.error(
        "[%s] OCO repair stopped: existing exits still reserve shares | orders=%s",
        symbol,
        ",".join(remaining) or "unknown",
    )
    return False


# ---------------------------------------------------------------------------
# Daily per-symbol attempt state
# ---------------------------------------------------------------------------

def _symbol_entry_key(symbol: str) -> str:
    safe = symbol.replace("/", "_").upper()
    return f"v13_1_symbol_entries:{engine.utc_date_string()}:{safe}"


def get_symbol_entry_count(symbol: str) -> int:
    value = engine.get_state(_symbol_entry_key(symbol))
    try:
        return int(value) if value else 0
    except (TypeError, ValueError):
        return 0


def increment_symbol_entry_count(symbol: str) -> None:
    engine.set_state(
        _symbol_entry_key(symbol),
        str(get_symbol_entry_count(symbol) + 1),
    )


_original_set_cooldown = engine.set_cooldown


def set_cooldown_v13_1(symbol: str) -> None:
    # Called only after a broker-accepted entry in main_v6.place_stock_bracket.
    _original_set_cooldown(symbol)
    increment_symbol_entry_count(symbol)


def is_on_cooldown_v13_1(symbol: str) -> bool:
    if get_symbol_entry_count(symbol) >= MAX_SYMBOL_ENTRIES_PER_DAY:
        return True

    value = engine.get_state(engine.cooldown_key(symbol))
    if not value:
        return False

    try:
        return (time.time() - float(value)) < ENTRY_COOLDOWN_SECONDS
    except (TypeError, ValueError):
        return False


# ---------------------------------------------------------------------------
# BUY-only pending entries
# ---------------------------------------------------------------------------

def get_pending_entry_symbols(orders: List[Any]) -> Set[str]:
    symbols: Set[str] = set()
    for order in _iter_order_tree(orders):
        symbol = str(getattr(order, "symbol", "") or "").upper()
        side = _enum_text(getattr(order, "side", None))
        if symbol and side == "buy":
            symbols.add(symbol)
    return symbols


# ---------------------------------------------------------------------------
# Extended 1m history + calibrated data quality
# ---------------------------------------------------------------------------

_last_signal_symbol: Optional[str] = None


def fetch_1m_bars_with_macd_history(
    ticker: str,
    limit: int = 180,
    retries: int = 3,
) -> Optional[pd.DataFrame]:
    global _last_signal_symbol
    _last_signal_symbol = ticker.upper()

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

            return df.tail(max(limit, MACD_MIN_1M_BARS))

        except Exception as exc:
            engine.logger.warning(
                "[%s] extended 1m data attempt %s/%s failed: %s",
                ticker, attempt + 1, retries, exc,
            )
            if attempt < retries - 1:
                time.sleep(1.5)

    return None


def passes_data_quality(
    df_1m: pd.DataFrame,
    indicators: Dict[str, Any],
) -> Tuple[bool, str]:
    if len(df_1m) < 60:
        return False, f"insufficient 1m history ({len(df_1m)} < 60)"

    required = {"Open", "High", "Low", "Close", "Volume"}
    if not required.issubset(df_1m.columns):
        return False, "missing OHLCV columns"

    if not np.isfinite(df_1m[list(required)].tail(30).to_numpy()).all():
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
        return False, f"insufficient RVOL ({rvol:.2f} < {engine.MIN_RVOL:.2f})"
    if atr_pct < MIN_1M_ATR_PCT:
        return False, (
            f"volatility too low (1m ATR%={atr_pct:.3f} < {MIN_1M_ATR_PCT:.3f})"
        )

    return True, f"data/volatility valid (RVOL={rvol:.2f}, 1m ATR%={atr_pct:.3f})"


# ---------------------------------------------------------------------------
# Fresh-setup requirement on a second attempt
# ---------------------------------------------------------------------------

_original_evaluate_signal = engine.evaluate_signal


def evaluate_signal_v13_1(indicators: Dict[str, Any]):
    result = _original_evaluate_signal(indicators)
    symbol = (_last_signal_symbol or "").upper()

    if not result.valid or not symbol:
        return result

    attempts = get_symbol_entry_count(symbol)
    if attempts >= MAX_SYMBOL_ENTRIES_PER_DAY:
        return engine.SignalResult(
            valid=False,
            score=result.score,
            reasons=result.reasons,
            rejection=(
                f"per-symbol daily entry limit reached "
                f"({attempts}/{MAX_SYMBOL_ENTRIES_PER_DAY})"
            ),
        )

    if attempts >= 1:
        fresh_reentry = (
            bool(indicators.get("breakout"))
            and bool(indicators.get("macd_improving"))
            and bool(indicators.get("short_term_bullish"))
        )
        if not fresh_reentry:
            return engine.SignalResult(
                valid=False,
                score=result.score,
                reasons=result.reasons,
                rejection=(
                    "re-entry requires fresh breakout + improving 15m MACD "
                    "+ bullish 1m EMA structure"
                ),
            )

    return result


# ---------------------------------------------------------------------------
# Stop floor, target and position sizing
# ---------------------------------------------------------------------------

def effective_stop_distance(price: float, atr: float) -> float:
    if price <= 0 or atr <= 0:
        return 0.0
    return max(
        engine.ATR_MULTIPLIER_STOP * atr,
        price * MIN_STOP_DISTANCE_PCT,
    )


def calculate_exit_prices_v13_1(
    entry: float,
    atr: float,
) -> Tuple[float, float]:
    distance = effective_stop_distance(entry, atr)
    stop = entry - distance
    target = entry + distance * TARGET_R_MULTIPLE
    return round(stop, 2), round(target, 2)


def calculate_position_size_v13_1(
    equity: float,
    buying_power: float,
    price: float,
    atr: float,
) -> float:
    if equity <= 0 or buying_power <= 0 or price <= 0 or atr <= 0:
        return 0.0

    distance = effective_stop_distance(price, atr)
    if distance <= 0:
        return 0.0

    risk_qty = (equity * engine.RISK_PER_TRADE_PCT) / distance
    cap_qty = (equity * engine.MAX_POSITION_PCT) / price
    bp_qty = (buying_power * 0.95) / price
    return float(max(0, int(min(risk_qty, cap_qty, bp_qty))))


# Apply runtime patches before run_cycle is ever called.
engine.set_cooldown = set_cooldown_v13_1
engine.is_on_cooldown = is_on_cooldown_v13_1
engine.get_pending_symbols = get_pending_entry_symbols
engine.fetch_1m_bars = fetch_1m_bars_with_macd_history
engine.passes_data_quality = passes_data_quality
engine.evaluate_signal = evaluate_signal_v13_1
engine.calculate_exit_prices = calculate_exit_prices_v13_1
engine.calculate_position_size = calculate_position_size_v13_1


# ---------------------------------------------------------------------------
# Portfolio protection / orphan repair
# ---------------------------------------------------------------------------

def _sell_orders_by_symbol(orders: List[Any]) -> Dict[str, List[Any]]:
    grouped: Dict[str, List[Any]] = {}
    for order in _iter_order_tree(orders):
        symbol = str(getattr(order, "symbol", "") or "").upper()
        if symbol and _enum_text(getattr(order, "side", None)) == "sell":
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
        if limit_price not in (None, "", "0", 0) and order_type != "stop_limit":
            has_target = True

    return has_stop, has_target


def _latest_atr(symbol: str) -> float:
    df = engine.fetch_1m_bars(symbol, limit=180)
    if df is None or df.empty or len(df) < engine.ATR_WINDOW + 5:
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
    current = _safe_float(getattr(position, "current_price", 0))

    if not symbol or "/" in symbol or qty <= 0 or current <= 0:
        return False

    rounded_qty = round(qty)
    if abs(qty - rounded_qty) > 1e-8:
        engine.logger.error(
            "[%s] OCO repair skipped: fractional qty %.8f", symbol, qty
        )
        return False

    atr = _latest_atr(symbol)
    if atr <= 0:
        return False

    stop_price, target_price = calculate_exit_prices_v13_1(current, atr)
    stop_price = max(0.01, min(stop_price, round(current - 0.01, 2)))
    target_price = max(round(current + 0.01, 2), target_price)

    try:
        # A non-nested response can expose the profit target while hiding its
        # OCO stop sibling. Confirm the complete tree before changing orders.
        nested = _sell_orders_by_symbol(_get_open_orders_nested())
        has_stop, has_target = _classify_protection(nested.get(symbol, []))
        if has_stop and has_target:
            engine.logger.info(
                "[%s] OCO repair not needed: nested stop and target confirmed.",
                symbol,
            )
            return True

        # An orphan target reserves every share, so Alpaca will reject another
        # full-quantity OCO until that reservation has actually been released.
        if not _cancel_existing_exits_and_wait(symbol):
            return False

        # Cancel/fill processing is asynchronous. Refresh the position before
        # sizing the replacement so stale quantity can never create an oversell.
        fresh_position = engine.get_open_positions().get(symbol)
        if fresh_position is None:
            engine.logger.warning(
                "[%s] OCO repair ended: position no longer exists.", symbol
            )
            return True

        fresh_qty = _safe_float(getattr(fresh_position, "qty", 0))
        fresh_rounded_qty = round(fresh_qty)
        if fresh_qty <= 0:
            engine.logger.warning(
                "[%s] OCO repair ended: position quantity is now zero.", symbol
            )
            return True
        if abs(fresh_qty - fresh_rounded_qty) > 1e-8:
            engine.logger.error(
                "[%s] OCO repair stopped after cancel: fractional qty %.8f",
                symbol,
                fresh_qty,
            )
            return False

        rounded_qty = fresh_rounded_qty
        refreshed_current = _safe_float(
            getattr(fresh_position, "current_price", current), current
        )
        if refreshed_current > 0:
            current = refreshed_current
            stop_price, target_price = calculate_exit_prices_v13_1(current, atr)
            stop_price = max(0.01, min(stop_price, round(current - 0.01, 2)))
            target_price = max(round(current + 0.01, 2), target_price)

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
            "[%s] REPAIRED V13.1 OCO | qty=%s current=%.2f stop=%.2f "
            "target=%.2f order_id=%s",
            symbol, int(rounded_qty), current, stop_price, target_price,
            getattr(response, "id", "unknown"),
        )
        return True
    except Exception as exc:
        engine.logger.exception("[%s] OCO repair FAILED: %s", symbol, exc)
        return False


def reconcile_portfolio() -> Tuple[bool, Set[str]]:
    positions = engine.get_open_positions()
    orders = _get_open_orders_nested()
    sell_orders = _sell_orders_by_symbol(orders)

    engine.logger.info("========== PORTFOLIO RECONCILIATION ==========")

    if not positions:
        engine.logger.info("No open positions. Portfolio protection status: CLEAR")
        return True, set()

    unprotected: Set[str] = set()

    for symbol, position in positions.items():
        symbol = str(symbol).upper()
        has_stop, has_target = _classify_protection(sell_orders.get(symbol, []))

        if has_stop and has_target:
            continue

        unprotected.add(symbol)
        engine.logger.warning(
            "[%s] UNPROTECTED POSITION | stop=%s target=%s",
            symbol,
            "YES" if has_stop else "NO",
            "YES" if has_target else "NO",
        )

        if AUTO_REPAIR_ORPHANED_EXITS and _repair_oco_for_position(position):
            unprotected.discard(symbol)

    if AUTO_REPAIR_ORPHANED_EXITS:
        time.sleep(REPAIR_SETTLE_SECONDS)
        fresh = _sell_orders_by_symbol(_get_open_orders_nested())
        unprotected = {
            str(symbol).upper()
            for symbol in positions
            if not all(_classify_protection(fresh.get(str(symbol).upper(), [])))
        }

    if unprotected:
        engine.logger.error(
            "PORTFOLIO RECONCILIATION FAILED | unprotected=%s",
            ", ".join(sorted(unprotected)),
        )
        return False, unprotected

    engine.logger.info("PORTFOLIO RECONCILIATION PASSED")
    return True, set()


def run_supervised_cycle() -> None:
    protected, unprotected = reconcile_portfolio()
    if BLOCK_NEW_ENTRIES_IF_UNPROTECTED and not protected:
        engine.logger.error(
            "NEW ENTRIES BLOCKED: unprotected=%s",
            ", ".join(sorted(unprotected)),
        )
        return
    engine.run_cycle()


def main() -> None:
    engine.init_db()

    engine.logger.info("==========================================")
    engine.logger.info("BULL. BEAR AND BROKE — V13.1 SAFETY SUPERVISOR")
    engine.logger.info("Paper trading: %s", engine.PAPER_TRADING)
    engine.logger.info(
        "Risk %.2f%% | position cap %.1f%% | daily breaker %.1f%%",
        engine.RISK_PER_TRADE_PCT * 100,
        engine.MAX_POSITION_PCT * 100,
        engine.MAX_DAILY_DRAWDOWN_PCT * 100,
    )
    engine.logger.info(
        "Cooldown %ss | max symbol entries %s | minimum stop %.2f%%",
        ENTRY_COOLDOWN_SECONDS,
        MAX_SYMBOL_ENTRIES_PER_DAY,
        MIN_STOP_DISTANCE_PCT * 100,
    )
    engine.logger.info("==========================================")

    while True:
        try:
            run_supervised_cycle()
        except KeyboardInterrupt:
            engine.logger.info("Shutdown requested.")
            break
        except Exception as exc:
            engine.logger.exception("Supervisor cycle failure: %s", exc)

        engine.logger.info("Sleeping %s seconds...", engine.CYCLE_SECONDS)
        time.sleep(engine.CYCLE_SECONDS)


if __name__ == "__main__":
    main()
