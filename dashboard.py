# dashboard_v2.py

import os
import sqlite3
import datetime
import zoneinfo

import pandas as pd
import streamlit as st

from alpaca.trading.client import TradingClient
from alpaca.trading.requests import GetOrdersRequest
from alpaca.trading.enums import QueryOrderStatus

# =====================================================================

# PAGE CONFIGURATION

# =====================================================================

st.set_page_config(
page_title="Bull, Bear & Broke Control Center",
page_icon="📈",
layout="wide",
)

ET = zoneinfo.ZoneInfo("America/New_York")

st.title("📈 Bull, Bear & Broke | Trading Terminal")
st.caption("Automated Trading System Control Center")

# =====================================================================

# CONFIGURATION

# =====================================================================

TARGETS = ["SPY", "QQQ", "TQQQ", "SQQQ"]

ENTRY_COOLDOWN_SECONDS = 300
MAX_OPEN_POSITIONS = 3

LOG_DIR = "logs"
DB_FILE = os.path.join(LOG_DIR, "trading_state.db")
DECISION_LOG_FILE = os.path.join(LOG_DIR, "decisions.log")

PROFILE_IMAGE = os.path.join("assets", "profile.jpg")

# =====================================================================

# SIDEBAR

# =====================================================================

with st.sidebar:

```
if os.path.exists(PROFILE_IMAGE):
    st.image(PROFILE_IMAGE, width=140)
else:
    st.markdown("# 👤")

st.markdown("### Bull, Bear & Broke")
st.caption("Automated Trading Control Center")

st.divider()

st.subheader("⚙️ Dashboard Settings")

selected_symbol = st.selectbox(
    "Chart Symbol",
    TARGETS,
    index=0,
)

auto_refresh_seconds = st.selectbox(
    "Auto Refresh",
    [15, 30, 60, 120],
    index=2,
    format_func=lambda x: f"{x} seconds",
)

decision_count = st.slider(
    "Recent Decisions",
    min_value=10,
    max_value=100,
    value=30,
    step=10,
)

st.divider()

st.subheader("System")

if os.path.exists(DB_FILE):
    st.success("SQLite State: Connected")
else:
    st.warning("SQLite State: Not Found")

if os.path.exists(DECISION_LOG_FILE):
    st.success("Decision Log: Connected")
else:
    st.warning("Decision Log: Not Found")

st.caption(
    "For separate Render services, migrate "
    "shared bot state to PostgreSQL."
)
```

# =====================================================================

# ALPACA CLIENT

# =====================================================================

def get_secret_value(section, key, env_name):

```
try:
    return st.secrets[section][key]
except Exception:
    return os.environ.get(env_name)
```

def get_alpaca_client():

```
api_key = get_secret_value(
    "alpaca",
    "api_key",
    "APCA_API_KEY_ID",
)

secret_key = get_secret_value(
    "alpaca",
    "secret_key",
    "APCA_API_SECRET_KEY",
)

if not api_key or not secret_key:
    return None, "Missing Alpaca API credentials."

paper_trading = (
    os.environ.get(
        "PAPER_TRADING",
        "true",
    ).lower() == "true"
)

try:

    client = TradingClient(
        api_key,
        secret_key,
        paper=paper_trading,
    )

    return client, None

except Exception as e:
    return None, str(e)
```

# =====================================================================

# SQLITE HELPERS

# =====================================================================

def sqlite_available():

```
return os.path.exists(DB_FILE)
```

def get_db_connection():

```
if not sqlite_available():
    return None

try:
    return sqlite3.connect(DB_FILE)
except Exception:
    return None
```

def get_db_state(key):

```
conn = get_db_connection()

if conn is None:
    return None

try:

    cursor = conn.cursor()

    cursor.execute(
        "SELECT value FROM state WHERE key = ?",
        (key,),
    )

    row = cursor.fetchone()

    return row[0] if row else None

except Exception:

    return None

finally:

    conn.close()
```

def get_all_state():

```
conn = get_db_connection()

if conn is None:
    return {}

try:

    query = "SELECT key, value FROM state"

    df = pd.read_sql_query(
        query,
        conn,
    )

    return dict(
        zip(
            df["key"],
            df["value"],
        )
    )

except Exception:

    return {}

finally:

    conn.close()
```

def get_trade_journal(limit=25):

```
conn = get_db_connection()

if conn is None:
    return pd.DataFrame()

try:

    query = """
    SELECT
        id,
        timestamp,
        symbol,
        side,
        qty,
        entry_price,
        stop_loss,
        take_profit,
        status
    FROM trade_journal
    ORDER BY id DESC
    LIMIT ?
    """

    df = pd.read_sql_query(
        query,
        conn,
        params=(limit,),
    )

    return df

except Exception:

    return pd.DataFrame()

finally:

    conn.close()
```

# =====================================================================

# COOLDOWN HELPERS

# =====================================================================

def get_cooldown_status(symbol):

```
value = get_db_state(
    f"last_entry_time:{symbol}"
)

if not value:

    return {
        "active": False,
        "remaining": 0,
        "status": "Ready",
    }

try:

    last_entry = float(value)

    elapsed = (
        datetime.datetime.now().timestamp()
        - last_entry
    )

    remaining = (
        ENTRY_COOLDOWN_SECONDS
        - elapsed
    )

    if remaining > 0:

        return {
            "active": True,
            "remaining": int(remaining),
            "status": f"{int(remaining)}s",
        }

    return {
        "active": False,
        "remaining": 0,
        "status": "Ready",
    }

except Exception:

    return {
        "active": False,
        "remaining": 0,
        "status": "Unknown",
    }
```

# =====================================================================

# DECISION LOG

# =====================================================================

def get_recent_decisions(limit=30):

```
if not os.path.exists(
    DECISION_LOG_FILE
):
    return []

try:

    with open(
        DECISION_LOG_FILE,
        "r",
        encoding="utf-8",
        errors="ignore",
    ) as file:

        lines = file.readlines()

    lines = [
        line.strip()
        for line in lines
        if line.strip()
    ]

    return lines[-limit:][::-1]

except Exception:

    return []
```

# =====================================================================

# FORMAT HELPERS

# =====================================================================

def format_currency(value):

```
try:
    return f"${float(value):,.2f}"
except Exception:
    return "$0.00"
```

def format_percent(value):

```
try:
    return f"{float(value):.2f}%"
except Exception:
    return "0.00%"
```

def status_badge(status):

```
if status == "ONLINE":
    return "🟢 ONLINE"

if status == "OFFLINE":
    return "🔴 OFFLINE"

if status == "TRIPPED":
    return "🔴 TRIPPED"

return "🟡 UNKNOWN"
```

# =====================================================================

# CONNECT TO ALPACA

# =====================================================================

trading_client, alpaca_error = (
get_alpaca_client()
)

if alpaca_error:

```
st.error(
    f"❌ Failed to initialize Alpaca: "
    f"{alpaca_error}"
)

st.stop()
```

# =====================================================================

# FETCH LIVE BROKER DATA

# =====================================================================

try:

```
account = (
    trading_client.get_account()
)

positions = (
    trading_client.get_all_positions()
)

order_request = GetOrdersRequest(
    status=QueryOrderStatus.OPEN,
    nested=False,
)

open_orders = (
    trading_client.get_orders(
        filter=order_request
    )
)

clock = trading_client.get_clock()
```

except Exception as e:

```
st.error(
    f"❌ Failed to connect to Alpaca: {e}"
)

st.stop()
```

# =====================================================================

# BOT / MARKET STATUS BAR

# =====================================================================

current_time = datetime.datetime.now(
ET
)

market_status = (
"OPEN"
if clock.is_open
else "CLOSED"
)

paper_trading = (
os.environ.get(
"PAPER_TRADING",
"true",
).lower() == "true"
)

circuit_breaker_date = get_db_state(
"circuit_breaker_date"
)

today_str = current_time.strftime(
"%Y-%m-%d"
)

if circuit_breaker_date == today_str:

```
circuit_status = "TRIPPED"
```

else:

```
circuit_status = "ONLINE"
```

status_col1, status_col2, status_col3, status_col4 = (
st.columns(4)
)

status_col1.metric(
"🤖 Bot Status",
status_badge(circuit_status),
)

status_col2.metric(
"🏛️ Market",
f"🟢 {market_status}"
if market_status == "OPEN"
else "🔴 CLOSED",
)

status_col3.metric(
"💼 Trading Mode",
"🧪 PAPER"
if paper_trading
else "💰 LIVE",
)

status_col4.metric(
"🕒 Eastern Time",
current_time.strftime(
"%I:%M:%S %p"
),
)

st.divider()

# =====================================================================

# PORTFOLIO METRICS

# =====================================================================

st.subheader("💰 Portfolio Overview")

equity = float(account.equity)
last_equity = float(account.last_equity)
buying_power = float(account.buying_power)
cash = float(account.cash)

daily_pnl = equity - last_equity

daily_pnl_pct = (
(daily_pnl / last_equity) * 100
if last_equity
else 0.0
)

metric_cols = st.columns(6)

metric_cols[0].metric(
"Portfolio Equity",
format_currency(equity),
)

metric_cols[1].metric(
"Today's P&L",
format_currency(daily_pnl),
format_percent(daily_pnl_pct),
)

metric_cols[2].metric(
"Buying Power",
format_currency(buying_power),
)

metric_cols[3].metric(
"Cash",
format_currency(cash),
)

metric_cols[4].metric(
"Open Positions",
f"{len(positions)} / {MAX_OPEN_POSITIONS}",
)

metric_cols[5].metric(
"Pending Orders",
len(open_orders),
)

st.divider()

# =====================================================================

# CHART

# =====================================================================

st.subheader(
f"📊 Live Market Chart — {selected_symbol}"
)

tradingview_html = f"""

<div
    class="tradingview-widget-container"
    style="height:550px;width:100%;"
>
    <div
        id="tradingview_chart"
        style="height:550px;width:100%;"
    ></div>

```
<script
    type="text/javascript"
    src="https://s3.tradingview.com/tv.js"
></script>

<script type="text/javascript">

new TradingView.widget({{

    "autosize": true,

    "symbol": "AMEX:{selected_symbol}",

    "interval": "5",

    "timezone": "America/New_York",

    "theme": "dark",

    "style": "1",

    "locale": "en",

    "enable_publishing": false,

    "allow_symbol_change": true,

    "container_id": "tradingview_chart"

}});

</script>
```

</div>
"""

st.components.v1.html(
tradingview_html,
height=570,
)

st.divider()

# =====================================================================

# OPEN POSITIONS

# =====================================================================

left_col, right_col = st.columns(2)

with left_col:

```
st.subheader("💼 Current Positions")

if positions:

    position_data = []

    total_unrealized_pnl = 0.0

    for position in positions:

        unrealized_pnl = float(
            position.unrealized_pl
        )

        total_unrealized_pnl += (
            unrealized_pnl
        )

        position_data.append(
            {
                "Symbol": position.symbol,
                "Quantity": float(
                    position.qty
                ),
                "Entry": float(
                    position.avg_entry_price
                ),
                "Current": float(
                    position.current_price
                ),
                "Market Value": float(
                    position.market_value
                ),
                "Unrealized P&L": (
                    unrealized_pnl
                ),
                "Unrealized %": float(
                    position.unrealized_plpc
                )
                * 100,
            }
        )

    st.metric(
        "Total Unrealized P&L",
        format_currency(
            total_unrealized_pnl
        ),
    )

    position_df = pd.DataFrame(
        position_data
    )

    st.dataframe(
        position_df,
        use_container_width=True,
        hide_index=True,
        column_config={
            "Entry": st.column_config.NumberColumn(
                format="$%.2f"
            ),
            "Current": st.column_config.NumberColumn(
                format="$%.2f"
            ),
            "Market Value": (
                st.column_config.NumberColumn(
                    format="$%.2f"
                )
            ),
            "Unrealized P&L": (
                st.column_config.NumberColumn(
                    format="$%.2f"
                )
            ),
            "Unrealized %": (
                st.column_config.NumberColumn(
                    format="%.2f%%"
                )
            ),
        },
    )

else:

    st.info(
        "No open positions currently active."
    )
```

# =====================================================================

# PENDING ORDERS

# =====================================================================

with right_col:

```
st.subheader("📋 Pending Orders")

if open_orders:

    order_data = []

    for order in open_orders:

        order_data.append(
            {
                "Symbol": order.symbol,
                "Side": str(order.side),
                "Qty": order.qty,
                "Type": str(order.type),
                "Status": str(order.status),
                "Limit Price": (
                    float(order.limit_price)
                    if order.limit_price
                    else None
                ),
                "Created": str(
                    order.created_at
                ),
            }
        )

    order_df = pd.DataFrame(
        order_data
    )

    st.dataframe(
        order_df,
        use_container_width=True,
        hide_index=True,
    )

else:

    st.info(
        "No pending orders."
    )
```

st.divider()

# =====================================================================

# BOT COOLDOWNS & CIRCUIT BREAKER

# =====================================================================

st.subheader("🤖 Bot Risk & Entry Status")

risk_col1, risk_col2 = st.columns(
[1, 2]
)

with risk_col1:

```
st.markdown(
    "### 🚨 Circuit Breaker"
)

if circuit_breaker_date == today_str:

    st.error(
        "TRIPPED FOR TODAY"
    )

    st.caption(
        "The bot recorded a circuit "
        "breaker event for the current "
        "UTC trading date."
    )

else:

    st.success(
        "NORMAL"
    )

    st.caption(
        "No circuit breaker trip "
        "recorded for today."
    )
```

with risk_col2:

```
st.markdown(
    "### ⏳ Symbol Cooldowns"
)

cooldown_data = []

for symbol in TARGETS:

    cooldown = (
        get_cooldown_status(
            symbol
        )
    )

    cooldown_data.append(
        {
            "Symbol": symbol,
            "Status": (
                "🔴 COOLDOWN"
                if cooldown["active"]
                else "🟢 READY"
            ),
            "Remaining": (
                f"{cooldown['remaining']} seconds"
                if cooldown["active"]
                else "Ready for entry"
            ),
        }
    )

cooldown_df = pd.DataFrame(
    cooldown_data
)

st.dataframe(
    cooldown_df,
    use_container_width=True,
    hide_index=True,
)
```

st.divider()

# =====================================================================

# RECENT BOT DECISIONS

# =====================================================================

st.subheader("🧠 Recent Bot Decisions")

decisions = get_recent_decisions(
decision_count
)

if decisions:

```
decision_df = pd.DataFrame(
    {
        "Decision": decisions
    }
)

st.dataframe(
    decision_df,
    use_container_width=True,
    hide_index=True,
    height=400,
)
```

else:

```
st.info(
    "No bot decision log is currently "
    "available to this dashboard."
)
```

st.divider()

# =====================================================================

# TRADE JOURNAL

# =====================================================================

st.subheader("📒 Recent Trade Journal")

trade_journal = get_trade_journal(
limit=50
)

if not trade_journal.empty:

```
st.dataframe(
    trade_journal,
    use_container_width=True,
    hide_index=True,
    column_config={
        "entry_price": (
            st.column_config.NumberColumn(
                "Entry Price",
                format="$%.2f",
            )
        ),
        "stop_loss": (
            st.column_config.NumberColumn(
                "Stop Loss",
                format="$%.2f",
            )
        ),
        "take_profit": (
            st.column_config.NumberColumn(
                "Take Profit",
                format="$%.2f",
            )
        ),
    },
)
```

else:

```
st.info(
    "No trade journal records available."
)
```

# =====================================================================

# DATABASE STATE DEBUG PANEL

# =====================================================================

with st.expander(
"🔧 Bot State / Debug Information"
):

```
state = get_all_state()

if state:

    state_df = pd.DataFrame(
        list(state.items()),
        columns=[
            "Key",
            "Value",
        ],
    )

    st.dataframe(
        state_df,
        use_container_width=True,
        hide_index=True,
    )

else:

    st.info(
        "No shared SQLite state found."
    )
```

# =====================================================================

# FOOTER

# =====================================================================

st.divider()

footer_left, footer_right = st.columns(
2
)

with footer_left:

```
st.caption(
    "Bull, Bear & Broke • "
    "Automated Trading Terminal"
)
```

with footer_right:

```
st.caption(
    f"Last dashboard render: "
    f"{datetime.datetime.now(ET).strftime('%Y-%m-%d %I:%M:%S %p ET')}"
)
```

# =====================================================================

# AUTO REFRESH

# =====================================================================

st.markdown(
f""" <script>
setTimeout(function(){{
window.parent.location.reload();
}}, {auto_refresh_seconds * 1000}); </script>
""",
unsafe_allow_html=True,
)
