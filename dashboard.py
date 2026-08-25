import os
import sqlite3
import pandas as pd
import streamlit as st
from alpaca.trading.client import TradingClient

st.set_page_config(page_title="Bull, Bear & Broke Control Center", layout="wide")
st.title("📈 Bull, Bear & Broke | Trading Terminal")

# Sidebar setup
with st.sidebar:
    if os.path.exists(r"C:\MarketAgents\profile.jpg"):
        st.image(r"C:\MarketAgents\profile.jpg", width=120)
    else:
        st.write("👤 [No Profile Image]")
    
    st.markdown("### Welcome, Ryan Horne")
    st.markdown("Automated Trading System Control Panel")
    st.divider()

# Fetch credentials from Streamlit Secrets
api_key = st.secrets["alpaca"]["api_key"]
secret_key = st.secrets["alpaca"]["secret_key"]

# Portfolio Metrics & Positions
if api_key and secret_key:
    try:
        trading_client = TradingClient(api_key, secret_key, paper=True)
        account = trading_client.get_account()
        open_orders = trading_client.get_orders()
        
        # 5-Column Metrics Bar
        col1, col2, col3, col4, col5 = st.columns(5)
        col1.metric("Portfolio Equity", f"${float(account.equity):,.2f}")
        
        prev_close = float(account.last_equity)
        todays_change = ((float(account.equity) - prev_close) / prev_close) * 100 if prev_close else 0.0
        col2.metric("Today's Change", f"${float(account.equity) - prev_close:,.2f}", f"{todays_change:.2f}%")
        
        col3.metric("Buying Power", f"${float(account.buying_power):,.2f}")
        col4.metric("Cash Balance", f"${float(account.cash):,.2f}")
        col5.metric("Open Orders", f"{len(open_orders)}")
        
        st.divider()
        st.subheader("Current Open Positions")
        positions = trading_client.get_all_positions()
        if positions:
            pos_data = [{
                "Symbol": p.symbol,
                "Qty": p.qty,
                "Entry Price": f"${float(p.avg_entry_price):.2f}",
                "Current Price": f"${float(p.current_price):.2f}",
                "Unrealized PnL": f"${float(p.unrealized_pl):.2f}"
            } for p in positions]
            st.dataframe(pd.DataFrame(pos_data), use_container_width=True)
        else:
            st.info("No open positions currently active.")
    except Exception as e:
        st.error(f"Failed to connect to Alpaca: {e}")
else:
    st.info("Please set up your Alpaca API Key and Secret Key in Streamlit Secrets.")

st.divider()

# Live TradingView Chart Widget
st.subheader("Live Market Chart")
tradingview_html = """
<div class="tradingview-widget-container" style="height:500px;width:100%;">
  <script type="text/javascript" src="https://s3.tradingview.com/tv.js"></script>
  <script type="text/javascript">
  new TradingView.widget({
    "autosize": true,
    "symbol": "NASDAQ:NVDA",
    "interval": "1",
    "timezone": "Etc/UTC",
    "theme": "dark",
    "style": "1",
    "locale": "en",
    "enable_publishing": false,
    "allow_symbol_change": true,
    "container_id": "tradingview_chart"
  });
  </script>
  <div id="tradingview_chart" style="height:500px;width:100%;"></div>
</div>
"""
st.components.v1.html(tradingview_html, height=520)
