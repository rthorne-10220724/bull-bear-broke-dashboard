import os
import sqlite3
import pandas as pd
import streamlit as st
from alpaca.trading.client import TradingClient

st.set_page_config(page_title="Bull, Bear & Broke Control Center", layout="wide")
st.title("📈 Bull, Bear & Broke | Trading Terminal")

# Sidebar - Alpaca Credentials Setup
st.sidebar.header("Alpaca Credentials")
api_key = st.sidebar.text_input("Alpaca API Key", type="password")
secret_key = st.sidebar.text_input("Alpaca Secret Key", type="password")

if api_key and secret_key:
    try:
        trading_client = TradingClient(api_key, secret_key, paper=True)
        account = trading_client.get_account()
        
        col1, col2, col3 = st.columns(3)
        col1.metric("Portfolio Equity", f"${float(account.equity):,.2f}")
        col2.metric("Buying Power", f"${float(account.buying_power):,.2f}")
        col3.metric("Cash Balance", f"${float(account.cash):,.2f}")
        
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