import streamlit as st
from alpaca.trading.client import TradingClient
from alpaca.trading.requests import MarketOrderRequest
from alpaca.trading.enums import OrderSide, TimeInForce

# 1. Initialize the Live or Paper Client
# Set paper=False ONLY when you are ready to use real money
trading_client = TradingClient(
    st.secrets["ALPACA_API_KEY"], 
    st.secrets["ALPACA_SECRET_KEY"], 
    paper=True 
)

st.divider()
st.subheader("⚡ Framd. Quick Execution Panel")
st.markdown("Manual override for sudden market momentum.")

# 2. The Input Fields
col1, col2 = st.columns(2)
with col1:
    # Auto-capitalizes the ticker so the API doesn't throw an error
    trade_symbol = st.text_input("Ticker Symbol", "NVDA").upper()
with col2:
    trade_qty = st.number_input("Shares", min_value=1, value=5)

# 3. The Action Buttons
col_buy, col_sell = st.columns(2)

# BUY Logic
with col_buy:
    if st.button("🟢 MARKET BUY"):
        try:
            order_data = MarketOrderRequest(
                symbol=trade_symbol,
                qty=trade_qty,
                side=OrderSide.BUY,
                time_in_force=TimeInForce.GTC
            )
            trading_client.submit_order(order_data=order_data)
            st.success(f"Order Sent! Bought {trade_qty} shares of {trade_symbol}.")
        except Exception as e:
            st.error(f"Trade failed: {e}")

# SELL / SHORT Logic
with col_sell:
    if st.button("🔴 MARKET SELL"):
        try:
            order_data = MarketOrderRequest(
                symbol=trade_symbol,
                qty=trade_qty,
                side=OrderSide.SELL,
                time_in_force=TimeInForce.GTC
            )
            trading_client.submit_order(order_data=order_data)
            st.success(f"Order Sent! Sold {trade_qty} shares of {trade_symbol}.")
        except Exception as e:
            st.error(f"Trade failed: {e}")
st.divider()
