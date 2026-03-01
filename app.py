import streamlit as st
from alpaca.trading.client import TradingClient

st.title("Alpaca Diagnostic Tool")

try:
    # 1. Test if Streamlit can even see the keys
    api_key = st.secrets["ALPACA_API_KEY"]
    sec_key = st.secrets["ALPACA_SECRET_KEY"]
    st.success("✅ Streamlit Secrets found!")
    st.write(f"Your API Key starts with: {api_key[:4]}...")
    
    # 2. Test if Alpaca accepts the keys
    try:
        client = TradingClient(api_key, sec_key, paper=True)
        acc = client.get_account()
        st.success(f"✅ Alpaca Connected! Your Buying Power is: ${float(acc.buying_power):,.2f}")
    except Exception as e:
        st.error(f"❌ Alpaca rejected the keys. Error message from Alpaca: {e}")
        
except Exception as e:
    st.error(f"❌ Streamlit cannot find the keys. Error: {e}")
