import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import nltk
from textblob import TextBlob
from sklearn.ensemble import RandomForestClassifier

# Alpaca Integration
from alpaca.trading.client import TradingClient
from alpaca.trading.requests import MarketOrderRequest
from alpaca.trading.enums import OrderSide, TimeInForce

# --- FIX FOR STREAMLIT CLOUD NLTK ---
@st.cache_resource
def download_nltk_data():
    try:
        nltk.download("punkt", quiet=True)
        nltk.download("punkt_tab", quiet=True)
    except:
        pass
download_nltk_data()

# --- ALPACA CONNECTION SHIELD ---
def get_alpaca_client():
    try:
        # .strip() automatically removes accidental invisible spaces!
        api_key = st.secrets.get("ALPACA_API_KEY", "").strip()
        sec_key = st.secrets.get("ALPACA_SECRET_KEY", "").strip()
        
        if not api_key or not sec_key:
            return None, "Keys are missing from Streamlit Secrets."
            
        if api_key.startswith("AK"):
            return None, "LIVE key detected. Please use PAPER keys (starts with PK) for safety."
            
        client = TradingClient(api_key, sec_key, paper=True)
        # Quick test to make sure keys are valid
        client.get_account() 
        return client, None
    except Exception as e:
        return None, f"Alpaca rejected keys: {e}"

def execute_alpaca_order(ticker, side, qty):
    client, error = get_alpaca_client()
    if not client: return False, error
    
    try:
        order_data = MarketOrderRequest(
            symbol=ticker, qty=qty, side=side, time_in_force=TimeInForce.GTC
        )
        client.submit_order(order_data=order_data)
        return True, None
    except Exception as e:
        return False, str(e)

# --- MARKET DATA & AI BRAIN ---
def fetch_market_data(ticker, period="2y"):
    data = yf.download(ticker, period=period, progress=False)
    if isinstance(data.columns, pd.MultiIndex):
        data.columns = data.columns.get_level_values(0)
    return data

def calculate_indicators(df):
    close = df["Close"].squeeze()
    
    # RSI
    delta = close.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.rolling(14).mean()
    avg_loss = loss.rolling(14).mean()
    rs = avg_gain / avg_loss
    df["RSI"] = 100 - (100 / (1 + rs))
    
    # MACD
    ema12 = close.ewm(span=12, adjust=False).mean()
    ema26 = close.ewm(span=26, adjust=False).mean()
    df["MACD_Line"] = ema12 - ema26
    df["Signal_Line"] = df["MACD_Line"].ewm(span=9, adjust=False).mean()
    df["MACD_Hist"] = df["MACD_Line"] - df["Signal_Line"]
    
    # ATR
    high, low = df["High"].squeeze(), df["Low"].squeeze()
    prev_close = close.shift(1)
    tr = pd.concat([high-low, (high-prev_close).abs(), (low-prev_close).abs()], axis=1).max(axis=1)
    df["ATR"] = tr.rolling(14).mean()
    
    return df

def run_ai_prediction(df):
    ml_df = df.copy().dropna()
    ml_df["Target"] = np.where(ml_df["Close"].shift(-1) > ml_df["Close"], 1, 0)
    features = ["RSI", "MACD_Line", "MACD_Hist", "ATR"]
    
    train_df = ml_df.iloc[:-1]
    X = train_df[features]
    y = train_df["Target"]

    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X, y)

    latest_data = ml_df.iloc[-1:][features]
    prediction = model.predict(latest_data)[0]
    probabilities = model.predict_proba(latest_data)[0]
    
    confidence = probabilities[prediction] * 100
    signal = "BULLISH 📈" if prediction == 1 else "BEARISH 📉"
    
    return signal, round(confidence, 1)

def fetch_yfinance_news(ticker):
    try:
        stock = yf.Ticker(ticker)
        news = stock.news
        if not news: return []
        
        parsed_news = []
        for item in news[:5]:
            title = item.get("title", "")
            try:
                sentiment = TextBlob(title).sentiment.polarity
            except:
                sentiment = 0.0
            label = "🟢 Pos" if sentiment > 0.1 else "🔴 Neg" if sentiment < -0.1 else "⚪ Neu"
            parsed_news.append({"title": title, "label": label, "link": item.get("link", "")})
        return parsed_news
    except:
        return []

# --- USER INTERFACE ---
def render_ui():
    st.set_page_config(page_title="Framd AI Terminal", page_icon="⚡", layout="wide")
    
    # Framd Vibe Styling
    st.markdown("""
    <style>
    .stButton>button { border-color: #facc15; color: #facc15; }
    .stButton>button:hover { background-color: #facc15; color: black; }
    </style>
    """, unsafe_allow_html=True)
    
    # Sidebar
    st.sidebar.markdown("## ⚡ Framd Setup")
    ticker = st.sidebar.text_input("Ticker", value="AAPL").upper().strip()
    risk_pct = st.sidebar.slider("Risk Per Trade %", 0.1, 5.0, 1.0)
    
    client, error_msg = get_alpaca_client()
    
    if client:
        acc = client.get_account()
        buying_power = float(acc.buying_power)
        st.sidebar.success("✅ Alpaca Connected")
    else:
        buying_power = 0.0
        st.sidebar.error(f"⚠️ Alpaca Disconnected: {error_msg}")

    if not ticker: return

    with st.spinner("Framd AI Neural Net is processing..."):
        df = fetch_market_data(ticker)
        if df.empty:
            st.error("No data found.")
            return
            
        df = calculate_indicators(df)
        curr_price = float(df["Close"].iloc[-1])
        curr_atr = float(df["ATR"].iloc[-1])
        signal, confidence = run_ai_prediction(df)

    # Top Metrics
    m1, m2, m3, m4 = st.columns(4)
    m1.metric(f"{ticker} Price", f"${curr_price:,.2f}")
    m2.metric("Framd AI Signal", signal)
    m3.metric("AI Confidence", f"{confidence}%")
    m4.metric("Live Buying Power", f"${buying_power:,.2f}")

    # Main Tabs
    t1, t2, t3 = st.tabs(["📊 Charts", "🧠 AI Brain & News", "🚀 Live Execution"])

    with t1:
        fig = go.Figure(data=[go.Candlestick(x=df.index, open=df['Open'], high=df['High'], low=df['Low'], close=df['Close'])])
        fig.update_layout(xaxis_rangeslider_visible=False, height=450, margin=dict(t=10, b=10))
        st.plotly_chart(fig, use_container_width=True)

    with t2:
        col1, col2 = st.columns([1, 1])
        with col1:
            st.subheader("🧠 Model Analysis")
            st.write(f"Based on 2 years of historical RSI, MACD, and Volatility patterns, the Machine Learning model predicts the next closing price will be **{signal}**.")
            st.progress(confidence / 100.0)
            st.caption(f"Statistical Confidence Level: {confidence}%")
            
        with col2:
            st.subheader("📰 Market Pulse")
            news_items = fetch_yfinance_news(ticker)
            if news_items:
                for n in news_items:
                    st.write(f"{n['label']} | [{n['title']}]({n['link']})")
            else:
                st.info("No recent news found.")

    with t3:
        # RISK CALCULATOR
        dollar_risk = buying_power * (risk_pct / 100.0)
        stop_dist = curr_atr * 1.5
        shares = int(dollar_risk // stop_dist) if stop_dist > 0 else 0
        
        st.subheader("Action Center")
        c1, c2 = st.columns(2)
        c1.write(f"**Target Shares:** {shares}")
        c1.write(f"**Risk Amount:** ${dollar_risk:,.2f}")
        c2.write(f"**Stop Loss:** ${curr_price - stop_dist:,.2f}")
        c2.write(f"**Total Cost:** ${shares * curr_price:,.2f}")

        st.markdown("---")
        if client:
            st.warning("⚠️ Clicking these buttons will route a live paper trade to Alpaca.")
            b1, b2 = st.columns(2)
            if b1.button("🚀 LIVE EXECUTE BUY", use_container_width=True):
                if shares > 0:
                    success, err = execute_alpaca_order(ticker, OrderSide.BUY, shares)
                    if success: st.success(f"Order Sent: Bought {shares} of {ticker}!")
                    else: st.error(f"Failed: {err}")
                else:
                    st.error("Share count is 0. Check your buying power.")
                    
            if b2.button("📉 LIVE EXECUTE SELL", use_container_width=True):
                if shares > 0:
                    success, err = execute_alpaca_order(ticker, OrderSide.SELL, shares)
                    if success: st.success(f"Order Sent: Sold {shares} of {ticker}!")
                    else: st.error(f"Failed: {err}")
                else:
                    st.error("Share count is 0. Check your buying power.")
        else:
            st.error("Execution disabled. Alpaca is not connected. Check the sidebar for errors.")

if __name__ == "__main__":
    render_ui()
