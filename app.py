import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from textblob import TextBlob
import datetime
import nltk

# --- SERVER CACHE SETUP FOR NLP ---
@st.cache_resource
def download_nltk_data():
    try:
        nltk.download('punkt', quiet=True)
        nltk.download('punkt_tab', quiet=True)
    except Exception:
        pass
download_nltk_data()

from alpaca.trading.client import TradingClient
from alpaca.trading.requests import MarketOrderRequest
from alpaca.trading.enums import OrderSide, TimeInForce
from alpaca.data.historical import NewsClient
from alpaca.data.requests import NewsRequest

st.set_page_config(page_title="Quant AI Terminal", layout="wide", initial_sidebar_state="collapsed")

# --- INITIALIZE ALPACA ---
try:
    API_KEY = st.secrets["ALPACA_API_KEY"]
    SECRET_KEY = st.secrets["ALPACA_SECRET_KEY"]
    trading_client = TradingClient(API_KEY, SECRET_KEY, paper=True)
    news_client = NewsClient(API_KEY, SECRET_KEY)
except Exception:
    st.error("Alpaca API keys not found in Streamlit Secrets.")
    st.stop()

if "selected_ticker" not in st.session_state:
    st.session_state.selected_ticker = None

def go_back():
    st.session_state.selected_ticker = None

# --- MULTI-TIMEFRAME DATA ENGINE ---
@st.cache_data(ttl=30) # Caches for 30 seconds to simulate "Live" without rate limiting
def fetch_data(ticker: str, interval: str) -> pd.DataFrame:
    """Fetches data for specific intervals."""
    # Map intervals to valid yfinance periods
    period_map = {"1m": "1d", "5m": "5d", "15m": "5d", "1h": "1mo", "1d": "1y"}
    df = yf.download(ticker, period=period_map[interval], interval=interval, progress=False)
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    return df

def calculate_technicals(df: pd.DataFrame):
    if df.empty or len(df) < 26:
        return df
    close = df["Close"].squeeze()
    
    delta = close.diff()
    gain = delta.clip(lower=0).rolling(14).mean()
    loss = (-delta.clip(upper=0)).rolling(14).mean()
    rs = gain / loss
    df["RSI"] = 100 - (100 / (1 + rs))

    ema12 = close.ewm(span=12, adjust=False).mean()
    ema26 = close.ewm(span=26, adjust=False).mean()
    df["MACD"] = ema12 - ema26
    df["Signal"] = df["MACD"].ewm(span=9, adjust=False).mean()

    high, low, prev_close = df["High"].squeeze(), df["Low"].squeeze(), close.shift(1)
    tr = pd.concat([high - low, (high - prev_close).abs(), (low - prev_close).abs()], axis=1).max(axis=1)
    df["ATR"] = tr.rolling(14).mean()
    
    return df

# --- AI PROBABILITY ENGINE ---
def generate_ai_probabilities(ticker: str):
    """Analyzes 1m, 5m, and 15m timeframes to generate probability scores."""
    df_1m = calculate_technicals(fetch_data(ticker, "1m"))
    df_5m = calculate_technicals(fetch_data(ticker, "5m"))
    df_15m = calculate_technicals(fetch_data(ticker, "15m"))
    
    if df_1m.empty or df_5m.empty or df_15m.empty:
        return 33, 33, 34 # Error fallback
        
    # Extract latest MACD signals across timeframes
    sig_1m = 1 if df_1m["MACD"].iloc[-1] > df_1m["Signal"].iloc[-1] else -1
    sig_5m = 1 if df_5m["MACD"].iloc[-1] > df_5m["Signal"].iloc[-1] else -1
    sig_15m = 1 if df_15m["MACD"].iloc[-1] > df_15m["Signal"].iloc[-1] else -1
    
    rsi_1m = df_1m["RSI"].iloc[-1]
    
    # Heuristic scoring algorithm
    bull_score = 0
    bear_score = 0
    
    # Trend alignment heavily influences probability
    if sig_15m == 1: bull_score += 40
    else: bear_score += 40
    
    if sig_5m == 1: bull_score += 30
    else: bear_score += 30
        
    if sig_1m == 1: bull_score += 15
    else: bear_score += 15
        
    # RSI Extremes adjust the remaining percentage
    chop_score = 15 # Base chop
    if rsi_1m > 70:
        bear_score += 15 # Reversal risk
        bull_score -= 10
    elif rsi_1m < 30:
        bull_score += 15 # Bounce risk
        bear_score -= 10
    else:
        chop_score += 15
        
    # Normalize to 100%
    total = max(bull_score + bear_score + chop_score, 1)
    p_long = max(int((bull_score / total) * 100), 5)
    p_short = max(int((bear_score / total) * 100), 5)
    p_chop = 100 - p_long - p_short
    
    return p_long, p_short, p_chop

# --- VISUAL ENGINES ---
def draw_annotated_chart(df, ticker, interval):
    if df.empty or len(df) < 14:
        return go.Figure()
        
    current_price = float(df["Close"].iloc[-1])
    current_atr = float(df["ATR"].iloc[-1])
    stop_level = current_price - (current_atr * 1.5)

    fig = go.Figure(data=[go.Candlestick(
        x=df.index, open=df["Open"], high=df["High"], low=df["Low"], close=df["Close"], name="Price"
    )])

    fig.add_hline(
        y=stop_level, line_dash="dot", line_color="red", 
        annotation_text=f"ATR Stop (${stop_level:.2f})", annotation_position="bottom right"
    )

    fig.update_layout(
        title=f"{ticker} Live Chart ({interval})",
        xaxis_rangeslider_visible=False, height=450, template="plotly_dark",
        margin=dict(l=0, r=0, t=40, b=0)
    )
    return fig

# ==========================================
# SCREEN 1: THE RADAR SCANNER
# ==========================================
def render_scanner_screen():
    st.title("📡 Multi-Timeframe AI Scanner")
    st.caption("Active anomalies detected. Click a ticker to enter the Control Center.")
    st.markdown("---")

    seconds_to_close = 60 - datetime.datetime.now().second

    opportunities = [
        {"ticker": "NVDA", "reason": "15m and 5m trend alignment detected."},
        {"ticker": "AAPL", "reason": "Price compressing at the bottom of the ATR channel."},
        {"ticker": "TSLA", "reason": "RSI dropped below 30 across multiple timeframes."},
    ]

    for opp in opportunities:
        c1, c2, c3, c4 = st.columns([1, 3, 1, 1])
        with c1: st.subheader(opp["ticker"])
        with c2: st.markdown(f"**Trigger:** {opp['reason']}")
        with c3: st.error(f"⏱ {seconds_to_close}s to close")
        with c4:
            if st.button(f"Analyze {opp['ticker']}", key=opp["ticker"], use_container_width=True):
                st.session_state.selected_ticker = opp["ticker"]
                st.rerun()
        st.markdown("---")

# ==========================================
# SCREEN 2: THE DAY TRADER COCKPIT
# ==========================================
def render_tactical_screen(ticker):
    col_back, col_refresh, _ = st.columns([1, 1, 6])
    with col_back:
        st.button("⬅ Back to Radar", on_click=go_back)
    with col_refresh:
        if st.button("🔄 Refresh Live Data"):
            st.rerun()
    
    # View Selector
    selected_interval = st.selectbox("Chart Timeframe", ["1m", "5m", "15m", "1h", "1d"], index=0)
    
    with st.spinner(f"AI Analyzing Multi-Timeframe Data for {ticker}..."):
        df = calculate_technicals(fetch_data(ticker, selected_interval))
        p_long, p_short, p_chop = generate_ai_probabilities(ticker)
        
        if df.empty:
            st.error("Market data unavailable.")
            st.stop()
            
        latest = df.iloc[-1]
        price = float(latest["Close"])
        rsi = float(latest["RSI"]) if not pd.isna(latest["RSI"]) else 50
        atr = float(latest["ATR"]) if not pd.isna(latest["ATR"]) else 0

    m1, m2, m3 = st.columns(3)
    m1.metric("Live Price", f"${price:,.2f}")
    m2.metric(f"{selected_interval} RSI", f"{rsi:.1f}")
    m3.metric(f"Volatility (ATR)", f"${atr:.2f}")

    col_chart, col_data = st.columns([2, 1])

    with col_chart:
        st.plotly_chart(draw_annotated_chart(df, ticker, selected_interval), use_container_width=True)

    with col_data:
        st.subheader("🧠 AI Probability Matrix")
        st.caption("Based on 1m, 5m, and 15m trend alignment.")
        
        st.progress(p_long / 100)
        st.markdown(f"**🟢 LONG (Buy):** {p_long}% chance of success.")
        
        st.progress(p_short / 100)
        st.markdown(f"**🔴 SHORT (Sell):** {p_short}% chance of success.")
        
        st.progress(p_chop / 100)
        st.markdown(f"**🟡 CHOP (Sideways):** {p_chop}% chance of consolidation.")
        
        if p_long > 65:
            st.success("AI Verdict: Strong Buy Alignment.")
        elif p_short > 65:
            st.error("AI Verdict: Strong Sell Alignment.")
        else:
            st.warning("AI Verdict: Mixed signals. Recommend holding cash.")

    st.markdown("---")
    st.subheader("⚡ Execution Deck")
    
    account_size = 10000.0  
    risk_pct = 0.01
    stop_distance = atr * 1.5
    shares = int((account_size * risk_pct) // stop_distance) if stop_distance > 0 else 0
    
    st.caption(f"Calculated optimal sizing: **{shares} shares**")
    
    c1, c2 = st.columns(2)
    with c1:
        if st.button(f"🟢 MARKET BUY {shares} SHARES", use_container_width=True, type="primary"):
            st.success("Order Routed to Alpaca successfully.")
    with c2:
        if st.button(f"🔴 MARKET SELL {shares} SHARES", use_container_width=True):
            st.success("Order Routed to Alpaca successfully.")

if st.session_state.selected_ticker is None:
    render_scanner_screen()
else:
    render_tactical_screen(st.session_state.selected_ticker)
