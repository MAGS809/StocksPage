import streamlit as st
import yfinance as yf
import pandas as pd
import plotly.graph_objects as go
import praw
from alpaca.trading.client import TradingClient
from alpaca.trading.requests import MarketOrderRequest
from alpaca.trading.enums import OrderSide, TimeInForce

# ── PAGE CONFIG & CSS ──────────────────────────────────────────
st.set_page_config(page_title="Framd. Core", layout="wide")

st.markdown("""
<style>
    /* Framd. Core Minimalist Dark Theme */
    .stApp { background-color: #000000; color: #FFFFFF; }
    .stTextInput>div>div>input { background-color: #121212; color: #FFD700; border: 1px solid #333; }
    .stButton>button { border-radius: 4px; font-weight: bold; }
    .btn-buy>div>button { background-color: #27AE60; color: white; border: none; }
    .btn-sell>div>button { background-color: #E74C3C; color: white; border: none; }
    .ai-insight { background-color: #121212; border-left: 4px solid #FFD700; padding: 15px; border-radius: 4px; margin-bottom: 20px;}
    .metric-value { color: #FFD700; font-size: 1.5rem; font-weight: bold; }
</style>
""", unsafe_allow_html=True)

# ── SESSION STATE ──────────────────────────────────────────────
if 'cash' not in st.session_state:
    st.session_state.cash = 100000.0
if 'portfolio' not in st.session_state:
    st.session_state.portfolio = {}
if 'api_key' not in st.session_state:
    st.session_state.api_key = ""
if 'secret_key' not in st.session_state:
    st.session_state.secret_key = ""

# ── SIDEBAR: AUTHENTICATION & MODE ─────────────────────────────
with st.sidebar:
    st.markdown("<h2 style='color: #FFD700;'>Framd. Core</h2>", unsafe_allow_html=True)
    mode = st.radio("Operating Mode", ["Virtual Simulation", "Live Brokerage"])
    
    with st.expander("Brokerage Connection"):
        with st.form("alpaca_auth"):
            st.caption("Secure Alpaca API Port")
            api_input = st.text_input("API Key", type="password", value=st.session_state.api_key)
            sec_input = st.text_input("Secret Key", type="password", value=st.session_state.secret_key)
            if st.form_submit_button("Connect Broker"):
                st.session_state.api_key = api_input
                st.session_state.secret_key = sec_input
                st.success("Keys Locked In.")

    if mode == "Live Brokerage":
        st.error("⚠️ LIVE FUNDS ACTIVE")

# ── DATA FETCHING & REDDIT SCRAPING ────────────────────────────
@st.cache_data(ttl=60)
def get_stock_data(ticker):
    try:
        # yfinance strictly for drawing the OHLC pixels
        df = yf.download(ticker, period="1mo", interval="15m")
        return df
    except Exception:
        return pd.DataFrame()

@st.cache_data(ttl=60)
def get_reddit_chatter(ticker):
    """Pulls the 3 newest headlines from retail trading subreddits."""
    # Failsafe in case secrets aren't loaded yet
    client_id = st.secrets.get("REDDIT_CLIENT_ID", "")
    secret = st.secrets.get("REDDIT_SECRET", "")
    
    if not client_id or not secret:
        return ["⚠️ Reddit API keys missing from Streamlit secrets."]
        
    try:
        reddit = praw.Reddit(
            client_id=client_id,
            client_secret=secret,
            user_agent='Framd_Core_Terminal'
        )
        subreddit = reddit.subreddit('wallstreetbets+stocks+investing')
        return [submission.title for submission in subreddit.search(ticker, sort='new', limit=3)]
    except Exception as e:
        return [f"Reddit API Error: {e}"]

# ── MAIN UI: CHART & ANALYSIS ──────────────────────────────────
ticker = st.text_input("TICKER SYMBOL", value="NVDA").upper()
df = get_stock_data(ticker)

if not df.empty:
    current_price = float(df['Close'].iloc[-1])
    
    # Minimalist Candlestick Chart
    fig = go.Figure(data=[go.Candlestick(
        x=df.index, open=df['Open'], high=df['High'], low=df['Low'], close=df['Close'],
        increasing_line_color='#FFD700', decreasing_line_color='#555555'
    )])
    fig.update_layout(
        margin=dict(l=0, r=0, t=10, b=0),
        height=400,
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        xaxis_rangeslider_visible=False,
        xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
        yaxis=dict(showgrid=True, gridcolor='#222', zeroline=False)
    )
    # Using width='stretch' to clear the Streamlit deprecation warnings
    st.plotly_chart(fig, width='stretch', config={'displayModeBar': False})

    # ── LIVE REDDIT AI INSIGHT BLOCK ───────────────────────────
    live_headlines = get_reddit_chatter(ticker)
    chatter_html = "".join([f"<li>{title}</li>" for title in live_headlines])

    st.markdown(f"""
    <div class="ai-insight">
        <h4 style="margin-top:0; color:#FFD700;">📡 Live Retail Sentiment: {ticker}</h4>
        <p style="margin-bottom: 5px;"><strong>Latest Chatter:</strong></p>
        <ul style="margin-top: 0; padding-left: 20px;">
            {chatter_html}
        </ul>
        <hr style="border-color: #333;">
        <strong>Suggested Risk Parameters:</strong><br>
        Stop Loss: ${current_price * 0.95:.2f} &nbsp;|&nbsp; Take Profit: ${current_price * 1.10:.2f}
    </div>
    """, unsafe_allow_html=True)

    # ── EXECUTION PANEL ────────────────────────────────────────
    st.markdown("### ⚡ Execution Terminal")
    qty = st.number_input("Shares", min_value=1, value=10)
    
    c1, c2, c3, c4 = st.columns(4)
    if mode == "Virtual Simulation":
        c1.markdown(f"**Cash:** <span class='metric-value'>${st.session_state.cash:,.2f}</span>", unsafe_allow_html=True)
        owned = st.session_state.portfolio.get(ticker, 0)
        c2.markdown(f"**Holding:** <span class='metric-value'>{owned} Shares</span>", unsafe_allow_html=True)
    else:
        c1.markdown("**Cash:** <span class='metric-value'>LIVE DATA</span>", unsafe_allow_html=True)
        c2.markdown(f"**Holding:** <span class='metric-value'>LIVE DATA</span>", unsafe_allow_html=True)

    c_buy, c_sell = st.columns(2)
    with c_buy:
        st.markdown('<div class="btn-buy">', unsafe_allow_html=True)
        if st.button("🟢 MARKET BUY", use_container_width=True):
            if mode == "Virtual Simulation":
                cost = current_price * qty
                if st.session_state.cash >= cost:
                    st.session_state.cash -= cost
                    st.session_state.portfolio[ticker] = st.session_state.portfolio.get(ticker, 0) + qty
                    st.success(f"Virtually bought {qty} {ticker} at ${current_price:.2f}")
                else:
                    st.error("Insufficient virtual funds.")
            else:
                if st.session_state.api_key and st.session_state.secret_key:
                    try:
                        client = TradingClient(st.session_state.api_key, st.session_state.secret_key, paper=True)
                        order = MarketOrderRequest(symbol=ticker, qty=qty, side=OrderSide.BUY, time_in_force=TimeInForce.GTC)
                        client.submit_order(order_data=order)
                        st.success(f"LIVE ORDER ROUTED: Bought {qty} {ticker}")
                    except Exception as e:
                        st.error(f"Alpaca API Error: {e}")
                else:
                    st.warning("Please connect your broker in the sidebar first.")
        st.markdown('</div>', unsafe_allow_html=True)

    with c_sell:
        st.markdown('<div class="btn-sell">', unsafe_allow_html=True)
        if st.button("🔴 MARKET SELL", use_container_width=True):
            if mode == "Virtual Simulation":
                if st.session_state.portfolio.get(ticker, 0) >= qty:
                    revenue = current_price * qty
                    st.session_state.cash += revenue
                    st.session_state.portfolio[ticker] -= qty
                    st.success(f"Virtually sold {qty} {ticker} at ${current_price:.2f}")
                else:
                    st.error("Not enough virtual shares to sell.")
            else:
                if st.session_state.api_key and st.session_state.secret_key:
                    try:
                        client = TradingClient(st.session_state.api_key, st.session_state.secret_key, paper=True)
                        order = MarketOrderRequest(symbol=ticker, qty=qty, side=OrderSide.SELL, time_in_force=TimeInForce.GTC)
                        client.submit_order(order_data=order)
                        st.success(f"LIVE ORDER ROUTED: Sold {qty} {ticker}")
                    except Exception as e:
                        st.error(f"Alpaca API Error: {e}")
                else:
                    st.warning("Please connect your broker in the sidebar first.")
        st.markdown('</div>', unsafe_allow_html=True)

else:
    st.warning("Awaiting valid ticker data...")
