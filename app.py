import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import nltk
from textblob import TextBlob
from alpaca.data.requests import NewsRequest
from alpaca.data.historical import StockHistoricalDataClient
from datetime import datetime, timedelta

# ---------------------------------------------------------------------------
# NLTK SETUP
# ---------------------------------------------------------------------------
@st.cache_resource
def download_nltk_data():
    nltk.download("punkt", quiet=True)
    nltk.download("punkt_tab", quiet=True)

download_nltk_data()

# ---------------------------------------------------------------------------
# 1. DATA FETCHING
# ---------------------------------------------------------------------------
def fetch_market_data(ticker, period="6mo"):
    data = yf.download(ticker, period=period, progress=False)
    if data.empty:
        st.error(f"No data found for '{ticker}'.")
    if isinstance(data.columns, pd.MultiIndex):
        data.columns = data.columns.get_level_values(0)
    return data

# ---------------------------------------------------------------------------
# 2. INDICATOR CALCULATIONS
# ---------------------------------------------------------------------------
def calculate_indicators(df):
    close = df["Close"].squeeze()
    delta = close.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.rolling(14, min_periods=14).mean()
    avg_loss = loss.rolling(14, min_periods=14).mean()
    rs = avg_gain / avg_loss
    df["RSI"] = 100 - (100 / (1 + rs))

    ema12 = close.ewm(span=12, adjust=False).mean()
    ema26 = close.ewm(span=26, adjust=False).mean()
    df["MACD_Line"] = ema12 - ema26
    df["Signal_Line"] = df["MACD_Line"].ewm(span=9, adjust=False).mean()
    df["MACD_Hist"] = df["MACD_Line"] - df["Signal_Line"]

    latest = df.dropna(subset=["RSI", "MACD_Line"]).iloc[-1]
    return {
        "rsi": round(float(latest["RSI"]), 2),
        "macd_line": round(float(latest["MACD_Line"]), 2),
        "signal_line": round(float(latest["Signal_Line"]), 2),
        "macd_hist": round(float(latest["MACD_Hist"]), 2),
        "df": df,
    }

# ---------------------------------------------------------------------------
# 3. ATR CALCULATION
# ---------------------------------------------------------------------------
def calculate_atr(df, period=14):
    high = df["High"].squeeze()
    low = df["Low"].squeeze()
    close = df["Close"].squeeze()
    prev_close = close.shift(1)
    tr1 = high - low
    tr2 = (high - prev_close).abs()
    tr3 = (low - prev_close).abs()
    df["True_Range"] = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    df["ATR"] = df["True_Range"].rolling(period, min_periods=period).mean()
    return df
# ---------------------------------------------------------------------------
# 4. NEWS SENTIMENT VIA ALPACA
# ---------------------------------------------------------------------------
def fetch_and_analyze_news(ticker):
    api_key = st.secrets.get("ALPACA_API_KEY", "")
    secret_key = st.secrets.get("ALPACA_SECRET_KEY", "")
    if not api_key or not secret_key:
        return pd.DataFrame()
    try:
        client = StockHistoricalDataClient(api_key, secret_key)
        request = NewsRequest(symbols=ticker, start=datetime.now() - timedelta(days=7), limit=5)
        news_items = client.get_news(request).news
    except Exception:
        return pd.DataFrame()
    if not news_items:
        return pd.DataFrame()
    rows = []
    for item in news_items[:5]:
        title = item.headline or ""
        source = item.source or "Unknown"
        url = item.url or ""
        blob = TextBlob(title)
        polarity = round(blob.sentiment.polarity, 3)
        if polarity > 0.05:
            label = "Positive"
        elif polarity < -0.05:
            label = "Negative"
        else:
            label = "Neutral"
        rows.append({"Headline": title, "Source": source, "Polarity": polarity, "Sentiment": label, "Link": url})
    return pd.DataFrame(rows)

# ---------------------------------------------------------------------------
# 5. EDUCATIONAL EXPLAINER
# ---------------------------------------------------------------------------
def generate_educational_explainer(indicators):
    rsi = indicators["rsi"]
    macd_line = indicators["macd_line"]
    signal_line = indicators["signal_line"]
    macd_hist = indicators["macd_hist"]
    sections = []

    sections.append(f"**RSI is at {rsi}.**")
    if rsi >= 70:
        sections.append(
            "This is above the traditional 70 threshold, described in textbooks as "
            "**overbought** territory. This means sustained upward momentum relative "
            "to recent history. It does NOT mean the price will reverse."
        )
    elif rsi <= 30:
        sections.append(
            "This is below the traditional 30 threshold, described in textbooks as "
            "**oversold** territory. This means sustained downward momentum relative "
            "to recent history. It does NOT mean the price will reverse."
        )
    else:
        sections.append(
            "This falls in **neutral territory** (between 30 and 70). Neither "
            "overbought nor oversold conditions are indicated."
        )

    sections.append(f"\n**MACD Line is at {macd_line}, Signal Line at {signal_line}.**")
    if macd_line > signal_line:
        sections.append(
            "MACD is **above** the Signal Line, which textbooks describe as a "
            "bullish crossover - short-term momentum exceeds longer-term momentum."
        )
    elif macd_line < signal_line:
        sections.append(
            "MACD is **below** the Signal Line, which textbooks describe as a "
            "bearish crossover - short-term momentum trails longer-term momentum."
        )
    else:
        sections.append("MACD and Signal Line are equal - a crossover point.")

    if macd_line > 0:
        sections.append("MACD is positive (above zero line) - upward trend momentum.")
    elif macd_line < 0:
        sections.append("MACD is negative (below zero line) - downward trend momentum.")

    return "\n\n".join(sections)
# ---------------------------------------------------------------------------
# 6. POSITION SIZING CALCULATOR
# ---------------------------------------------------------------------------
def calculate_position_size(current_price, current_atr, account_balance, max_risk_pct, atr_multiplier):
    dollar_risk = account_balance * (max_risk_pct / 100.0)
    stop_distance = current_atr * atr_multiplier
    stop_price = current_price - stop_distance
    if stop_distance > 0:
        shares = int(dollar_risk // stop_distance)
    else:
        shares = 0
    return {
        "dollar_risk": dollar_risk,
        "stop_distance": stop_distance,
        "stop_price": stop_price,
        "shares": shares,
        "total_cost": shares * current_price,
    }

# ---------------------------------------------------------------------------
# 7. STREAMLIT UI
# ---------------------------------------------------------------------------
def render_ui():
    st.set_page_config(page_title="StocksPage", page_icon="📈", layout="wide")

    # -- Custom CSS --
    st.markdown("""
    <style>
    .block-container { padding-top: 1rem; padding-bottom: 1rem; }
    [data-testid="stMetricValue"] { font-size: 1.8rem; }
    [data-testid="stMetricLabel"] { font-size: 0.85rem; }
    .stTabs [data-baseweb="tab-list"] { gap: 8px; }
    .stTabs [data-baseweb="tab"] { padding: 8px 20px; font-weight: 600; }
    </style>
    """, unsafe_allow_html=True)

    # -- Sidebar --
    with st.sidebar:
        st.markdown("## 📈 StocksPage")
        st.caption("Educational market data dashboard")
        st.markdown("---")
        ticker = st.text_input("Ticker Symbol", value="AAPL").upper().strip()
        period = st.selectbox("Look-back Period", ["1mo", "3mo", "6mo", "1y", "2y"], index=2)
        st.markdown("---")
        st.markdown("*This dashboard is strictly educational. Nothing here is financial advice.*")

    if not ticker:
        st.info("Enter a ticker symbol in the sidebar to get started.")
        return

    # -- Fetch & compute --
    with st.spinner(f"Loading {ticker}..."):
        df = fetch_market_data(ticker, period=period)
        if df.empty:
            return
        indicators = calculate_indicators(df)
        enriched_df = indicators["df"]
        enriched_df = calculate_atr(enriched_df)
        current_price = float(enriched_df["Close"].squeeze().iloc[-1])
        prev_close = float(enriched_df["Close"].squeeze().iloc[-2])
        price_change = current_price - prev_close
        price_pct = (price_change / prev_close) * 100
        atr_val = float(enriched_df["ATR"].dropna().iloc[-1])

    # -- Header bar --
    h1, h2, h3, h4, h5 = st.columns([2, 1, 1, 1, 1])
    with h1:
        st.markdown(f"# {ticker}")
    with h2:
        st.metric("Price", f"${current_price:,.2f}", f"{price_change:+,.2f} ({price_pct:+.1f}%)")
    with h3:
        st.metric("RSI (14)", indicators["rsi"])
    with h4:
        st.metric("MACD", indicators["macd_line"])
    with h5:
        st.metric("ATR (14)", f"${atr_val:,.2f}")

    st.markdown("---")

    # -- Tabs --
    tab_chart, tab_learn, tab_news, tab_risk, tab_journal = st.tabs(
        ["📊 Charts", "📖 Explainer", "📰 News", "🧮 Risk Calculator", "📝 Journal"]
    )
    # -- TAB 1: Charts --
    with tab_chart:
        price_fig = go.Figure(data=[go.Candlestick(
            x=enriched_df.index,
            open=enriched_df["Open"].squeeze(),
            high=enriched_df["High"].squeeze(),
            low=enriched_df["Low"].squeeze(),
            close=enriched_df["Close"].squeeze(),
            name="Price",
        )])
        price_fig.update_layout(
            xaxis_rangeslider_visible=False,
            height=400,
            margin=dict(l=0, r=0, t=10, b=0),
        )
        st.plotly_chart(price_fig, use_container_width=True)

        c1, c2 = st.columns(2)
        with c1:
            rsi_fig = go.Figure()
            rsi_fig.add_trace(go.Scatter(x=enriched_df.index, y=enriched_df["RSI"], name="RSI", line=dict(color="#6C9EFF")))
            rsi_fig.add_hline(y=70, line_dash="dash", line_color="#ef4444", annotation_text="70")
            rsi_fig.add_hline(y=30, line_dash="dash", line_color="#22c55e", annotation_text="30")
            rsi_fig.update_layout(title="RSI (14)", height=250, margin=dict(l=0, r=0, t=30, b=0))
            st.plotly_chart(rsi_fig, use_container_width=True)
        with c2:
            macd_fig = go.Figure()
            macd_fig.add_trace(go.Scatter(x=enriched_df.index, y=enriched_df["MACD_Line"], name="MACD", line=dict(color="#6C9EFF")))
            macd_fig.add_trace(go.Scatter(x=enriched_df.index, y=enriched_df["Signal_Line"], name="Signal", line=dict(color="#f59e0b")))
            macd_fig.add_trace(go.Bar(x=enriched_df.index, y=enriched_df["MACD_Hist"], name="Histogram", opacity=0.3, marker_color="#8b5cf6"))
            macd_fig.update_layout(title="MACD (12, 26, 9)", height=250, margin=dict(l=0, r=0, t=30, b=0))
            st.plotly_chart(macd_fig, use_container_width=True)

    # -- TAB 2: Educational Explainer --
    with tab_learn:
        st.markdown("### What do the current indicators mean?")
        st.caption("Objective textbook explanations only. No predictions or trade suggestions.")
        st.markdown("---")
        st.markdown(generate_educational_explainer(indicators))
        st.markdown("---")
        st.caption(
            "This is strictly educational. Nothing here constitutes financial advice, "
            "a prediction, or a recommendation to buy, sell, or hold any security."
        )

    # -- TAB 3: News --
    with tab_news:
        st.markdown("### Recent Headlines")
        st.caption("Sentiment is a basic NLP tone measure, not a trade signal.")
        news_df = fetch_and_analyze_news(ticker)
        if news_df.empty:
            st.info("No news available. Add Alpaca API keys to enable this feature.")
        else:
            for _, row in news_df.iterrows():
                badge = {"Positive": "🟢", "Negative": "🔴", "Neutral": "⚪"}.get(row["Sentiment"], "⚪")
                with st.container():
                    nc1, nc2 = st.columns([5, 1])
                    with nc1:
                        st.markdown(f"**{row['Headline']}**")
                        st.caption(f"{row['Source']} \u2022 Polarity: {row['Polarity']}")
                    with nc2:
                        st.markdown(f"<div style='text-align:center;font-size:1.5rem;padding-top:8px'>{badge} {row['Sentiment']}</div>", unsafe_allow_html=True)
                    st.markdown("---")
    # -- TAB 4: Risk Calculator --
    with tab_risk:
        st.markdown("### Position Sizing & Risk Calculator")
        st.caption("Textbook math for volatility-based position sizing. Not financial advice.")
        current_atr = round(atr_val, 2)

        rc1, rc2, rc3 = st.columns(3)
        with rc1:
            account_balance = st.number_input(
                "Account Balance ($)", min_value=0.0, value=10000.0, step=500.0, format="%.2f"
            )
        with rc2:
            max_risk_pct = st.number_input(
                "Max Risk Per Trade (%)", min_value=0.1, max_value=100.0, value=1.0, step=0.25, format="%.2f"
            )
        with rc3:
            atr_multiplier = st.number_input(
                "ATR Multiplier", min_value=0.5, max_value=5.0, value=1.5, step=0.25, format="%.2f"
            )

        calc = calculate_position_size(current_price, current_atr, account_balance, max_risk_pct, atr_multiplier)

        st.markdown("---")
        st.markdown("#### Results")
        r1, r2, r3, r4 = st.columns(4)
        r1.metric("Shares to Buy", f"{calc['shares']:,}")
        r2.metric("Stop-Loss Price", f"${calc['stop_price']:,.2f}")
        r3.metric("Dollar at Risk", f"${calc['dollar_risk']:,.2f}")
        r4.metric("Total Cost", f"${calc['total_cost']:,.2f}")

        with st.expander("Show calculation steps"):
            st.markdown(
                f"1. **Dollar risk:** ${account_balance:,.2f} x {max_risk_pct}% = **${calc['dollar_risk']:,.2f}**\n"
                f"2. **Stop distance:** ${current_atr:,.2f} x {atr_multiplier} = **${calc['stop_distance']:,.2f}**\n"
                f"3. **Stop price:** ${current_price:,.2f} - ${calc['stop_distance']:,.2f} = **${calc['stop_price']:,.2f}**\n"
                f"4. **Shares:** ${calc['dollar_risk']:,.2f} / ${calc['stop_distance']:,.2f} = **{calc['shares']} shares**"
            )

        st.caption(
            "This calculator shows textbook math only. It does NOT account for "
            "slippage, gaps, commissions, or real-world execution."
        )

    # -- TAB 5: Paper Trade Journal --
    with tab_journal:
        st.markdown("### Manual Paper Trade Journal")
        st.caption(
            "Log your own decisions while you study. "
            "These do NOT execute any real or simulated orders."
        )
        st.markdown("---")

        j1, j2 = st.columns(2)
        with j1:
            if st.button("📗 Manual Paper Buy", type="primary", use_container_width=True):
                st.success(f"Paper BUY logged for {ticker} at ${current_price:,.2f}")
        with j2:
            if st.button("📕 Manual Paper Sell", type="secondary", use_container_width=True):
                st.success(f"Paper SELL logged for {ticker} at ${current_price:,.2f}")

# ---------------------------------------------------------------------------
# ENTRY POINT
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    render_ui()
