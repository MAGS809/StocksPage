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
# NLTK SETUP (needed for TextBlob tokenization)
# ---------------------------------------------------------------------------

@st.cache_resource
def download_nltk_data():
    """Download required NLTK data once and cache it."""
    nltk.download("punkt", quiet=True)
    nltk.download("punkt_tab", quiet=True)

download_nltk_data()


# ---------------------------------------------------------------------------
# 1. DATA FETCHING
# ---------------------------------------------------------------------------

def fetch_market_data(ticker: str, period: str = "6mo") -> pd.DataFrame:
    """
    Download historical OHLCV data from Yahoo Finance.

    Parameters
    ----------
    ticker : str
        Stock ticker symbol (e.g. "AAPL").
    period : str
        Look-back window understood by yfinance (default "6mo").

    Returns
    -------
    pd.DataFrame with columns: Open, High, Low, Close, Volume
    """
    data = yf.download(ticker, period=period, progress=False)
    if data.empty:
        st.error(f"No data found for ticker '{ticker}'. Please check the symbol.")
    if isinstance(data.columns, pd.MultiIndex):
        data.columns = data.columns.get_level_values(0)
    return data


# ---------------------------------------------------------------------------
# 2. INDICATOR CALCULATIONS
# ---------------------------------------------------------------------------

def calculate_indicators(df: pd.DataFrame) -> dict:
    """
    Compute RSI (14-period) and MACD (12, 26, 9) from a Close price series.

    Returns
    -------
    dict with keys: rsi, macd_line, signal_line, macd_hist, df
    """
    close = df["Close"].squeeze()

    # --- RSI (14) ---
    delta = close.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.rolling(window=14, min_periods=14).mean()
    avg_loss = loss.rolling(window=14, min_periods=14).mean()
    rs = avg_gain / avg_loss
    df["RSI"] = 100 - (100 / (1 + rs))

    # --- MACD (12, 26, 9) ---
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

def calculate_atr(df: pd.DataFrame, period: int = 14) -> pd.DataFrame:
    """
    Compute the Average True Range (ATR) over a given period.

    Returns the original df with an added 'ATR' column.
    """
    high = df["High"].squeeze()
    low = df["Low"].squeeze()
    close = df["Close"].squeeze()
    prev_close = close.shift(1)

    tr1 = high - low
    tr2 = (high - prev_close).abs()
    tr3 = (low - prev_close).abs()

    df["True_Range"] = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    df["ATR"] = df["True_Range"].rolling(window=period, min_periods=period).mean()

    return df


# ---------------------------------------------------------------------------
# 4. NEWS SENTIMENT VIA ALPACA  (read-only context, no trade signals)
# ---------------------------------------------------------------------------

def fetch_and_analyze_news(ticker: str) -> pd.DataFrame:
    """
    Fetch the 5 most recent news headlines for the given ticker using
    the Alpaca News API and run TextBlob sentiment analysis on each.

    Requires ALPACA_API_KEY and ALPACA_SECRET_KEY in Streamlit secrets
    or environment variables.

    This function provides CONTEXT ONLY.
    It does NOT generate trade signals, scores, or recommendations.

    Returns
    -------
    pd.DataFrame with columns: Headline, Source, Polarity, Sentiment
    """
    api_key = st.secrets.get("ALPACA_API_KEY", "")
    secret_key = st.secrets.get("ALPACA_SECRET_KEY", "")

    if not api_key or not secret_key:
        st.warning(
            "Alpaca API keys not found. Add ALPACA_API_KEY and "
            "ALPACA_SECRET_KEY to your Streamlit secrets to enable news."
        )
        return pd.DataFrame()

    try:
        client = StockHistoricalDataClient(api_key, secret_key)
        request = NewsRequest(
            symbols=ticker,
            start=datetime.now() - timedelta(days=7),
            limit=5,
        )
        news_items = client.get_news(request).news
    except Exception as e:
        st.warning(f"Could not fetch news: {e}")
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

        rows.append({
            "Headline": title,
            "Source": source,
            "Polarity": polarity,
            "Sentiment": label,
            "Link": url,
        })

    return pd.DataFrame(rows)


def render_news_ui(ticker: str):
    """
    Render the News Context section in the Streamlit UI.
    Purely informational - no trade signals.
    """
    st.subheader("News Context")
    st.caption(
        "Recent headlines and their basic NLP sentiment tone. "
        "This is for reading context only - no trade signals, "
        "predictions, or recommendations are generated from this data."
    )

    with st.spinner("Fetching recent news..."):
        news_df = fetch_and_analyze_news(ticker)

    if news_df.empty:
        st.info(f"No recent news found for {ticker}.")
        return

    def color_sentiment(val):
        if val == "Positive":
            return "color: #22c55e"
        elif val == "Negative":
            return "color: #ef4444"
        else:
            return "color: #a3a3a3"

    display_df = news_df[["Headline", "Source", "Polarity", "Sentiment"]]
    styled = display_df.style.applymap(color_sentiment, subset=["Sentiment"])
    st.dataframe(styled, use_container_width=True, hide_index=True)

    with st.expander("Article links"):
        for _, row in news_df.iterrows():
            if row["Link"]:
                st.markdown(f"- [{row['Headline']}]({row['Link']})")

    st.caption(
        "Sentiment is computed using TextBlob's basic NLP polarity score "
        "(-1.0 to +1.0). This is a simple lexicon-based measure of tone "
        "in the headline text and does NOT reflect the full article content, "
        "market impact, or any actionable insight."
    )


# ---------------------------------------------------------------------------
# 5. EDUCATIONAL EXPLAINER  (strictly informational)
# ---------------------------------------------------------------------------

def generate_educational_explainer(indicators: dict) -> str:
    """
    Return an objective, textbook-style explanation of what the current
    RSI and MACD values traditionally indicate in technical analysis.

    EDUCATION ONLY - no ratings, predictions, or trade suggestions.
    """
    rsi = indicators["rsi"]
    macd_line = indicators["macd_line"]
    signal_line = indicators["signal_line"]
    macd_hist = indicators["macd_hist"]

    sections = []

    # ---- RSI ----
    sections.append("### RSI (Relative Strength Index)")
    sections.append(f"**Current value:** {rsi}")
    sections.append(
        "The RSI is a momentum oscillator that measures the speed and magnitude "
        "of recent price changes on a scale of 0 to 100. It was introduced by "
        "J. Welles Wilder Jr. in 1978."
    )

    if rsi >= 70:
        sections.append(
            f"An RSI of {rsi} is above the traditional 70 threshold. "
            "In textbook technical analysis, readings above 70 are described as "
            "**overbought**, meaning the asset has experienced sustained upward "
            "price momentum relative to its recent history. This does NOT imply "
            "the price will reverse; overbought conditions can persist for "
            "extended periods in strong trends."
        )
    elif rsi <= 30:
        sections.append(
            f"An RSI of {rsi} is below the traditional 30 threshold. "
            "In textbook technical analysis, readings below 30 are described as "
            "**oversold**, meaning the asset has experienced sustained downward "
            "price momentum relative to its recent history. This does NOT imply "
            "the price will reverse; oversold conditions can persist for "
            "extended periods in strong downtrends."
        )
    else:
        sections.append(
            f"An RSI of {rsi} falls between the conventional 30 and 70 "
            "boundaries. Technical analysis textbooks describe this range as "
            "**neutral territory**, where neither overbought nor oversold "
            "conditions are indicated by this single metric."
        )

    # ---- MACD ----
    sections.append("---")
    sections.append("### MACD (Moving Average Convergence Divergence)")
    sections.append(
        f"**MACD Line:** {macd_line}  |  "
        f"**Signal Line:** {signal_line}  |  "
        f"**Histogram:** {macd_hist}"
    )
    sections.append(
        "The MACD is a trend-following momentum indicator created by Gerald "
        "Appel. It shows the relationship between two exponential moving "
        "averages (EMA-12 and EMA-26) of the closing price. The Signal Line "
        "is a 9-period EMA of the MACD Line."
    )

    if macd_line > signal_line:
        sections.append(
            "The MACD Line is currently **above** the Signal Line. "
            "In traditional technical analysis, this crossover condition is "
            "described as a **bullish signal**, indicating that short-term "
            "momentum is exceeding longer-term momentum. This is a description "
            "of the current mathematical relationship, not a forecast."
        )
    elif macd_line < signal_line:
        sections.append(
            "The MACD Line is currently **below** the Signal Line. "
            "In traditional technical analysis, this crossover condition is "
            "described as a **bearish signal**, indicating that short-term "
            "momentum is trailing longer-term momentum. This is a description "
            "of the current mathematical relationship, not a forecast."
        )
    else:
        sections.append(
            "The MACD Line and Signal Line are currently at the same value, "
            "indicating a **crossover point**. Technical analysis textbooks "
            "note that crossovers may precede a change in momentum direction, "
            "though this is not guaranteed."
        )

    if macd_line > 0:
        sections.append(
            "The MACD Line is positive (above the zero line), which "
            "traditionally indicates that the 12-period EMA is above the "
            "26-period EMA - a condition textbooks associate with upward "
            "price trend momentum."
        )
    elif macd_line < 0:
        sections.append(
            "The MACD Line is negative (below the zero line), which "
            "traditionally indicates that the 12-period EMA is below the "
            "26-period EMA - a condition textbooks associate with downward "
            "price trend momentum."
        )

    sections.append("---")
    sections.append(
        "*This explainer is strictly educational. It describes what these "
        "indicator values traditionally represent in technical analysis "
        "textbooks. Nothing here constitutes financial advice, a prediction, "
        "or a recommendation to buy, sell, or hold any security.*"
    )

    return "\n\n".join(sections)


# ---------------------------------------------------------------------------
# 6. POSITION SIZING & RISK CALCULATOR  (educational math only)
# ---------------------------------------------------------------------------

def render_risk_calculator(df: pd.DataFrame, current_price: float) -> int:
    """
    Render an educational Position Sizing & Risk Calculator in the
    Streamlit UI and return the calculated position size.

    This is a math worksheet - it does NOT place, simulate, or
    recommend any trades.

    Returns
    -------
    int - the calculated number of shares (position size).
    """
    st.subheader("Position Sizing & Risk Calculator")
    st.caption(
        "An educational tool that shows the textbook math behind "
        "volatility-based position sizing. This does NOT execute, "
        "simulate, or recommend any trade."
    )

    atr_series = df["ATR"].dropna()
    if atr_series.empty:
        st.warning("Not enough data to calculate ATR. Try a longer look-back period.")
        return 0

    current_atr = round(float(atr_series.iloc[-1]), 2)

    info_col1, info_col2 = st.columns(2)
    info_col1.metric("Current Price", f"${current_price:,.2f}")
    info_col2.metric("ATR (14-day)", f"${current_atr:,.2f}")

    st.markdown("---")

    input_col1, input_col2, input_col3 = st.columns(3)
    with input_col1:
        account_balance = st.number_input(
            "Total Account Balance ($)",
            min_value=0.0,
            value=10000.0,
            step=500.0,
            format="%.2f",
        )
    with input_col2:
        max_risk_pct = st.number_input(
            "Max Risk Per Trade (%)",
            min_value=0.1,
            max_value=100.0,
            value=1.0,
            step=0.25,
            format="%.2f",
        )
    with input_col3:
        atr_multiplier = st.number_input(
            "ATR Multiplier for Stop-Loss",
            min_value=0.5,
            max_value=5.0,
            value=1.5,
            step=0.25,
            format="%.2f",
        )

    dollar_risk = account_balance * (max_risk_pct / 100.0)
    stop_loss_distance = current_atr * atr_multiplier
    stop_loss_price = current_price - stop_loss_distance

    if stop_loss_distance > 0:
        position_size = int(dollar_risk // stop_loss_distance)
    else:
        position_size = 0

    total_cost = position_size * current_price

    st.markdown("---")
    st.markdown("### Calculation Breakdown")

    st.markdown(
        f"**Step 1 - Dollar risk per trade:**  \n"
        f"`Account Balance x Max Risk %`  \n"
        f"`${account_balance:,.2f} x {max_risk_pct}% = ${dollar_risk:,.2f}`"
    )
    st.markdown(
        f"**Step 2 - Volatility-based stop-loss distance:**  \n"
        f"`ATR x Multiplier`  \n"
        f"`${current_atr:,.2f} x {atr_multiplier} = ${stop_loss_distance:,.2f}`"
    )
    st.markdown(
        f"**Step 3 - Stop-loss price level:**  \n"
        f"`Current Price - Stop Distance`  \n"
        f"`${current_price:,.2f} - ${stop_loss_distance:,.2f} = ${stop_loss_price:,.2f}`"
    )
    st.markdown(
        f"**Step 4 - Position size (shares):**  \n"
        f"`Dollar Risk / Stop Distance`  \n"
        f"`${dollar_risk:,.2f} / ${stop_loss_distance:,.2f} = "
        f"{position_size} shares`"
    )

    st.markdown("---")
    st.markdown("### Summary")
    sum_col1, sum_col2, sum_col3 = st.columns(3)
    sum_col1.metric("Shares", f"{position_size:,}")
    sum_col2.metric("Stop-Loss Price", f"${stop_loss_price:,.2f}")
    sum_col3.metric("Est. Total Cost", f"${total_cost:,.2f}")

    st.info(
        f"If you bought **{position_size} shares** at **${current_price:,.2f}** "
        f"and the price dropped to the stop-loss at **${stop_loss_price:,.2f}**, "
        f"the loss would be approximately **${dollar_risk:,.2f}** "
        f"({max_risk_pct}% of your ${account_balance:,.2f} account)."
    )

    st.caption(
        "This calculator shows textbook position-sizing math only. "
        "It is NOT financial advice and does NOT account for slippage, "
        "gaps, commissions, or real-world execution. Always do your own "
        "research and consult a financial professional."
    )

    return position_size


# ---------------------------------------------------------------------------
# 7. STREAMLIT UI
# ---------------------------------------------------------------------------

def render_ui():
    """Render the full Streamlit interface."""

    st.set_page_config(page_title="StocksPage", layout="wide")
    st.title("StocksPage - Market Data & Educational Explainer")
    st.caption(
        "A read-only dashboard for viewing market data and learning what "
        "common technical indicators represent. No predictions. No advice."
    )

    # ---- Sidebar ----
    ticker = st.sidebar.text_input("Ticker Symbol", value="AAPL").upper().strip()
    period = st.sidebar.selectbox(
        "Look-back Period",
        options=["1mo", "3mo", "6mo", "1y", "2y"],
        index=2,
    )

    if not ticker:
        st.info("Enter a ticker symbol in the sidebar to get started.")
        return

    # ---- Fetch & compute ----
    with st.spinner(f"Fetching data for {ticker}..."):
        df = fetch_market_data(ticker, period=period)

    if df.empty:
        return

    indicators = calculate_indicators(df)
    enriched_df = indicators["df"]
    enriched_df = calculate_atr(enriched_df)
    current_price = float(enriched_df["Close"].squeeze().iloc[-1])

    # ---- Price chart ----
    st.subheader(f"{ticker} - Price Chart")
    price_fig = go.Figure(
        data=[
            go.Candlestick(
                x=enriched_df.index,
                open=enriched_df["Open"].squeeze(),
                high=enriched_df["High"].squeeze(),
                low=enriched_df["Low"].squeeze(),
                close=enriched_df["Close"].squeeze(),
                name="Price",
            )
        ]
    )
    price_fig.update_layout(xaxis_rangeslider_visible=False, height=420)
    st.plotly_chart(price_fig, use_container_width=True)

    # ---- Indicator charts ----
    col1, col2 = st.columns(2)

    with col1:
        st.subheader("RSI (14)")
        rsi_fig = go.Figure()
        rsi_fig.add_trace(
            go.Scatter(x=enriched_df.index, y=enriched_df["RSI"], name="RSI")
        )
        rsi_fig.add_hline(y=70, line_dash="dash", line_color="red", annotation_text="Overbought (70)")
        rsi_fig.add_hline(y=30, line_dash="dash", line_color="green", annotation_text="Oversold (30)")
        rsi_fig.update_layout(height=300)
        st.plotly_chart(rsi_fig, use_container_width=True)

    with col2:
        st.subheader("MACD (12, 26, 9)")
        macd_fig = go.Figure()
        macd_fig.add_trace(
            go.Scatter(x=enriched_df.index, y=enriched_df["MACD_Line"], name="MACD Line")
        )
        macd_fig.add_trace(
            go.Scatter(x=enriched_df.index, y=enriched_df["Signal_Line"], name="Signal Line")
        )
        macd_fig.add_trace(
            go.Bar(x=enriched_df.index, y=enriched_df["MACD_Hist"], name="Histogram", opacity=0.4)
        )
        macd_fig.update_layout(height=300)
        st.plotly_chart(macd_fig, use_container_width=True)

    # ---- Current values ----
    st.subheader("Current Indicator Values")
    val_col1, val_col2, val_col3 = st.columns(3)
    val_col1.metric("RSI (14)", indicators["rsi"])
    val_col2.metric("MACD Line", indicators["macd_line"])
    val_col3.metric("Signal Line", indicators["signal_line"])

    # ---- Educational Explainer ----
    st.subheader("Educational Explainer")
    explainer_text = generate_educational_explainer(indicators)
    st.markdown(explainer_text)

    # ---- News Context ----
    st.markdown("---")
    render_news_ui(ticker)

    # ---- Risk Calculator ----
    st.markdown("---")
    position_size = render_risk_calculator(enriched_df, current_price)

    # ---- Manual Paper Trade Journal ----
    st.markdown("---")
    st.subheader("Manual Paper Trade Journal")
    st.caption(
        "Use these buttons to log your own manual paper-trade decisions "
        "while you study the data above. These do NOT execute any real or "
        "simulated orders."
    )
    btn_col1, btn_col2 = st.columns(2)
    with btn_col1:
        if st.button("Manual Paper Buy", type="primary"):
            st.success("Manual paper trade logged - BUY noted.")
    with btn_col2:
        if st.button("Manual Paper Sell", type="secondary"):
            st.success("Manual paper trade logged - SELL noted.")


# ---------------------------------------------------------------------------
# ENTRY POINT
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    render_ui()

