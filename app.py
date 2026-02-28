import streamlit as st
import yfinance as yf
import pandas as pd
import plotly.graph_objects as go
from datetime import datetime, timedelta


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
            # Flatten multi-level columns if present
            if isinstance(data.columns, pd.MultiIndex):
                      data.columns = data.columns.get_level_values(0)
                  return data


# ---------------------------------------------------------------------------
# 2. INDICATOR CALCULATIONS
# ---------------------------------------------------------------------------

def calculate_indicators(df: pd.DataFrame) -> dict:
      """
          Compute RSI (14-period) and MACD (12, 26, 9) from a Close price series.

              Parameters
                  ----------
                      df : pd.DataFrame
                              Must contain a 'Close' column.

                                  Returns
                                      -------
                                          dict with keys:
                                                  rsi           – current RSI value
                                                          macd_line     – current MACD line value
                                                                  signal_line   – current signal line value
                                                                          macd_hist     – current MACD histogram value
                                                                                  df            – original df augmented with indicator columns
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
# 3. EDUCATIONAL EXPLAINER  (strictly informational – no recommendations)
# ---------------------------------------------------------------------------

def generate_educational_explainer(indicators: dict) -> str:
      """
          Return an objective, textbook-style explanation of what the current
              RSI and MACD values traditionally indicate in technical analysis.

                  This function provides EDUCATION ONLY.
                      It does NOT rate, score, predict, or suggest any trading action.
                          """
    rsi = indicators["rsi"]
    macd_line = indicators["macd_line"]
    signal_line = indicators["signal_line"]
    macd_hist = indicators["macd_hist"]

    sections = []

    # ---- RSI explanation ----
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

    # ---- MACD explanation ----
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
                            "26-period EMA — a condition textbooks associate with upward "
                            "price trend momentum."
              )
elif macd_line < 0:
        sections.append(
                      "The MACD Line is negative (below the zero line), which "
                      "traditionally indicates that the 12-period EMA is below the "
                      "26-period EMA — a condition textbooks associate with downward "
                      "price trend momentum."
        )

    # ---- Disclaimer ----
    sections.append("---")
    sections.append(
              "*This explainer is strictly educational. It describes what these "
              "indicator values traditionally represent in technical analysis "
              "textbooks. Nothing here constitutes financial advice, a prediction, "
              "or a recommendation to buy, sell, or hold any security.*"
    )

    return "\n\n".join(sections)


# ---------------------------------------------------------------------------
# 4. STREAMLIT UI
# ---------------------------------------------------------------------------

def render_ui():
      """Render the full Streamlit interface."""

    st.set_page_config(page_title="StocksPage", layout="wide")
    st.title("StocksPage — Market Data & Educational Explainer")
    st.caption(
              "A read-only dashboard for viewing market data and learning what "
              "common technical indicators represent. No predictions. No advice."
    )

    # ---- Sidebar inputs ----
    ticker = st.sidebar.text_input("Ticker Symbol", value="AAPL").upper().strip()
    period = st.sidebar.selectbox(
              "Look-back Period",
              options=["1mo", "3mo", "6mo", "1y", "2y"],
              index=2,
    )

    if not ticker:
              st.info("Enter a ticker symbol in the sidebar to get started.")
              return

    # ---- Fetch data ----
    with st.spinner(f"Fetching data for {ticker}..."):
              df = fetch_market_data(ticker, period=period)

    if df.empty:
              return

    # ---- Calculate indicators ----
    indicators = calculate_indicators(df)
    enriched_df = indicators["df"]

    # ---- Price chart ----
    st.subheader(f"{ticker} — Price Chart")
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

    # ---- Indicator charts side-by-side ----
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

    # ---- Current indicator values ----
    st.subheader("Current Indicator Values")
    val_col1, val_col2, val_col3 = st.columns(3)
    val_col1.metric("RSI (14)", indicators["rsi"])
    val_col2.metric("MACD Line", indicators["macd_line"])
    val_col3.metric("Signal Line", indicators["signal_line"])

    # ---- Educational Explainer ----
    st.subheader("Educational Explainer")
    explainer_text = generate_educational_explainer(indicators)
    st.markdown(explainer_text)

    # ---- Manual Paper Trade Journal Buttons ----
    st.subheader("Manual Paper Trade Journal")
    st.caption(
              "Use these buttons to log your own manual paper-trade decisions "
              "while you study the data above. These do NOT execute any real or "
              "simulated orders."
    )
    btn_col1, btn_col2 = st.columns(2)
    with btn_col1:
              if st.button("Manual Paper Buy", type="primary"):
                            st.success("Manual paper trade logged — BUY noted.")
                    with btn_col2:
                              if st.button("Manual Paper Sell", type="secondary"):
                                            st.success("Manual paper trade logged — SELL noted.")


# ---------------------------------------------------------------------------
# ENTRY POINT
# ---------------------------------------------------------------------------

if __name__ == "__main__":
      render_ui()
