# ──────────────────────────────────────────────────────────────────────────────
# Quant AI Terminal – live‑money, hedge‑fund style, journalist narrative
# (including risk‑factor‑driven wait‑time estimator)
# ──────────────────────────────────────────────────────────────────────────────
import streamlit as st
import yfinance as yf
import pandas as pd
import plotly.graph_objects as go
import datetime
import numpy as np
from statistics import mean

from alpaca.trading.client import TradingClient
from alpaca.trading.requests import (
    MarketOrderRequest,
    TrailingStopOrderRequest,
)
from alpaca.trading.enums import OrderSide, TimeInForce

# ──────────────────────────────────────────────────────────────────────────────
# 1️⃣  PAGE CONFIG & SECRETS
# ──────────────────────────────────────────────────────────────────────────────
st.set_page_config(page_title="Quant AI Terminal – Live", layout="wide", initial_sidebar_state="collapsed")

# ----- Secrets -------------------------------------------------------------
API_KEY    = st.secrets["ALPACA_API_KEY"]
SECRET_KEY = st.secrets["ALPACA_SECRET_KEY"]
USE_PAPER  = bool(st.secrets.get("USE_PAPER", True))

RISK_TOLERANCE       = st.secrets.get("RISK_TOLERANCE", "medium").lower()
RISK_PCT             = float(st.secrets.get("RISK_PCT", 0.01))
COMMISSION_PER_SHARE = float(st.secrets.get("COMMISSION_PER_SHARE", 0.0))
FLAT_FEE             = float(st.secrets.get("FLAT_FEE", 0.0))

# ----- Initialise Alpaca client --------------------------------------------
try:
    trading_client = TradingClient(API_KEY, SECRET_KEY, paper=USE_PAPER)
except Exception as e:
    st.error(f"❌ Could not initialise Alpaca client: {e}")
    st.stop()

# ──────────────────────────────────────────────────────────────────────────────
# 2️⃣  SESSION STATE
# ──────────────────────────────────────────────────────────────────────────────
if "selected_ticker" not in st.session_state:
    st.session_state.selected_ticker = None

def go_back():
    st.session_state.selected_ticker = None

# ──────────────────────────────────────────────────────────────────────────────
# 3️⃣  ACCOUNT HELPERS (cash, buying power, tooltip)
# ──────────────────────────────────────────────────────────────────────────────
@st.cache_data(ttl=5)
def get_account_summary():
    acct = trading_client.get_account()
    return {
        "cash":          float(acct.cash or 0.0),
        "equity":        float(acct.equity or 0.0),
        "buying_power":  float(acct.buying_power or 0.0),
        "unrealised_pl": float(acct.unrealized_pl or 0.0),
        "realised_pl":   float(acct.realized_pl or 0.0),
    }

def display_cash_balance():
    acc = get_account_summary()
    cash = acc["cash"]
    bp   = acc["buying_power"]
    est_comm = 100 * COMMISSION_PER_SHARE + FLAT_FEE
    tooltip = (
        f"Cash: ${cash:,.2f}\\n"
        f"Buying Power: ${bp:,.2f}\\n"
        f"Sample commission (100 shares): ${est_comm:,.2f}"
    )
    html = f'''
    <div style="font-size:1.4rem; font-weight:600; margin-bottom:0.5rem;">
        💰 <span title="{tooltip}" style="cursor:help;">${cash:,.2f}</span>
    </div>
    '''
    st.markdown(html, unsafe_allow_html=True)

def get_all_positions():
    try:
        return trading_client.get_all_positions()
    except Exception:
        return []

# ──────────────────────────────────────────────────────────────────────────────
# 4️⃣  RISK‑TOLERANCE HELPERS (multiplier & probability cut‑off)
# ──────────────────────────────────────────────────────────────────────────────
# Map textual tolerance → numeric factor that shrinks the position size
RISK_FACTOR_MAP = {"low": 0.5, "medium": 1.0, "high": 1.5}
risk_factor = RISK_FACTOR_MAP.get(RISK_TOLERANCE, 1.0)

# Probability cut‑off per tolerance
PROB_CUTOFF_MAP = {"low": 70, "medium": 60, "high": 50}
prob_cutoff = PROB_CUTOFF_MAP.get(RISK_TOLERANCE, 60)

def risk_adjusted_shares(equity: float, atr: float) -> int:
    """Shares you may risk while respecting the chosen tolerance."""
    if atr <= 0:
        return 0
    adjusted_risk_pct = RISK_PCT * risk_factor          # shrink / enlarge risk
    risk_capital = equity * adjusted_risk_pct
    stop_distance = atr * 1.5
    cash_available = max(risk_capital - FLAT_FEE, 0)
    cost_per_share = stop_distance + COMMISSION_PER_SHARE
    return int(cash_available // cost_per_share)

def apply_risk_filter(p_long: int, p_short: int, p_chop: int):
    """Return a dict telling us whether BUY/SELL buttons should be shown."""
    return {
        "buy":  p_long  >= prob_cutoff,
        "sell": p_short >= prob_cutoff,
        "chop": p_chop >= prob_cutoff,
    }

# ──────────────────────────────────────────────────────────────────────────────
# 5️⃣  DATA ENGINE (yFinance + technicals) – unchanged except docstring
# ──────────────────────────────────────────────────────────────────────────────
@st.cache_data(ttl=30)
def fetch_data(ticker: str, interval: str) -> pd.DataFrame:
    period_map = {"1m": "1d", "5m": "5d", "15m": "5d", "1h": "1mo", "1d": "1y"}
    df = yf.download(ticker, period=period_map[interval], interval=interval, progress=False)
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    return df

def calculate_technicals(df: pd.DataFrame):
    if df.empty or len(df) < 26:
        return df
    df = df.copy()
    close = df["Close"]

    # RSI
    delta = close.diff()
    gain = delta.clip(lower=0).rolling(14).mean()
    loss = (-delta.clip(upper=0)).rolling(14).mean()
    rs = gain / loss
    df["RSI"] = 100 - (100 / (1 + rs))

    # MACD
    ema12 = close.ewm(span=12, adjust=False).mean()
    ema26 = close.ewm(span=26, adjust=False).mean()
    df["MACD"] = ema12 - ema26
    df["Signal"] = df["MACD"].ewm(span=9, adjust=False).mean()

    # ATR
    high, low, prev = df["High"], df["Low"], close.shift(1)
    tr = pd.concat([high - low,
                    (high - prev).abs(),
                    (low - prev).abs()], axis=1).max(axis=1)
    df["ATR"] = tr.rolling(14).mean()
    return df

# ──────────────────────────────────────────────────────────────────────────────
# 6️⃣  AI SCORE CORE – split so we can reuse on historic slices
# ──────────────────────────────────────────────────────────────────────────────
def calc_ai_scores(df_1m: pd.DataFrame,
                   df_5m: pd.DataFrame,
                   df_15m: pd.DataFrame):
    """
    Implements the exact logic of the original `generate_ai_probabilities`,
    but works on **already‑calculated** data frames (so we can feed historic
    slices).
    Returns (p_long, p_short, p_chop) as integers 0‑100.
    """
    if df_1m.empty or df_5m.empty or df_15m.empty:
        return 33, 33, 34

    # MACD direction per timeframe
    sig_1m  = 1 if df_1m["MACD"].iloc[-1] > df_1m["Signal"].iloc[-1] else -1
    sig_5m  = 1 if df_5m["MACD"].iloc[-1] > df_5m["Signal"].iloc[-1] else -1
    sig_15m = 1 if df_15m["MACD"].iloc[-1] > df_15m["Signal"].iloc[-1] else -1
    rsi_1m = df_1m["RSI"].iloc[-1]

    bull, bear, chop = 0, 0, 15

    bull += 40 if sig_15m == 1 else 0
    bear += 40 if sig_15m == -1 else 0
    bull += 30 if sig_5m == 1 else 0
    bear += 30 if sig_5m == -1 else 0
    bull += 15 if sig_1m == 1 else 0
    bear += 15 if sig_1m == -1 else 0

    if rsi_1m > 70:
        bear += 15
        bull = max(bull - 10, 0)
    elif rsi_1m < 30:
        bull += 15
        bear = max(bear - 10, 0)
    else:
        chop += 15

    total = max(bull + bear + chop, 1)
    p_long = max(int((bull / total) * 100), 5)
    p_short = max(int((bear / total) * 100), 5)
    p_chop = max(100 - p_long - p_short, 0)

    return p_long, p_short, p_chop

def generate_ai_probabilities(ticker: str):
    """Convenient wrapper that loads the three time‑frames and calls `calc_ai_scores`. """
    df_1m = calculate_technicals(fetch_data(ticker, "1m"))
    df_5m = calculate_technicals(fetch_data(ticker, "5m"))
    df_15m = calculate_technicals(fetch_data(ticker, "15m"))
    return calc_ai_scores(df_1m, df_5m, df_15m)

# ──────────────────────────────────────────────────────────────────────────────
# 7️⃣  ESTIMATED WAIT‑TIME LOGIC
# ──────────────────────────────────────────────────────────────────────────────
INTERVAL_MINUTES = {"1m":1, "5m":5, "15m":15, "1h":60, "1d":1440}

def estimate_wait_time_for_signal(ticker: str,
                                 interval: str,
                                 target_prob: int,
                                 current_prob: int,
                                 direction: str = "buy",
                                 history_bars: int = 6) -> str:
    """
    Returns a human‑readable estimate (e.g. "≈ 12 minutes") or
    "unlikely" if the trend does not suggest progress.
    The estimate is *inflated* when the user is in low‑risk mode (risk_factor < 1).
    """
    # Map interval → minutes per bar
    minutes_per_bar = INTERVAL_MINUTES.get(interval, 1)

    # Load full historical data for the three time‑frames only once
    df_1m_full  = calculate_technicals(fetch_data(ticker, "1m"))
    df_5m_full  = calculate_technicals(fetch_data(ticker, "5m"))
    df_15m_full = calculate_technicals(fetch_data(ticker, "15m"))

    # We need at least `history_bars` previous points
    if len(df_1m_full) < history_bars + 1:
        return "—"

    # Grab past probabilities (excluding the current bar)
    past_probs = []
    for i in range(-history_bars, 0):          # e.g. -6 … -1  (skip the latest bar at -0)
        # slice each timeframe up to the i‑th row (inclusive)
        p_long_i, _, _ = calc_ai_scores(
            df_1m_full.iloc[:i],
            df_5m_full.iloc[:i],
            df_15m_full.iloc[:i]
        )
        past_probs.append(p_long_i)

    # Compute average change per bar
    deltas = [j - i for i, j in zip(past_probs[:-1], past_probs[1:])]
    if not deltas:
        return "—"
    avg_delta = mean(deltas)

    if avg_delta <= 0:
        # No upward trend detected – we consider the target unlikely soon.
        return "unlikely"

    # Bars needed to climb from current to target
    bars_needed = (target_prob - current_prob) / avg_delta
    if bars_needed < 0:
        # Already above target (should not happen because we call only when below)
        bars_needed = 0

    # Convert to minutes, then inflate by risk‑factor (low risk → longer wait)
    raw_minutes = bars_needed * minutes_per_bar
    inflated_minutes = raw_minutes / risk_factor        # risk_factor <1 → longer wait

    # Round to a nice integer and cap to a maximum (e.g., 8 hours) for readability
    est_minutes = int(round(inflated_minutes))
    if est_minutes <= 0:
        return "≈ now"
    # Decide a friendly unit
    if est_minutes < 60:
        return f"≈ {est_minutes} minutes"
    elif est_minutes < 1440:
        hours = round(est_minutes / 60, 1)
        return f"≈ {hours} hours"
    else:
        days = round(est_minutes / 1440, 1)
        return f"≈ {days} days"

# ──────────────────────────────────────────────────────────────────────────────
# 8️⃣  VISUAL ENGINE (unchanged – candlestick, ATR stop line)
# ──────────────────────────────────────────────────────────────────────────────
def draw_annotated_chart(df, ticker, interval):
    if df.empty or len(df) < 14:
        return go.Figure()
    cur_price = float(df["Close"].iloc[-1])
    cur_atr   = float(df["ATR"].iloc[-1]) if not pd.isna(df["ATR"].iloc[-1]) else 0.0
    stop_lvl  = cur_price - (cur_atr * 1.5)

    fig = go.Figure(
        data=[go.Candlestick(
            x=df.index,
            open=df["Open"], high=df["High"],
            low=df["Low"], close=df["Close"],
            name="Price"
        )]
    )
    if cur_atr > 0:
        fig.add_hline(y=stop_lvl,
                      line_dash="dot", line_color="red",
                      annotation_text=f"ATR Stop (${stop_lvl:,.2f})",
                      annotation_position="bottom right")
    fig.update_layout(
        title=f"{ticker} Live Chart ({interval})",
        xaxis_rangeslider_visible=False,
        height=450,
        margin=dict(l=0, r=0, t=40, b=0),
        template="plotly_dark")
    return fig

# ──────────────────────────────────────────────────────────────────────────────
# 9️⃣  DYNAMIC SCANNER ENGINE (unchanged)
# ──────────────────────────────────────────────────────────────────────────────
def scan_for_opportunities():
    watchlist = ["NVDA", "AAPL", "TSLA", "AMD", "MSFT", "GOOGL", "AMZN", "META"]
    opportunities = []
    for ticker in watchlist:
        try:
            df_5m  = calculate_technicals(fetch_data(ticker, "5m"))
            df_15m = calculate_technicals(fetch_data(ticker, "15m"))
            if df_5m.empty or df_15m.empty:
                continue
            rsi_15m   = df_15m["RSI"].iloc[-1]
            macd_5m   = df_5m["MACD"].iloc[-1]
            signal_5m = df_5m["Signal"].iloc[-1]
            macd_15m  = df_15m["MACD"].iloc[-1]
            signal_15m= df_15m["Signal"].iloc[-1]

            bullish_5m  = macd_5m > signal_5m
            bullish_15m = macd_15m > signal_15m

            if bullish_5m and bullish_15m:
                opportunities.append({"ticker": ticker,
                                      "reason": "5m & 15m MACD bullish alignment detected."})
            elif not bullish_5m and not bullish_15m:
                opportunities.append({"ticker": ticker,
                                      "reason": "5m & 15m MACD bearish alignment detected."})
            elif not pd.isna(rsi_15m) and rsi_15m < 30:
                opportunities.append({"ticker": ticker,
                                      "reason": f"15m RSI oversold ({rsi_15m:.1f}) – potential bounce."})
            elif not pd.isna(rsi_15m) and rsi_15m > 70:
                opportunities.append({"ticker": ticker,
                                      "reason": f"15m RSI overbought ({rsi_15m:.1f}) – potential reversal."})
        except Exception:
            continue
    return opportunities

# ──────────────────────────────────────────────────────────────────────────────
# 🔟  SCREEN 1 – RADAR SCANNER (unchanged apart from cash display)
# ──────────────────────────────────────────────────────────────────────────────
def render_scanner_screen():
    st.title("📡 Multi‑Timeframe AI Scanner")
    st.caption("Scanning watchlist for live multi‑timeframe signals.")
    st.markdown("---")
    display_cash_balance()

    with st.spinner("Scanning watchlist…"):
        opportunities = scan_for_opportunities()

    if not opportunities:
        st.info("🚩 No actionable signals detected right now – check back later.")
        return

    for opp in opportunities:
        c1, c2, c3 = st.columns([1, 4, 1])
        with c1:
            st.subheader(opp["ticker"])
        with c2:
            st.markdown(f"**Signal:** {opp['reason']}")
        with c3:
            if st.button(f"Analyze {opp['ticker']}", key=opp["ticker"], use_container_width=True):
                st.session_state.selected_ticker = opp["ticker"]
                st.rerun()
        st.markdown("---")

# ──────────────────────────────────────────────────────────────────────────────
# 1️⃣1️⃣  SCREEN 2 – TACTICAL COCKPIT (now shows estimated wait‑time)
# ──────────────────────────────────────────────────────────────────────────────
def render_tactical_screen(ticker):
    # ── Header (Back, Refresh, Cash) ──────────────────────────────────────
    col_back, col_refresh, _ = st.columns([1, 1, 6])
    with col_back:
        st.button("⬅ Back to Radar", on_click=go_back)
    with col_refresh:
        if st.button("🔄 Refresh Live Data"):
            st.rerun()
    display_cash_balance()

    # ── Choose chart timeframe ───────────────────────────────────────────────
    selected_interval = st.selectbox(
        "Chart Timeframe",
        ["1m", "5m", "15m", "1h", "1d"],
        index=0
    )

    # ── Load data, run AI, compute sizing ─────────────────────────────────────
    with st.spinner(f"AI analysing {ticker} @ {selected_interval}…"):
        df = calculate_technicals(fetch_data(ticker, selected_interval))
        p_long, p_short, p_chop = generate_ai_probabilities(ticker)

    if df.empty:
        st.error("❌ Market data unavailable for this ticker.")
        st.stop()

    # ── Journalistic brief (top of screen) ───────────────────────────────────
    st.subheader("📰 Morning Briefing")
    st.caption(
        f"""
        **{datetime.datetime.now().strftime('%b %d, %Y %H:%M ET')} – AI Verdict**  

        {('🟢 BUY' if p_long >= prob_cutoff else '')}
        {('🔴 SELL' if p_short >= prob_cutoff else '')}
        {('⚪ SIDEWAYS' if p_chop >= prob_cutoff else '')}  

        Current price: ${df['Close'].iloc[-1]:,.2f} – ATR: ${df['ATR'].iloc[-1]:,.2f}
        (risk‑tolerance: **{RISK_TOLERANCE.title()}**)  
        """
    )

    # ── Metric strip (price / RSI / ATR) ─────────────────────────────────────
    latest = df.iloc[-1]
    price = float(latest["Close"])
    rsi   = float(latest["RSI"]) if not pd.isna(latest["RSI"]) else 50.0
    atr   = float(latest["ATR"]) if not pd.isna(latest["ATR"]) else 0.0

    m1, m2, m3 = st.columns(3)
    m1.metric("Live Price", f"${price:,.2f}")
    m2.metric(f"{selected_interval} RSI", f"{rsi:.1f}")
    m3.metric("Volatility (ATR)", f"${atr:,.2f}")

    # ── Chart & AI matrix ─────────────────────────────────────────────────────
    col_chart, col_data = st.columns([2, 1])
    with col_chart:
        st.plotly_chart(draw_annotated_chart(df, ticker, selected_interval),
                         use_container_width=True)

    with col_data:
        st.subheader("🤖 AI Probability Matrix")
        st.caption("1‑, 5‑ and 15‑minute trend alignment")
        st.progress(p_long / 100);   st.markdown(f"**🟢 LONG:** {p_long}%")
        st.progress(p_short / 100); st.markdown(f"**🔴 SHORT:** {p_short}%")
        st.progress(p_chop / 100);  st.markdown(f"**🟡 CHOP:** {p_chop}%")

        # Verdict colour based on probability + risk tolerance
        if p_long >= prob_cutoff:
            st.success("AI Verdict → **Strong BUY**")
        elif p_short >= prob_cutoff:
            st.error("AI Verdict → **Strong SELL**")
        else:
            st.warning("AI Verdict → Mixed signals – consider staying flat.")

        # ----- ESTIMATED WAIT TIME -------------------------------------------------
        # Show an estimate only when the button would be *disabled* because we are
        # below the cut‑off.
        if not p_long >= prob_cutoff:
            wait_buy = estimate_wait_time_for_signal(
                ticker, selected_interval, prob_cutoff, p_long,
                direction="buy", history_bars=6)
            st.info(f"⏳ Estimated wait for BUY threshold: **{wait_buy}**")
        if not p_short >= prob_cutoff:
            wait_sell = estimate_wait_time_for_signal(
                ticker, selected_interval, prob_cutoff, p_short,
                direction="sell", history_bars=6)
            st.info(f"⏳ Estimated wait for SELL threshold: **{wait_sell}**")

    # ── POSITION SIZING (risk‑adjusted) ───────────────────────────────────────
    st.markdown("---")
    st.subheader("⚖️ Position Sizing (risk‑adjusted)")

    acct = get_account_summary()
    shares = risk_adjusted_shares(acct["equity"], atr)

    sizing_tooltip = (
        f"Equity: ${acct['equity']:,.2f}\\n"
        f"Risk tolerance: {RISK_TOLERANCE.title()} (factor ×{risk_factor})\\n"
        f"ATR×1.5 stop: ${atr*1.5:,.2f}\\n"
        f"Flat fee: ${FLAT_FEE:,.2f}, commission/share: ${COMMISSION_PER_SHARE:,.4f}\\n"
        f"→ Max shares you may risk now: {shares}"
    )
    st.markdown(
        f'''
        <div style="font-size:1.1rem;">
            📊 <span title="{sizing_tooltip}" style="cursor:help;">
                Calculated optimal sizing: **{shares} shares**
            </span>
        </div>
        ''',
        unsafe_allow_html=True
    )

    # ── Confirmation (required for live) ─────────────────────────────────────
    if not USE_PAPER:
        confirm = st.checkbox(
            "✅ I understand this is a **real‑money** market order and accept the risk.",
            value=False,
            help="Live orders cannot be undone. Double‑check the size, ticker and stop‑loss."
        )
    else:
        confirm = True   # paper mode auto‑confirms

    # ── BUY / SELL BUTTONS (with trailing‑stop attached) ───────────────────────
    c_buy, c_sell = st.columns(2)
    with c_buy:
        if st.button(f"🟢 BUY {shares} SHARES", use_container_width=True, type="primary"):
            if shares <= 0:
                st.warning("⚠️ Calculated share size is 0 – order not sent.")
            elif not confirm:
                st.warning("⚠️ Please tick the confirmation box before placing a real order.")
            else:
                try:
                    # 1️⃣ Market BUY
                    buy_order = MarketOrderRequest(
                        symbol=ticker,
                        qty=shares,
                        side=OrderSide.BUY,
                        time_in_force=TimeInForce.DAY
                    )
                    trading_client.submit_order(buy_order)

                    # 2️⃣ Attach trailing STOP (sell side)
                    trail_percent = 0.5   # 0.5 % trailing stop – tweak as you like
                    trailing_stop = TrailingStopOrderRequest(
                        symbol=ticker,
                        qty=shares,
                        side=OrderSide.SELL,
                        type="trailing_stop",
                        time_in_force=TimeInForce.GTC,
                        trail_percent=trail_percent
                    )
                    trading_client.submit_order(trailing_stop)

                    st.success(f"✅ BUY order for **{shares}** shares of **{ticker}** submitted. "
                               f"A trailing‑stop ({trail_percent*100:.1f} %) is now attached.")
                except Exception as e:
                    st.error(f"❌ Order failed – {e}")

    with c_sell:
        if st.button(f"🔴 SELL {shares} SHARES", use_container_width=True):
            if shares <= 0:
                st.warning("⚠️ Calculated share size is 0 – order not sent.")
            elif not confirm:
                st.warning("⚠️ Please tick the confirmation box before placing a real order.")
            else:
                try:
                    sell_order = MarketOrderRequest(
                        symbol=ticker,
                        qty=shares,
                        side=OrderSide.SELL,
                        time_in_force=TimeInForce.DAY
                    )
                    trading_client.submit_order(sell_order)

                    # Trailing stop in the opposite direction (buy‑stop)
                    trail_percent = 0.5
                    trailing_stop = TrailingStopOrderRequest(
                        symbol=ticker,
                        qty=shares,
                        side=OrderSide.BUY,
                        type="trailing_stop",
                        time_in_force=TimeInForce.GTC,
                        trail_percent=trail_percent
                    )
                    trading_client.submit_order(trailing_stop)

                    st.success(f"✅ SELL order for **{shares}** shares of **{ticker}** submitted. "
                               f"A trailing‑stop ({trail_percent*100:.1f} %) is now attached.")
                except Exception as e:
                    st.error(f"❌ Order failed – {e}")

    # ── CURRENT POSITIONS QUICK VIEW ────────────────────────────────────────
    st.markdown("---")
    positions = get_all_positions()
    if positions:
        st.subheader("📊 Current Positions")
        pos_df = pd.DataFrame([{
            "Ticker": p.symbol,
            "Qty": int(p.qty),
            "Avg Entry": f"${float(p.avg_entry_price):,.2f}",
            "Current": f"${float(p.current_price):,.2f}",
            "Unrealised PnL": f"${float(p.unrealized_pl):,.2f}",
            "Side": "Long" if float(p.qty) > 0 else "Short"
        } for p in positions])
        st.dataframe(pos_df, hide_index=True, use_container_width=True)
    else:
        st.info("🚫 No open positions at the moment.")

# ──────────────────────────────────────────────────────────────────────────────
# 2️⃣2️⃣  MAIN – which screen to render
# ──────────────────────────────────────────────────────────────────────────────
if st.session_state.selected_ticker is None:
    render_scanner_screen()
else:
    render_tactical_screen(st.session_state.selected_ticker)
