# ──────────────────────────────────────────────────────────────────────────────
# Quant AI Terminal – live-money, hedge-fund style, journalist narrative
# (with conversational intel engine + dynamic confidence meter)
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
# 1  PAGE CONFIG & SECRETS
# ──────────────────────────────────────────────────────────────────────────────
st.set_page_config(page_title="Quant AI Terminal - Live", layout="wide", initial_sidebar_state="collapsed")

API_KEY    = st.secrets["ALPACA_API_KEY"]
SECRET_KEY = st.secrets["ALPACA_SECRET_KEY"]
USE_PAPER  = bool(st.secrets.get("USE_PAPER", True))

RISK_TOLERANCE       = st.secrets.get("RISK_TOLERANCE", "medium").lower()
RISK_PCT             = float(st.secrets.get("RISK_PCT", 0.01))
COMMISSION_PER_SHARE = float(st.secrets.get("COMMISSION_PER_SHARE", 0.0))
FLAT_FEE             = float(st.secrets.get("FLAT_FEE", 0.0))

try:
    trading_client = TradingClient(API_KEY, SECRET_KEY, paper=USE_PAPER)
except Exception as e:
    st.error(f"Could not initialise Alpaca client: {e}")
    st.stop()

# ──────────────────────────────────────────────────────────────────────────────
# 2  SESSION STATE
# ──────────────────────────────────────────────────────────────────────────────
if "selected_ticker" not in st.session_state:
    st.session_state.selected_ticker = None
if "intel_answers" not in st.session_state:
    st.session_state.intel_answers = {}
if "intel_submitted" not in st.session_state:
    st.session_state.intel_submitted = False
if "chat_log" not in st.session_state:
    st.session_state.chat_log = []

def go_back():
    st.session_state.selected_ticker = None
    st.session_state.intel_answers = {}
    st.session_state.intel_submitted = False
    st.session_state.chat_log = []
# ──────────────────────────────────────────────────────────────────────────────
# 3  ACCOUNT HELPERS
# ──────────────────────────────────────────────────────────────────────────────
@st.cache_data(ttl=5)
def get_account_summary():
    acct = trading_client.get_account()
    return {
        "cash": float(acct.cash or 0.0),
        "equity": float(acct.equity or 0.0),
        "buying_power": float(acct.buying_power or 0.0),
        "unrealised_pl": float(acct.unrealized_pl or 0.0),
        "realised_pl": float(acct.realized_pl or 0.0),
    }

def display_cash_balance():
    acc = get_account_summary()
    cash = acc["cash"]
    bp = acc["buying_power"]
    est_comm = 100 * COMMISSION_PER_SHARE + FLAT_FEE
    tooltip = (
        f"Cash: ${cash:,.2f}\\nBuying Power: ${bp:,.2f}\\n"
        f"Sample commission (100 shares): ${est_comm:,.2f}"
    )
    html = f'''
    <div style="font-size:1.4rem; font-weight:600; margin-bottom:0.5rem;">
    <span title="{tooltip}" style="cursor:help;">${cash:,.2f}</span>
    </div>
    '''
    st.markdown(html, unsafe_allow_html=True)

def get_all_positions():
    try:
        return trading_client.get_all_positions()
    except Exception:
        return []

# ──────────────────────────────────────────────────────────────────────────────
# 4  RISK TOLERANCE HELPERS
# ──────────────────────────────────────────────────────────────────────────────
RISK_FACTOR_MAP = {"low": 0.5, "medium": 1.0, "high": 1.5}
risk_factor = RISK_FACTOR_MAP.get(RISK_TOLERANCE, 1.0)

PROB_CUTOFF_MAP = {"low": 70, "medium": 60, "high": 50}
prob_cutoff = PROB_CUTOFF_MAP.get(RISK_TOLERANCE, 60)

def risk_adjusted_shares(equity, atr):
    if atr <= 0:
        return 0
    adjusted_risk_pct = RISK_PCT * risk_factor
    risk_capital = equity * adjusted_risk_pct
    stop_distance = atr * 1.5
    cash_available = max(risk_capital - FLAT_FEE, 0)
    cost_per_share = stop_distance + COMMISSION_PER_SHARE
    return int(cash_available // cost_per_share)

def apply_risk_filter(p_long, p_short, p_chop):
    return {
        "buy": p_long >= prob_cutoff,
        "sell": p_short >= prob_cutoff,
        "chop": p_chop >= prob_cutoff,
    }
# ──────────────────────────────────────────────────────────────────────────────
# 5  DATA ENGINE (yFinance + technicals)
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
    delta = close.diff()
    gain = delta.clip(lower=0).rolling(14).mean()
    loss = (-delta.clip(upper=0)).rolling(14).mean()
    rs = gain / loss
    df["RSI"] = 100 - (100 / (1 + rs))
    ema12 = close.ewm(span=12, adjust=False).mean()
    ema26 = close.ewm(span=26, adjust=False).mean()
    df["MACD"] = ema12 - ema26
    df["Signal"] = df["MACD"].ewm(span=9, adjust=False).mean()
    high, low, prev = df["High"], df["Low"], close.shift(1)
    tr = pd.concat([high - low,
                    (high - prev).abs(),
                    (low - prev).abs()], axis=1).max(axis=1)
    df["ATR"] = tr.rolling(14).mean()
    return df
# ──────────────────────────────────────────────────────────────────────────────
# 6  AI SCORE CORE (technical-only base scores)
# ──────────────────────────────────────────────────────────────────────────────
def calc_ai_scores(df_1m, df_5m, df_15m):
    """Base technical scores from MACD/RSI across 3 timeframes."""
    if df_1m.empty or df_5m.empty or df_15m.empty:
        return 33, 33, 34
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
    p_long  = max(int((bull / total) * 100), 5)
    p_short = max(int((bear / total) * 100), 5)
    p_chop  = max(100 - p_long - p_short, 0)
    return p_long, p_short, p_chop

def generate_ai_probabilities(ticker: str):
    df_1m  = calculate_technicals(fetch_data(ticker, "1m"))
    df_5m  = calculate_technicals(fetch_data(ticker, "5m"))
    df_15m = calculate_technicals(fetch_data(ticker, "15m"))
    return calc_ai_scores(df_1m, df_5m, df_15m)
# ──────────────────────────────────────────────────────────────────────────────
# 7  CONVERSATIONAL INTEL ENGINE
#    The AI identifies what it CANNOT see from technicals alone, then asks
#    the user targeted questions. Answers shift the confidence meter.
# ──────────────────────────────────────────────────────────────────────────────

# --- Question templates keyed by what the AI detects ---
# Each question has: text, impact direction, weight
INTEL_QUESTIONS = {
    "earnings_near": {
        "text": "Is there an earnings report within the next 5 trading days?",
        "options": ["Yes - expected beat", "Yes - expected miss", "Yes - uncertain", "No / not soon"],
        "weights": {"Yes - expected beat": 12, "Yes - expected miss": -12, "Yes - uncertain": -5, "No / not soon": 0},
        "tag": "earnings",
    },
    "news_catalyst": {
        "text": "Are you aware of any breaking news or catalyst for this ticker?",
        "options": ["Strong bullish catalyst", "Mild positive", "None that I know of", "Negative news"],
        "weights": {"Strong bullish catalyst": 15, "Mild positive": 7, "None that I know of": 0, "Negative news": -15},
        "tag": "news",
    },
    "sector_rotation": {
        "text": "Is money flowing INTO or OUT OF this sector right now?",
        "options": ["Money flowing in", "Neutral / unclear", "Money flowing out"],
        "weights": {"Money flowing in": 8, "Neutral / unclear": 0, "Money flowing out": -8},
        "tag": "sector",
    },
    "insider_activity": {
        "text": "Any recent insider buying or selling (SEC Form 4)?",
        "options": ["Insider buying", "No notable activity", "Insider selling"],
        "weights": {"Insider buying": 10, "No notable activity": 0, "Insider selling": -10},
        "tag": "insider",
    },
    "volume_confirm": {
        "text": "Is today's volume notably higher than the 20-day average?",
        "options": ["Yes - much higher", "About average", "Lower than usual"],
        "weights": {"Yes - much higher": 6, "About average": 0, "Lower than usual": -4},
        "tag": "volume",
    },
    "macro_headwind": {
        "text": "Any major macro events today (Fed, CPI, jobs report)?",
        "options": ["Yes - likely market tailwind", "Yes - likely headwind", "Nothing major today"],
        "weights": {"Yes - likely market tailwind": 8, "Yes - likely headwind": -10, "Nothing major today": 0},
        "tag": "macro",
    },
    "conviction_level": {
        "text": "What is YOUR gut conviction on this trade direction?",
        "options": ["High conviction long", "Moderate lean long", "Truly neutral", "Moderate lean short", "High conviction short"],
        "weights": {"High conviction long": 10, "Moderate lean long": 5, "Truly neutral": 0, "Moderate lean short": -5, "High conviction short": -10},
        "tag": "conviction",
    },
}
def pick_questions(p_long, p_short, p_chop, rsi, atr, price):
    """
    Based on current technicals, decide WHICH questions to ask.
    The AI only asks what it needs to know - not everything every time.
    Returns a list of question keys.
    """
    questions = []
    # Always ask about earnings and the user's conviction
    questions.append("earnings_near")
    questions.append("conviction_level")
    
    # If signals are mixed (no clear direction), ask more
    spread = abs(p_long - p_short)
    if spread < 20:
        questions.append("news_catalyst")
        questions.append("sector_rotation")
        questions.append("macro_headwind")
    elif spread < 40:
        questions.append("news_catalyst")
        questions.append("volume_confirm")
    else:
        # Strong signal - just confirm no headwinds
        questions.append("macro_headwind")
    
    # If RSI is extreme, check for insider activity (potential trap)
    if rsi > 65 or rsi < 35:
        if "insider_activity" not in questions:
            questions.append("insider_activity")
    
    return questions

def compute_intel_adjustment(answers):
    """
    Take user answers and compute a single adjustment value.
    Positive = more bullish, Negative = more bearish.
    Returns (adjustment_points, breakdown_dict)
    """
    total_adj = 0
    breakdown = {}
    for qkey, answer in answers.items():
        if qkey in INTEL_QUESTIONS:
            q = INTEL_QUESTIONS[qkey]
            pts = q["weights"].get(answer, 0)
            total_adj += pts
            breakdown[q["tag"]] = pts
    # Cap adjustment so user intel cannot swing more than +/- 30 points
    total_adj = max(min(total_adj, 30), -30)
    return total_adj, breakdown

def blend_scores(p_long, p_short, p_chop, intel_adj):
    """
    Blend base technical scores with the intel adjustment.
    Positive intel_adj boosts long / shrinks short.
    Negative intel_adj boosts short / shrinks long.
    Returns (final_long, final_short, final_chop)
    """
    # Split adjustment: positive helps long, negative helps short
    if intel_adj >= 0:
        adj_long = intel_adj
        adj_short = -intel_adj // 2
    else:
        adj_long = intel_adj
        adj_short = abs(intel_adj)
    
    final_long  = max(min(p_long + adj_long, 95), 5)
    final_short = max(min(p_short + adj_short, 95), 5)
    final_chop  = max(100 - final_long - final_short, 0)
    return int(final_long), int(final_short), int(final_chop)

def generate_ai_narrative(ticker, p_long, p_short, rsi, atr, intel_adj, breakdown):
    """
    Generate a hedge-fund style narrative about the current setup.
    This tells the user WHY the AI is leaning a certain direction.
    """
    direction = "LONG" if p_long > p_short else "SHORT" if p_short > p_long else "FLAT"
    confidence = max(p_long, p_short)
    
    lines = []
    lines.append(f"**AI Assessment for {ticker}:**")
    
    # Technical read
    if confidence >= prob_cutoff:
        lines.append(f"Technicals are aligned {direction} with {confidence}% conviction.")
    else:
        lines.append(f"Technicals are inconclusive. Leaning {direction} at {confidence}% - below the {prob_cutoff}% threshold.")
    
    # RSI context
    if rsi > 70:
        lines.append(f"RSI at {rsi:.0f} is overbought territory - caution on new longs.")
    elif rsi < 30:
        lines.append(f"RSI at {rsi:.0f} is oversold - potential bounce setup.")
    
    # Intel contribution
    if intel_adj > 0:
        lines.append(f"Your intel adds +{intel_adj}pts of bullish conviction.")
    elif intel_adj < 0:
        lines.append(f"Your intel subtracts {intel_adj}pts - bearish lean from fundamentals.")
    else:
        lines.append("Your intel is neutral - no adjustment to the base read.")
    
    # Specific factor callouts
    for tag, pts in breakdown.items():
        if pts >= 10:
            lines.append(f"  -> {tag.upper()} is a strong tailwind (+{pts}pts).")
        elif pts <= -10:
            lines.append(f"  -> {tag.upper()} is a significant headwind ({pts}pts).")
    
    return "\n".join(lines)
# ──────────────────────────────────────────────────────────────────────────────
# 8  WAIT-TIME ESTIMATOR
# ──────────────────────────────────────────────────────────────────────────────
INTERVAL_MINUTES = {"1m":1, "5m":5, "15m":15, "1h":60, "1d":1440}

def estimate_wait_time_for_signal(ticker, interval, target_prob, current_prob,
                                  direction="buy", history_bars=6):
    minutes_per_bar = INTERVAL_MINUTES.get(interval, 1)
    df_1m_full  = calculate_technicals(fetch_data(ticker, "1m"))
    df_5m_full  = calculate_technicals(fetch_data(ticker, "5m"))
    df_15m_full = calculate_technicals(fetch_data(ticker, "15m"))
    if len(df_1m_full) < history_bars + 1:
        return "insufficient data"
    past_probs = []
    for i in range(-history_bars, 0):
        scores = calc_ai_scores(
            df_1m_full.iloc[:i], df_5m_full.iloc[:i], df_15m_full.iloc[:i]
        )
        # Use p_long for buy direction, p_short for sell
        past_probs.append(scores[0] if direction == "buy" else scores[1])
    deltas = [j - i for i, j in zip(past_probs[:-1], past_probs[1:])]
    if not deltas:
        return "insufficient data"
    avg_delta = mean(deltas)
    if avg_delta <= 0:
        return "unlikely (trend moving away)"
    bars_needed = (target_prob - current_prob) / avg_delta
    if bars_needed < 0:
        bars_needed = 0
    raw_minutes = bars_needed * minutes_per_bar
    inflated_minutes = raw_minutes / risk_factor
    est_minutes = int(round(inflated_minutes))
    if est_minutes <= 0:
        return "imminent"
    if est_minutes < 60:
        return f"~{est_minutes} minutes"
    elif est_minutes < 1440:
        return f"~{round(est_minutes / 60, 1)} hours"
    else:
        return f"~{round(est_minutes / 1440, 1)} days"

# ──────────────────────────────────────────────────────────────────────────────
# 9  CHART ENGINE
# ──────────────────────────────────────────────────────────────────────────────
def draw_annotated_chart(df, ticker, interval):
    if df.empty or len(df) < 14:
        return go.Figure()
    cur_price = float(df["Close"].iloc[-1])
    cur_atr = float(df["ATR"].iloc[-1]) if not pd.isna(df["ATR"].iloc[-1]) else 0.0
    stop_lvl = cur_price - (cur_atr * 1.5)
    fig = go.Figure(
        data=[go.Candlestick(
            x=df.index,
            open=df["Open"], high=df["High"],
            low=df["Low"], close=df["Close"],
            name="Price"
        )]
    )
    if cur_atr > 0:
        fig.add_hline(y=stop_lvl, line_dash="dot", line_color="red",
                      annotation_text=f"ATR Stop (${stop_lvl:,.2f})",
                      annotation_position="bottom right")
    fig.update_layout(
        title=f"{ticker} Live Chart ({interval})",
        xaxis_rangeslider_visible=False,
        height=450,
        margin=dict(l=0, r=0, t=40, b=0),
        template="plotly_dark")
    return fig
# ──────────────────────────────────────────────────────────────────────────────
# 10  SCANNER ENGINE
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
            signal_15m = df_15m["Signal"].iloc[-1]
            bullish_5m  = macd_5m > signal_5m
            bullish_15m = macd_15m > signal_15m
            if bullish_5m and bullish_15m:
                opportunities.append({"ticker": ticker,
                    "reason": "5m & 15m MACD bullish alignment detected."})
            elif not bullish_5m and not bullish_15m:
                opportunities.append({"ticker": ticker,
                    "reason": "5m & 15m MACD bearish alignment detected."})
            elif not pd.isna(rsi_15m) and rsi_15m < 30:
                opportunities.append({"ticker": ticker,
                    "reason": f"15m RSI oversold ({rsi_15m:.1f}) - potential bounce."})
            elif not pd.isna(rsi_15m) and rsi_15m > 70:
                opportunities.append({"ticker": ticker,
                    "reason": f"15m RSI overbought ({rsi_15m:.1f}) - potential reversal."})
        except Exception:
            continue
    return opportunities

# ──────────────────────────────────────────────────────────────────────────────
# 11  SCREEN 1 - RADAR SCANNER
# ──────────────────────────────────────────────────────────────────────────────
def render_scanner_screen():
    st.title("Multi-Timeframe AI Scanner")
    st.caption("Scanning watchlist for live multi-timeframe signals.")
    st.markdown("---")
    display_cash_balance()
    with st.spinner("Scanning watchlist..."):
        opportunities = scan_for_opportunities()
    if not opportunities:
        st.info("No actionable signals detected right now - check back later.")
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
# 12  SCREEN 2 - TACTICAL COCKPIT WITH CONVERSATIONAL INTEL
# ──────────────────────────────────────────────────────────────────────────────
def render_tactical_screen(ticker):
    # Header
    col_back, col_refresh, _ = st.columns([1, 1, 6])
    with col_back:
        st.button("Back to Radar", on_click=go_back)
    with col_refresh:
        if st.button("Refresh Live Data"):
            st.rerun()
    display_cash_balance()

    selected_interval = st.selectbox("Chart Timeframe", ["1m", "5m", "15m", "1h", "1d"], index=0)

    # Load data & base AI scores
    with st.spinner(f"AI analysing {ticker} @ {selected_interval}..."):
        df = calculate_technicals(fetch_data(ticker, selected_interval))
        base_long, base_short, base_chop = generate_ai_probabilities(ticker)

    if df.empty:
        st.error("Market data unavailable for this ticker.")
        st.stop()

    latest = df.iloc[-1]
    price = float(latest["Close"])
    rsi = float(latest["RSI"]) if not pd.isna(latest["RSI"]) else 50.0
    atr = float(latest["ATR"]) if not pd.isna(latest["ATR"]) else 0.0

    # ── CONVERSATIONAL INTEL SECTION ─────────────────────────────────────
    st.markdown("---")
    st.subheader("AI Intel Briefing")
    st.caption("The AI has analysed the technicals. Now it needs YOUR intel to sharpen its read.")
    st.markdown("")

    # Display base technical read first
    tech_direction = "BULLISH" if base_long > base_short else "BEARISH" if base_short > base_long else "NEUTRAL"
    tech_emoji = "🟢" if tech_direction == "BULLISH" else "🔴" if tech_direction == "BEARISH" else "⚪"
    st.info(f"{tech_emoji} **Technical Base Read:** {tech_direction} | Long {base_long}% | Short {base_short}% | Chop {base_chop}%")

    # Pick which questions to ask based on current state
    question_keys = pick_questions(base_long, base_short, base_chop, rsi, atr, price)

    # Show questions in a form
    if not st.session_state.intel_submitted:
        st.markdown("**Answer these to refine the signal:**")
        with st.form("intel_form"):
            answers = {}
            for qkey in question_keys:
                q = INTEL_QUESTIONS[qkey]
                answers[qkey] = st.radio(
                    q["text"],
                    options=q["options"],
                    index=len(q["options"]) // 2,  # default to middle/neutral
                    key=f"q_{qkey}",
                    horizontal=True,
                )
            submitted = st.form_submit_button("Submit Intel", use_container_width=True, type="primary")
            if submitted:
                st.session_state.intel_answers = answers
                st.session_state.intel_submitted = True
                st.rerun()
    # ── BLENDED RESULTS (after intel submitted) ─────────────────────────
    if st.session_state.intel_submitted:
        intel_adj, breakdown = compute_intel_adjustment(st.session_state.intel_answers)
        final_long, final_short, final_chop = blend_scores(base_long, base_short, base_chop, intel_adj)
        risk_gate = apply_risk_filter(final_long, final_short, final_chop)

        # Narrative
        narrative = generate_ai_narrative(ticker, final_long, final_short, rsi, atr, intel_adj, breakdown)
        st.markdown("---")
        st.subheader("AI + Human Blended Verdict")
        st.markdown(narrative)

        # Metric strip
        m1, m2, m3 = st.columns(3)
        m1.metric("Live Price", f"${price:,.2f}")
        m2.metric(f"{selected_interval} RSI", f"{rsi:.1f}")
        m3.metric("Volatility (ATR)", f"${atr:,.2f}")

        # Chart + Meter side by side
        col_chart, col_data = st.columns([2, 1])
        with col_chart:
            st.plotly_chart(draw_annotated_chart(df, ticker, selected_interval), use_container_width=True)
        with col_data:
            st.subheader("Blended Confidence Meter")
            st.caption(f"Technical base + your intel (adj: {intel_adj:+d}pts)")
            st.progress(final_long / 100)
            st.markdown(f"**LONG:** {final_long}%")
            st.progress(final_short / 100)
            st.markdown(f"**SHORT:** {final_short}%")
            st.progress(final_chop / 100)
            st.markdown(f"**CHOP:** {final_chop}%")

            # Verdict
            if risk_gate["buy"]:
                st.success(f"AI + Human Verdict: **STRONG BUY** ({final_long}%)")
            elif risk_gate["sell"]:
                st.error(f"AI + Human Verdict: **STRONG SELL** ({final_short}%)")
            else:
                st.warning("AI + Human Verdict: **Mixed - consider staying flat.**")

        # Intel breakdown expander
        with st.expander("Intel Adjustment Breakdown"):
            for tag, pts in breakdown.items():
                color = "green" if pts > 0 else "red" if pts < 0 else "gray"
                st.markdown(f":{color}[**{tag.upper()}**: {pts:+d} pts]")
            st.markdown(f"**Total adjustment: {intel_adj:+d} pts** (capped at +/-30)")

        # Wait time estimates when below threshold
        if not risk_gate["buy"]:
            wait_buy = estimate_wait_time_for_signal(
                ticker, selected_interval, prob_cutoff, final_long,
                direction="buy", history_bars=6)
            st.info(f"Est. wait for BUY threshold: **{wait_buy}**")
        if not risk_gate["sell"]:
            wait_sell = estimate_wait_time_for_signal(
                ticker, selected_interval, prob_cutoff, final_short,
                direction="sell", history_bars=6)
            st.info(f"Est. wait for SELL threshold: **{wait_sell}**")
        # ── POSITION SIZING ──────────────────────────────────────────────
        st.markdown("---")
        st.subheader("Position Sizing (risk-adjusted)")
        acct = get_account_summary()
        shares = risk_adjusted_shares(acct["equity"], atr)
        sizing_tooltip = (
            f"Equity: ${acct['equity']:,.2f}\\n"
            f"Risk tolerance: {RISK_TOLERANCE.title()} (factor x{risk_factor})\\n"
            f"ATR x1.5 stop: ${atr*1.5:,.2f}\\n"
            f"Max shares: {shares}"
        )
        st.markdown(
            f'''<div style="font-size:1.1rem;">'''
            f'''<span title="{sizing_tooltip}" style="cursor:help;">'''
            f'''Calculated optimal sizing: **{shares} shares**</span></div>''',
            unsafe_allow_html=True
        )

        # Confirmation for live trading
        if not USE_PAPER:
            confirm = st.checkbox(
                "I understand this is a real-money market order and accept the risk.",
                value=False,
                help="Live orders cannot be undone."
            )
        else:
            confirm = True

        # ── BUY / SELL BUTTONS ───────────────────────────────────────────
        c_buy, c_sell = st.columns(2)
        with c_buy:
            buy_disabled = not risk_gate["buy"]
            buy_label = f"BUY {shares} SHARES" if not buy_disabled else f"BUY LOCKED ({final_long}% < {prob_cutoff}%)"
            if st.button(buy_label, use_container_width=True, type="primary", disabled=buy_disabled):
                if shares <= 0:
                    st.warning("Calculated share size is 0 - order not sent.")
                elif not confirm:
                    st.warning("Please tick the confirmation box first.")
                else:
                    try:
                        buy_order = MarketOrderRequest(
                            symbol=ticker, qty=shares,
                            side=OrderSide.BUY, time_in_force=TimeInForce.DAY
                        )
                        trading_client.submit_order(buy_order)
                        trail_percent = round(max(0.5, (atr / price) * 100), 2)
                        trailing_stop = TrailingStopOrderRequest(
                            symbol=ticker, qty=shares,
                            side=OrderSide.SELL, type="trailing_stop",
                            time_in_force=TimeInForce.GTC, trail_percent=trail_percent
                        )
                        trading_client.submit_order(trailing_stop)
                        st.success(f"BUY order for {shares} shares of {ticker} submitted. "
                                   f"Trailing-stop ({trail_percent}%) attached.")
                    except Exception as e:
                        st.error(f"Order failed: {e}")

        with c_sell:
            sell_disabled = not risk_gate["sell"]
            sell_label = f"SELL {shares} SHARES" if not sell_disabled else f"SELL LOCKED ({final_short}% < {prob_cutoff}%)"
            if st.button(sell_label, use_container_width=True, disabled=sell_disabled):
                if shares <= 0:
                    st.warning("Calculated share size is 0 - order not sent.")
                elif not confirm:
                    st.warning("Please tick the confirmation box first.")
                else:
                    try:
                        sell_order = MarketOrderRequest(
                            symbol=ticker, qty=shares,
                            side=OrderSide.SELL, time_in_force=TimeInForce.DAY
                        )
                        trading_client.submit_order(sell_order)
                        trail_percent = round(max(0.5, (atr / price) * 100), 2)
                        trailing_stop = TrailingStopOrderRequest(
                            symbol=ticker, qty=shares,
                            side=OrderSide.BUY, type="trailing_stop",
                            time_in_force=TimeInForce.GTC, trail_percent=trail_percent
                        )
                        trading_client.submit_order(trailing_stop)
                        st.success(f"SELL order for {shares} shares of {ticker} submitted. "
                                   f"Trailing-stop ({trail_percent}%) attached.")
                    except Exception as e:
                        st.error(f"Order failed: {e}")
        # ── RE-ANSWER BUTTON ─────────────────────────────────────────────
        st.markdown("---")
        if st.button("Re-answer Intel Questions", use_container_width=True):
            st.session_state.intel_submitted = False
            st.session_state.intel_answers = {}
            st.rerun()

    # ── CURRENT POSITIONS ────────────────────────────────────────────────
    st.markdown("---")
    positions = get_all_positions()
    if positions:
        st.subheader("Current Positions")
        pos_df = pd.DataFrame([{
            "Ticker": p.symbol,
            "Qty": int(p.qty),
            "Avg Entry": f"${float(p.avg_entry_price):,.2f}",
            "Current": f"${float(p.current_price):,.2f}",
            "Unrealised PnL": f"${float(p.unrealized_pl):,.2f}",
            "Side": "Long" if float(p.qty) > 0 else "Short"
        } for p in positions])
        st.dataframe(pos_df, hide_index=True, use_container_width=True)
    else:
        st.info("No open positions at the moment.")

# ──────────────────────────────────────────────────────────────────────────────
# 13  MAIN ROUTER
# ──────────────────────────────────────────────────────────────────────────────
if st.session_state.selected_ticker is None:
    render_scanner_screen()
else:
    render_tactical_screen(st.session_state.selected_ticker)
