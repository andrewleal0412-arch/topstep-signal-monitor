import streamlit as st
import streamlit.components.v1 as components
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import datetime, timedelta
from zoneinfo import ZoneInfo
import ta
import time
import json
import uuid
import os
import hashlib
import requests
import feedparser
import re
import html
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

_vader = SentimentIntensityAnalyzer()

PT = ZoneInfo("America/Los_Angeles")

# ─── Supabase client (works locally via .streamlit/secrets.toml and on cloud) ──
def _supa():
    from supabase import create_client
    url = st.secrets["supabase"]["url"]
    key = st.secrets["supabase"]["key"]
    return create_client(url, key)

# ─── Notification Config ──────────────────────────────────────────────────────
def load_config() -> dict:
    try:
        res = _supa().table("config").select("data").eq("id", "default").execute()
        if res.data:
            return res.data[0]["data"]
    except Exception:
        pass
    return {"ntfy_topic": "topstepdraco42", "notify_enabled": False, "min_score": 3.5}

def save_config(cfg: dict):
    try:
        _supa().table("config").upsert({"id": "default", "data": cfg}).execute()
    except Exception:
        pass

def send_notification(symbol: str, signal: dict, ti: dict):
    cfg = load_config()
    if not cfg.get("notify_enabled") or not cfg.get("ntfy_topic", "").strip():
        return

    topic  = cfg["ntfy_topic"].strip()
    score  = signal["score"]
    min_sc = cfg.get("min_score", 2.5)

    if abs(score) < min_sc:
        return

    d        = signal["direction"]
    name     = ti["name"]
    strength = int(abs(score) / 6.0 * 100)
    tick_sz  = ti["tick"]
    sl_ticks = abs(signal["entry"] - signal["sl"])  / tick_sz
    tp1_ticks= abs(signal["entry"] - signal["tp1"]) / tick_sz

    # No emoji in title — causes latin-1 encoding errors on some servers
    title = f"{d} - {name} | Score {score:+.1f} ({strength}% strength)"
    body  = (
        f"Entry:  {signal['entry']:,.2f}\n"
        f"Stop:   {signal['sl']:,.2f}  ({sl_ticks:.0f} ticks)\n"
        f"TP1:    {signal['tp1']:,.2f}  ({tp1_ticks:.0f} ticks)\n"
        f"TP2:    {signal['tp2']:,.2f}\n"
        f"Time:   {now_pt().strftime('%I:%M %p PT')}"
    )

    priority = "urgent" if abs(score) >= 4.5 else "high" if abs(score) >= 3.5 else "default"
    tags     = "chart_with_upwards_trend" if d == "LONG" else "chart_with_downwards_trend"

    try:
        requests.post(
            f"https://ntfy.sh/{topic}",
            data=body.encode("utf-8"),
            headers={
                "Title":    title.encode("utf-8"),
                "Priority": priority,
                "Tags":     tags,
            },
            timeout=5,
        )
    except Exception:
        pass  # never crash the app if notification fails

def now_pt() -> datetime:
    return datetime.now(PT)

# ─── News Engine ──────────────────────────────────────────────────────────────

NEWS_FEEDS = [
    ("Yahoo Finance",  "https://finance.yahoo.com/news/rssindex"),
    ("Reuters",        "https://feeds.reuters.com/reuters/businessNews"),
    ("CNBC",           "https://www.cnbc.com/id/100003114/device/rss/rss.html"),
    ("MarketWatch",    "https://feeds.content.dowjones.io/public/rss/mw_realtimeheadlines"),
    ("Investing.com",  "https://www.investing.com/rss/news.rss"),
]

# Keywords that make an article relevant to each instrument group
INSTRUMENT_KEYWORDS = {
    "nasdaq": ["nasdaq", "tech", "technology", "apple", "aapl", "nvidia", "nvda",
               "microsoft", "msft", "meta", "amazon", "amzn", "alphabet", "google",
               "growth stocks", "mnq", "nq", "qqq"],
    "sp500":  ["s&p", "sp500", "s&p 500", "dow", "russell", "equities", "stocks",
               "wall street", "mes", "es", "spy", "market rally", "market selloff"],
    "gold":   ["gold", "xau", "precious metal", "safe haven", "bullion",
               "gc", "mgc", "silver", "commodities"],
    "macro":  ["fed", "federal reserve", "fomc", "interest rate", "rate hike", "rate cut",
               "inflation", "cpi", "pce", "nfp", "jobs report", "gdp", "recession",
               "jerome powell", "treasury", "yield", "dollar", "dxy", "debt ceiling",
               "banking crisis", "tariff", "trade war", "geopolitical", "ukraine",
               "china", "earnings"],
}

# Symbol -> which keyword groups are relevant
SYMBOL_GROUPS = {
    "MNQ=F": ["nasdaq", "macro"],
    "NQ=F":  ["nasdaq", "macro"],
    "MES=F": ["sp500",  "macro"],
    "ES=F":  ["sp500",  "macro"],
    "GC=F":  ["gold",   "macro"],
    "MGC=F": ["gold",   "macro"],
}

# Economic calendar — recurring high-impact events (weekday 0=Mon, day_of_month)
ECON_EVENTS = [
    {"name": "CPI Report",          "desc": "Inflation data. Hot CPI = bearish stocks, bullish gold. Cold CPI = bullish stocks.", "impact": "HIGH"},
    {"name": "FOMC Meeting",         "desc": "Fed interest rate decision. Rate hikes hurt stocks & gold. Cuts help both.",         "impact": "HIGH"},
    {"name": "Non-Farm Payrolls",    "desc": "Jobs report (1st Friday of month). Strong jobs = bearish for rate cuts.",             "impact": "HIGH"},
    {"name": "GDP Report",           "desc": "Economic growth. Strong GDP = bullish stocks. Weak = bearish.",                      "impact": "HIGH"},
    {"name": "PCE Inflation",        "desc": "Fed's preferred inflation gauge. Similar impact to CPI.",                             "impact": "HIGH"},
    {"name": "Initial Jobless Claims","desc": "Weekly Thursday release. High claims = economic weakness.",                          "impact": "MED"},
    {"name": "PPI Report",           "desc": "Producer prices — leading indicator for CPI.",                                        "impact": "MED"},
    {"name": "Retail Sales",         "desc": "Consumer spending strength. Big miss = bearish.",                                     "impact": "MED"},
    {"name": "FOMC Minutes",         "desc": "Released 3 weeks after FOMC — shows Fed's internal debate.",                          "impact": "MED"},
]

@st.cache_data(ttl=300, show_spinner=False)  # refresh news every 5 minutes
def fetch_news() -> list:
    """Pull headlines from RSS feeds and score them with VADER sentiment."""
    articles = []
    seen = set()

    for source, url in NEWS_FEEDS:
        try:
            feed = feedparser.parse(url)
            for entry in feed.entries[:15]:
                title   = entry.get("title", "").strip()
                summary = entry.get("summary", entry.get("description", "")).strip()
                link    = entry.get("link", "")

                if not title or title in seen:
                    continue
                seen.add(title)

                # Strip HTML: two passes (some feeds double-encode tags)
                summary_clean = re.sub(r"<[^>]+>", " ", summary)
                summary_clean = html.unescape(summary_clean)
                summary_clean = re.sub(r"<[^>]+>", " ", summary_clean)  # second pass after unescape
                summary_clean = re.sub(r"\s+", " ", summary_clean).strip()[:300]

                text = f"{title}. {summary_clean}"

                # VADER sentiment
                vs   = _vader.polarity_scores(text)
                compound = vs["compound"]  # -1.0 (very bearish) to +1.0 (very bullish)

                # Determine which instruments this article is relevant to
                text_lower = text.lower()
                relevant_groups = []
                for group, kws in INSTRUMENT_KEYWORDS.items():
                    if any(kw in text_lower for kw in kws):
                        relevant_groups.append(group)

                # High-impact flag
                high_impact = any(kw in text_lower for kw in
                                  ["fed", "fomc", "cpi", "nfp", "gdp", "rate", "inflation", "recession",
                                   "tariff", "crash", "crisis", "emergency"])

                # Parse publish time
                try:
                    pub = datetime(*entry.published_parsed[:6], tzinfo=PT)
                    age_min = (now_pt() - pub).total_seconds() / 60
                except Exception:
                    pub = now_pt()
                    age_min = 999

                articles.append({
                    "title":     title,
                    "summary":   summary_clean,
                    "source":    source,
                    "link":      link,
                    "compound":  compound,
                    "pos":       vs["pos"],
                    "neg":       vs["neg"],
                    "groups":    relevant_groups,
                    "high_impact": high_impact,
                    "pub":       pub,
                    "age_min":   age_min,
                })
        except Exception:
            continue

    # Sort by time, newest first
    articles.sort(key=lambda x: x["age_min"])
    return articles[:60]

def get_news_sentiment(symbol: str, articles: list) -> dict:
    """
    Aggregate news sentiment for a given symbol.
    Returns a dict with score (-1 to +1), label, articles list, and signal adjustment.
    """
    groups = SYMBOL_GROUPS.get(symbol, ["macro"])
    relevant = [a for a in articles if any(g in a["groups"] for g in groups)]

    if not relevant:
        return {"score": 0.0, "label": "Neutral", "color": "#8e8e93",
                "adjustment": 0.0, "articles": [], "count": 0}

    # Weight recent articles more heavily, high-impact articles 2x
    weights, scores = [], []
    for a in relevant[:20]:
        age_weight   = max(0.1, 1.0 - a["age_min"] / 240)  # decay over 4 hours
        impact_weight = 2.0 if a["high_impact"] else 1.0
        w = age_weight * impact_weight
        weights.append(w)
        scores.append(a["compound"])

    if not weights:
        return {"score": 0.0, "label": "Neutral", "color": "#8e8e93",
                "adjustment": 0.0, "articles": [], "count": 0}

    score = sum(s * w for s, w in zip(scores, weights)) / sum(weights)
    score = max(-1.0, min(1.0, score))

    # Map to label and signal adjustment
    if score >= 0.25:
        label, color = "Bullish", "#30d158"
    elif score <= -0.25:
        label, color = "Bearish", "#ff375f"
    else:
        label, color = "Neutral", "#8e8e93"

    # Signal adjustment: max ±1.5 points added to the technical score
    adjustment = round(score * 1.5, 2)

    return {
        "score":      round(score, 3),
        "label":      label,
        "color":      color,
        "adjustment": adjustment,
        "articles":   relevant[:10],
        "count":      len(relevant),
    }

def sentiment_label(compound: float) -> tuple:
    """Return (label, color, emoji) for a compound VADER score."""
    if compound >= 0.35:
        return "Good for market",  "#30d158", "🟢"
    elif compound >= 0.10:
        return "Slightly positive", "#34c759", "🟡"
    elif compound <= -0.35:
        return "Bad for market",   "#ff375f", "🔴"
    elif compound <= -0.10:
        return "Slightly negative", "#ff6b6b", "🟠"
    else:
        return "Mixed / Neutral",  "#8e8e93", "⚪"

st.set_page_config(
    page_title="PaperTrail",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# ─── CSS ─────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&display=swap');

*, *::before, *::after { box-sizing: border-box; margin: 0; padding: 0; }

/* ── Base — deep navy background ── */
html, body, [class*="css"], .stApp {
    font-family: 'Inter', -apple-system, BlinkMacSystemFont, 'SF Pro Text', sans-serif !important;
    background-color: #080c14 !important;
    color: #e2e8f0 !important;
    font-size: 15px !important;
}

/* ── Page background ── */
.stApp { background: #080c14 !important; }
section.main > div { background: #080c14 !important; }

/* ── Streamlit tab overrides ── */
.stTabs [data-baseweb="tab-list"] {
    background: transparent !important;
    border-bottom: 1px solid rgba(255,255,255,0.06) !important;
    gap: 4px;
}
.stTabs [data-baseweb="tab"] {
    background: transparent !important;
    color: #64748b !important;
    font-family: 'Inter', sans-serif !important;
    font-size: 15px !important;
    font-weight: 500 !important;
    padding: 10px 18px !important;
    border-radius: 8px 8px 0 0 !important;
    border: none !important;
    letter-spacing: 0.01em;
}
.stTabs [aria-selected="true"] {
    background: rgba(99,102,241,0.1) !important;
    color: #818cf8 !important;
    border-bottom: 2px solid #818cf8 !important;
}
.stTabs [data-baseweb="tab-panel"] { padding-top: 20px !important; }

/* ── Selectbox / inputs ── */
.stSelectbox > div > div {
    background: #0f1520 !important;
    border: 1px solid rgba(255,255,255,0.08) !important;
    border-radius: 8px !important;
    color: #e2e8f0 !important;
}
.stToggle label { color: #94a3b8 !important; }

/* ── Metric cards ── */
.metrics-grid {
    display: grid;
    grid-template-columns: repeat(5, 1fr);
    gap: 10px;
    margin-bottom: 20px;
}
.mc {
    background: #0f1520;
    border: 1px solid rgba(255,255,255,0.06);
    border-radius: 12px;
    padding: 14px 16px;
    min-height: 80px;
    display: flex;
    flex-direction: column;
    justify-content: space-between;
    position: relative;
    overflow: visible;
}
.mc::before {
    content: '';
    position: absolute;
    top: 0; left: 0; right: 0;
    height: 1px;
    background: linear-gradient(90deg, transparent, rgba(129,140,248,0.3), transparent);
}
.mc-label {
    font-size: 10px;
    font-weight: 600;
    letter-spacing: 0.08em;
    text-transform: uppercase;
    color: #475569;
}
.mc-value {
    font-size: 1.4em;
    font-weight: 700;
    color: #f1f5f9;
    letter-spacing: -0.02em;
    line-height: 1;
    margin: 6px 0 4px;
}
.mc-delta { font-size: 11px; font-weight: 500; }
.pos { color: #34d399; }
.neg { color: #f87171; }
.neu { color: #475569; }
.clr-pos { color: #34d399; }
.clr-neg { color: #f87171; }

/* ── Signal banner ── */
.sig-banner {
    border-radius: 16px;
    padding: 20px 26px;
    display: flex;
    align-items: center;
    gap: 20px;
    margin-bottom: 16px;
    position: relative;
    overflow: hidden;
}
.sig-banner-long {
    background: linear-gradient(135deg, #052e16 0%, #0a1f12 100%);
    border: 1px solid rgba(52,211,153,0.25);
    box-shadow: 0 0 40px rgba(52,211,153,0.08), inset 0 1px 0 rgba(52,211,153,0.1);
}
.sig-banner-short {
    background: linear-gradient(135deg, #2d0a0a 0%, #1f0a0d 100%);
    border: 1px solid rgba(248,113,113,0.25);
    box-shadow: 0 0 40px rgba(248,113,113,0.08), inset 0 1px 0 rgba(248,113,113,0.1);
}
.sig-banner-neu {
    background: #0f1520;
    border: 1px solid rgba(255,255,255,0.06);
}
.sig-icon   { font-size: 2.2em; line-height: 1; flex-shrink: 0; }
.sig-center { flex: 1; }
.sig-dir    { font-size: 1.7em; font-weight: 800; letter-spacing: 0.02em; line-height: 1; }
.sig-sub    { font-size: 12px; color: #64748b; margin-top: 5px; font-weight: 400; }
.sig-right  { text-align: right; flex-shrink: 0; }
.sig-score-num { font-size: 1.9em; font-weight: 800; letter-spacing: -0.03em; line-height: 1; }
.sig-score-lbl { font-size: 11px; color: #475569; margin-top: 2px; font-weight: 500; letter-spacing: 0.04em; text-transform: uppercase; }

/* strength bar */
.strength-bar-wrap {
    background: rgba(255,255,255,0.06);
    border-radius: 99px;
    height: 4px;
    margin-top: 12px;
    overflow: hidden;
    width: 100%;
    max-width: 260px;
}
.strength-bar-fill { height: 100%; border-radius: 99px; }

/* ── Two-column panel ── */
.panel-grid {
    display: grid;
    grid-template-columns: 1fr 1fr;
    gap: 12px;
    margin-bottom: 16px;
}
.panel-card {
    background: #0f1520;
    border: 1px solid rgba(255,255,255,0.06);
    border-radius: 14px;
    padding: 18px 20px;
}
.panel-title {
    font-size: 10px;
    font-weight: 700;
    letter-spacing: 0.1em;
    text-transform: uppercase;
    color: #475569;
    margin-bottom: 14px;
}

/* ── Reason items ── */
.reason-item {
    display: flex;
    align-items: center;
    gap: 10px;
    padding: 9px 12px;
    margin: 4px 0;
    background: rgba(255,255,255,0.03);
    border-radius: 8px;
    font-size: 15px;
    font-weight: 400;
    color: #cbd5e1;
    border-left: 2px solid transparent;
}
.reason-bull { border-left-color: #34d399; }
.reason-bear { border-left-color: #f87171; }

/* ── Trade levels table ── */
.tl-table { width: 100%; border-collapse: collapse; }
.tl-table tr { border-bottom: 1px solid rgba(255,255,255,0.04); }
.tl-table tr:last-child { border-bottom: none; }
.tl-table td { padding: 9px 8px; font-size: 15px; vertical-align: middle; }
.tl-label { color: #475569; font-weight: 500; width: 50px; }
.tl-price { font-variant-numeric: tabular-nums; font-weight: 600; font-size: 15px; color: #f1f5f9; }
.tl-meta  { color: #475569; font-size: 11px; text-align: right; }
.mono { font-family: 'SF Mono', 'Fira Code', 'Cascadia Code', monospace; }

/* ── Stats pills ── */
.stats-row { display: flex; gap: 8px; flex-wrap: wrap; margin-bottom: 14px; }
.stat-pill {
    background: rgba(129,140,248,0.08);
    border: 1px solid rgba(129,140,248,0.15);
    border-radius: 99px;
    padding: 4px 12px;
    font-size: 12px;
    font-weight: 500;
    color: #94a3b8;
    white-space: nowrap;
}
.stat-pill span { color: #475569; margin-right: 4px; }

/* ── Trade history table ── */
.th-table { width: 100%; border-collapse: collapse; font-size: 12px; }
.th-table th {
    font-size: 10px; font-weight: 700; letter-spacing: 0.08em;
    text-transform: uppercase; color: #475569;
    padding: 10px 12px; text-align: left;
    border-bottom: 1px solid rgba(255,255,255,0.06);
    background: #0a0f1a;
}
.th-table td { padding: 11px 12px; border-bottom: 1px solid rgba(255,255,255,0.03); color: #cbd5e1; }
.th-table tr:hover td { background: rgba(129,140,248,0.04); }

/* ── Badges ── */
.badge {
    display: inline-block; padding: 3px 10px;
    border-radius: 99px; font-size: 10px;
    font-weight: 700; letter-spacing: 0.05em;
    text-transform: uppercase; white-space: nowrap;
}
.badge-win2  { background: rgba(52,211,153,0.12);  color: #34d399; border: 1px solid rgba(52,211,153,0.2); }
.badge-win1  { background: rgba(52,211,153,0.08);  color: #34d399; border: 1px solid rgba(52,211,153,0.15); }
.badge-loss  { background: rgba(248,113,113,0.12);  color: #f87171; border: 1px solid rgba(248,113,113,0.2); }
.badge-open  { background: rgba(251,191,36,0.1);   color: #fbbf24; border: 1px solid rgba(251,191,36,0.2); }
.badge-long  { background: rgba(48,209,88,0.15);   color: #30d158; border: 1px solid rgba(48,209,88,0.35); }
.badge-short { background: rgba(255,55,95,0.15);   color: #ff375f; border: 1px solid rgba(255,55,95,0.35); }

/* ── Tooltips ── */
.tt {
    position: relative;
    display: inline-flex;
    align-items: center;
    gap: 4px;
    cursor: help;
}
.tt-label {
    border-bottom: 1px dashed rgba(255,255,255,0.25);
    line-height: 1.2;
}
.tt .tt-box {
    visibility: hidden; opacity: 0;
    background: #1e2a3a; color: #e2e8f0;
    border: 1px solid rgba(255,255,255,0.1);
    border-radius: 10px; padding: 10px 14px;
    position: absolute; bottom: calc(100% + 8px);
    left: 0; transform: none;
    width: 250px; font-size: 12px; line-height: 1.6;
    z-index: 9999; white-space: normal;
    box-shadow: 0 20px 40px rgba(0,0,0,0.6);
    transition: opacity 0.15s ease; pointer-events: none;
    font-weight: 400;
}
.tt:hover .tt-box { visibility: visible; opacity: 1; }
.tt-icon {
    display: inline-flex; align-items: center; justify-content: center;
    width: 14px; height: 14px; border-radius: 50%; flex-shrink: 0;
    background: rgba(129,140,248,0.15); border: 1px solid rgba(129,140,248,0.25);
    font-size: 9px; font-weight: 700; color: #818cf8; cursor: help;
}

/* ── Dividers ── */
hr { border-color: rgba(255,255,255,0.15) !important; border-width: 2px !important; }

/* ── Expanders ── */
.streamlit-expanderHeader {
    background: #0f1520 !important;
    border: 1px solid rgba(255,255,255,0.06) !important;
    border-radius: 10px !important;
    color: #94a3b8 !important;
    font-size: 15px !important;
    font-weight: 500 !important;
}
.streamlit-expanderContent {
    background: #0a0f1a !important;
    border: 1px solid rgba(255,255,255,0.05) !important;
    border-top: none !important;
    border-radius: 0 0 10px 10px !important;
}

/* ── Buttons ── */
.stButton > button {
    background: rgba(129,140,248,0.1) !important;
    border: 1px solid rgba(129,140,248,0.25) !important;
    color: #818cf8 !important;
    border-radius: 8px !important;
    font-family: 'Inter', sans-serif !important;
    font-weight: 600 !important;
    font-size: 15px !important;
    letter-spacing: 0.01em;
}
.stButton > button:hover {
    background: rgba(129,140,248,0.18) !important;
    border-color: rgba(129,140,248,0.4) !important;
}

/* ── Progress bars ── */
.stProgress > div > div {
    background: rgba(255,255,255,0.06) !important;
    border-radius: 99px !important;
}
.stProgress > div > div > div {
    background: linear-gradient(90deg, #6366f1, #818cf8) !important;
    border-radius: 99px !important;
}

/* ── Info / warning / error boxes ── */
.stAlert { border-radius: 10px !important; border-left-width: 3px !important; }

/* ── Hide Streamlit chrome ── */
#MainMenu, footer, header { visibility: hidden; }
section[data-testid="stSidebar"] { display: none !important; }
.block-container { padding-top: 1.5rem !important; padding-bottom: 2rem !important; }
div[data-testid="stVerticalBlock"] > div { gap: 0rem; }

/* ── Prevent page dimming on auto-refresh / cache rerun ── */
[data-stale="true"] { opacity: 1 !important; }
.stApp [data-stale="true"] * { opacity: 1 !important; }
</style>
""", unsafe_allow_html=True)

# ─── Glossary ─────────────────────────────────────────────────────────────────
GLOSSARY = {
    "EMA":        "Exponential Moving Average — a weighted average of recent prices. Reacts faster than a simple average. Used to identify trend direction.",
    "EMA9":       "9-period EMA — fastest line, average of last 9 candles. Crossing above EMA21 = bullish sign.",
    "EMA21":      "21-period EMA — medium-speed average of last 21 candles. Used with EMA9 to spot trend changes.",
    "EMA50":      "50-period EMA — slow average of last 50 candles. A major support/resistance level. Price above = bullish.",
    "RSI":        "Relative Strength Index (0–100). Above 70 = overbought (may drop). Below 30 = oversold (may bounce). 50 is neutral.",
    "MACD":       "Moving Average Convergence Divergence — compares two EMAs to measure momentum. MACD above signal line = bullish.",
    "VWAP":       "Volume Weighted Average Price — average price by trading volume. Institutions use it as a key benchmark. Above VWAP = buyers in control.",
    "ATR":        "Average True Range — how many points price moves on average per candle. Used to set stop and target distances.",
    "BB":         "Bollinger Bands — two bands around price showing normal range. Price at lower band = oversold zone, upper band = overbought zone.",
    "MNQ":        "Micro E-mini Nasdaq-100 — tracks top 100 Nasdaq stocks. Each 0.25pt tick = $0.50. Good for beginners.",
    "NQ":         "E-mini Nasdaq-100 — same as MNQ but 10x larger. Each 0.25pt tick = $5.00.",
    "MES":        "Micro E-mini S&P 500 — tracks the S&P 500 (500 biggest US companies). Each 0.25pt tick = $1.25. Good starter size.",
    "ES":         "E-mini S&P 500 — standard S&P 500 futures. Each 0.25pt tick = $12.50. Much larger than MES.",
    "GC":         "Gold Futures (COMEX) — tracks gold price (100 troy oz). Each 0.10pt tick = $10.00.",
    "MGC":        "Micro Gold Futures — 1/10th size of GC. Each 0.10pt tick = $1.00.",
    "SL":         "Stop Loss — price where your trade auto-closes to cap your loss. Always set one before entering.",
    "TP1":        "Take Profit 1 — first exit target. Close 60% of position here and move stop to break-even.",
    "TP2":        "Take Profit 2 — second, bigger target. Let remaining 40% ride here.",
    "R:R":        "Risk-to-Reward Ratio. 1:2 means risking $1 to make $2. Always target 1:1.5 or better.",
    "Tick":       "Smallest possible price movement. MNQ tick = 0.25pts = $0.50. MES tick = 0.25pts = $1.25.",
    "Long":       "Buying — you profit when price goes UP.",
    "Short":      "Selling (shorting) — you profit when price goes DOWN.",
    "Bullish":    "Expecting price to move higher.",
    "Bearish":    "Expecting price to move lower.",
    "Crossover":  "When one line crosses another. Fast line above slow line = bullish crossover.",
    "Overbought": "Price rose too fast, may pull back. RSI above 70 signals this.",
    "Oversold":   "Price fell too fast, may bounce. RSI below 30 signals this.",
    "Drawdown":   "Drop from account peak. The platform ends your eval if this exceeds the limit.",
    "Daily Loss": "Max you can lose in one trading day. Hit this = stop trading today.",
    "Contract":   "One unit of futures. 1 MNQ = $0.50/tick. 1 MES = $1.25/tick.",
    "Eval":       "Funded account challenge — trade profitably within rules to earn a real funded account.",
    "Scaling":    "Adding contracts to a winning position to increase profit.",
    "Momentum":   "Speed and strength of a price move. MACD and RSI both measure it.",
    "Support":    "Price level where buyers appear and stop price falling. Acts like a floor.",
    "Resistance": "Price level where sellers appear and stop price rising. Acts like a ceiling.",
}

def tip(label: str, key: str = None) -> str:
    k = key or label
    defn = GLOSSARY.get(k, GLOSSARY.get(k.upper(), ""))
    if not defn:
        return label
    return (f'<span class="tt">'
            f'<span class="tt-label">{label}</span>'
            f'<span class="tt-icon">?</span>'
            f'<span class="tt-box">{defn}</span>'
            f'</span>')

# ─── Trading Session Gate ─────────────────────────────────────────────────────
def trading_session_active(symbol: str) -> tuple:
    """Return (is_active: bool, reason: str, next_window: str)."""
    now     = now_pt()
    weekday = now.weekday()          # 0=Mon … 6=Sun
    h       = now.hour + now.minute / 60.0  # fractional hour in PT

    # Saturday — fully closed
    if weekday == 5:
        return False, "Markets closed — Saturday", "Sunday 3:00 PM PT"

    # Sunday before futures reopen (6 PM ET = 3 PM PT)
    if weekday == 6 and h < 15.0:
        return False, "Markets closed — reopens Sunday 3:00 PM PT", "3:00 PM PT today"

    if symbol in ("MNQ=F", "MES=F"):
        if 6.5 <= h < 8.5:
            return True,  "Market open session (6:30–8:30 AM PT)", ""
        if 11.0 <= h < 13.0:
            return True,  "Power hour / close session (11:00 AM–1:00 PM PT)", ""
        if 8.5 <= h < 11.0:
            return False, "Midday chop — signals paused", "11:00 AM PT"
        if 13.0 <= h:
            return False, "After-hours — signals paused", "6:30 AM PT tomorrow"
        # Overnight (h < 6.5)
        return False, "Pre-market — signals paused", "6:30 AM PT"

    if symbol == "MGC=F":
        if h < 2.0:
            return True,  "London session (12:00–2:00 AM PT)", ""
        if 5.0 <= h < 9.0:
            return True,  "COMEX / NY open session (5:00–9:00 AM PT)", ""
        if 2.0 <= h < 5.0:
            return False, "Between London & COMEX — signals paused", "5:00 AM PT"
        # h >= 9.0
        return False, "Outside gold sessions — signals paused", "12:00 AM PT"

    return True, "", ""

def trade_in_session(symbol: str, timestamp_str: str) -> bool:
    """Return True if the trade timestamp falls within an active session for the symbol."""
    try:
        ts = datetime.fromisoformat(timestamp_str).astimezone(PT)
    except Exception:
        return False
    weekday = ts.weekday()
    h = ts.hour + ts.minute / 60.0
    if weekday == 5:
        return False
    if weekday == 6 and h < 15.0:
        return False
    if symbol in ("MNQ=F", "MES=F"):
        return (6.5 <= h < 8.5) or (11.0 <= h < 13.0)
    if symbol == "MGC=F":
        return (h < 2.0) or (5.0 <= h < 9.0)
    return True

# ─── Constants ────────────────────────────────────────────────────────────────
TOPSTEP_ACCOUNTS = {
    "$50K":  {"target": 3000,  "daily_loss": 1500, "max_dd": 2500,  "contract_limit": 5},
    "$100K": {"target": 6000,  "daily_loss": 2500, "max_dd": 3000,  "contract_limit": 10},
    "$150K": {"target": 9000,  "daily_loss": 3500, "max_dd": 4500,  "contract_limit": 15},
}

TICK_INFO = {
    "MNQ=F": {"tick": 0.25, "value": 0.50,  "name": "MNQ"},
    "MES=F": {"tick": 0.25, "value": 1.25,  "name": "MES"},
    "MGC=F": {"tick": 0.10, "value": 1.00,  "name": "MGC"},
}

# Fallback info for old/removed symbols still in the DB
_SYMBOL_ALIAS = {
    "NQ=F":  {"tick": 0.25, "value": 5.00,  "name": "NQ"},
    "ES=F":  {"tick": 0.25, "value": 12.50, "name": "ES"},
    "GC=F":  {"tick": 0.10, "value": 10.00, "name": "GC"},
}

def _ti(symbol: str) -> dict:
    """Return tick info for a symbol, falling back to old-symbol aliases."""
    return TICK_INFO.get(symbol) or _SYMBOL_ALIAS.get(symbol) or {"tick": 0.25, "value": 0, "name": symbol}

# ─── Trade Persistence ────────────────────────────────────────────────────────
def load_trades() -> list:
    try:
        res = _supa().table("trades").select("data").order("created_at").execute()
        if res.data:
            return [row["data"] for row in res.data]
        return []
    except Exception as e:
        st.session_state["_db_error"] = str(e)
        return []

def save_trades(trades: list):
    try:
        db = _supa()
        # Upsert each trade individually — safe, never loses data on partial failure
        rows = [{"id": t["id"], "data": t} for t in trades[-200:]]
        if rows:
            db.table("trades").upsert(rows).execute()
    except Exception:
        pass

def save_single_trade(trade: dict):
    """Insert or update one trade — used when recording a new signal."""
    try:
        _supa().table("trades").upsert({"id": trade["id"], "data": trade}).execute()
    except Exception:
        pass

# Minimum score magnitude required to record a signal per symbol
# Based on trade log analysis: low-score signals have poor win rates
_MIN_RECORD_SCORE = {
    "MNQ=F": 3.5,   # off-hours noise heavily impacted MNQ; higher bar needed
    "MES=F": 2.5,   # 60% WR overall, insufficient data to raise yet
    "MGC=F": 3.0,   # both MGC losses were score 2.60–2.71; 3.0+ = all wins
}

# Symbols that trade 24h and should NOT be blocked by the session gate
# Gold is a global market — off-hours MGC signals had 75% win rate vs 67% in-session
_SESSION_GATE_EXEMPT = {"MGC=F"}

def should_record_signal(signal: dict, symbol: str) -> bool:
    """Only record a new signal if:
    - Direction is not NEUTRAL
    - Score clears the per-symbol minimum threshold
    - Currently in an active trading session for this symbol
    - No open trade already exists for this symbol (max 1 per symbol)
    - Direction changed from last recorded trade
    - At least 15 min since last trade on this symbol
    """
    if signal["direction"] == "NEUTRAL":
        return False

    # Per-symbol score gate — filters out weak signals that historically lose
    min_score = _MIN_RECORD_SCORE.get(symbol, 2.5)
    if abs(signal.get("score", 0)) < min_score:
        return False

    # Session gate — block off-hours signals for equity index futures (MNQ/MES)
    # MGC is exempt: gold trades 24h and off-hours signals have shown 75% win rate
    if symbol not in _SESSION_GATE_EXEMPT and not trading_session_active(symbol)[0]:
        return False

    trades = load_trades()
    sym_trades = [t for t in trades if t["symbol"] == symbol]

    # Block if there's already an open trade on this symbol
    if any(t["status"] == "open" for t in sym_trades):
        return False

    if not sym_trades:
        return True
    last = sym_trades[-1]
    # Cooldown: 30 min minimum between signals (prevents duplicate entries same setup)
    try:
        last_time = datetime.fromisoformat(last["timestamp"])
        if last_time.tzinfo is None:
            last_time = last_time.replace(tzinfo=PT)
        if (now_pt() - last_time).total_seconds() < 1800:
            return False
    except Exception:
        pass
    return True

def record_signal(signal: dict, symbol: str, interval: str) -> dict:
    trade = {
        "id":        str(uuid.uuid4())[:8],
        "symbol":    symbol,
        "name":      TICK_INFO[symbol]["name"],
        "interval":  interval,
        "direction": signal["direction"],
        "entry":     float(signal["entry"]),
        "sl":        float(signal["sl"]),
        "tp1":       float(signal["tp1"]),
        "tp2":       float(signal["tp2"]),
        "score":     signal["score"],
        "reasons":   signal["reasons"],
        "timestamp": now_pt().isoformat(),
        "status":    "open",
        "closed_at": None,
        "pnl_ticks": None,
    }
    save_single_trade(trade)
    # Only notify during active trading sessions — always log regardless
    if trading_session_active(symbol)[0]:
        send_notification(symbol, trade, TICK_INFO[symbol])
    return trade

_MAX_PERIOD = {"1m": "7d", "2m": "60d", "5m": "60d", "15m": "60d", "30m": "60d", "1h": "730d"}

def check_open_trades(symbol: str, df: pd.DataFrame) -> list:
    """Walk recent candles and close any open trades that hit SL/TP.
    Fetches max-allowed history so old open trades are never stuck unresolved."""
    trades = load_trades()
    open_trades = [t for t in trades if t["status"] == "open" and t["symbol"] == symbol]
    if not open_trades or df.empty:
        return trades

    # Detect the interval from the df index spacing, then fetch max history
    try:
        spacing_min = (df.index[-1] - df.index[-2]).total_seconds() / 60
        if spacing_min <= 1:    iv = "1m"
        elif spacing_min <= 2:  iv = "2m"
        elif spacing_min <= 5:  iv = "5m"
        elif spacing_min <= 15: iv = "15m"
        elif spacing_min <= 30: iv = "30m"
        else:                   iv = "1h"
        max_period = _MAX_PERIOD.get(iv, "60d")
        ext = _fetch_raw(symbol, iv, max_period)  # bypass cache — must have latest candles
        if not ext.empty and len(ext) > len(df):
            df = ext
    except Exception:
        pass  # fall back to the df already passed in

    changed = False
    for trade in trades:
        if trade["status"] != "open" or trade["symbol"] != symbol:
            continue

        try:
            sig_time = datetime.fromisoformat(trade["timestamp"])
            if sig_time.tzinfo is None:
                sig_time = sig_time.replace(tzinfo=PT)
            if df.index.tz is not None:
                # df index is tz-aware — align sig_time to same tz
                sig_time = sig_time.astimezone(df.index.tz)
                after = df[df.index > sig_time]
            else:
                # df index is naive UTC — convert sig_time to UTC then strip tz
                from zoneinfo import ZoneInfo as _ZI
                sig_time_utc = sig_time.astimezone(_ZI("UTC")).replace(tzinfo=None)
                after = df[df.index > sig_time_utc]
        except Exception:
            continue

        if after.empty:
            continue

        ti   = TICK_INFO[symbol]["tick"]
        d    = trade["direction"]
        sl   = trade["sl"]
        tp1  = trade["tp1"]
        tp2  = trade["tp2"]
        entry = trade["entry"]

        for idx, candle in after.iterrows():
            hi = float(candle["High"])
            lo = float(candle["Low"])

            if d == "LONG":
                sl_hit  = lo <= sl
                tp2_hit = hi >= tp2
                tp1_hit = hi >= tp1
                if tp2_hit:
                    trade.update(status="win_tp2", pnl_ticks=round(abs(tp2 - entry) / ti, 1), closed_at=idx.isoformat())
                    changed = True; break
                elif tp1_hit and sl_hit:
                    # Both TP1 and SL tagged same candle — price likely ran to TP1 first
                    trade.update(status="win_tp1", pnl_ticks=round(abs(tp1 - entry) / ti, 1), closed_at=idx.isoformat())
                    changed = True; break
                elif tp1_hit:
                    trade.update(status="win_tp1", pnl_ticks=round(abs(tp1 - entry) / ti, 1), closed_at=idx.isoformat())
                    changed = True; break
                elif sl_hit:
                    trade.update(status="loss", pnl_ticks=round(-abs(entry - sl) / ti, 1), closed_at=idx.isoformat())
                    changed = True; break
            else:  # SHORT
                sl_hit  = hi >= sl
                tp2_hit = lo <= tp2
                tp1_hit = lo <= tp1
                if tp2_hit:
                    trade.update(status="win_tp2", pnl_ticks=round(abs(entry - tp2) / ti, 1), closed_at=idx.isoformat())
                    changed = True; break
                elif tp1_hit and sl_hit:
                    # Both TP1 and SL tagged same candle — price likely ran to TP1 first
                    trade.update(status="win_tp1", pnl_ticks=round(abs(entry - tp1) / ti, 1), closed_at=idx.isoformat())
                    changed = True; break
                elif tp1_hit:
                    trade.update(status="win_tp1", pnl_ticks=round(abs(entry - tp1) / ti, 1), closed_at=idx.isoformat())
                    changed = True; break
                elif sl_hit:
                    trade.update(status="loss", pnl_ticks=round(-abs(sl - entry) / ti, 1), closed_at=idx.isoformat())
                    changed = True; break

    if changed:
        # Only upsert the trades that actually changed — safer than bulk delete/reinsert
        for trade in trades:
            if trade["status"] != "open" and trade["symbol"] == symbol:
                save_single_trade(trade)
    return trades

def get_stats(trades: list, symbol: str = None) -> dict:
    pool = [t for t in trades if t["status"] != "open"]
    if symbol:
        pool = [t for t in pool if t["symbol"] == symbol]
    if not pool:
        return {"total": 0, "wins": 0, "losses": 0, "win_rate": 0, "total_ticks": 0, "open": 0}
    wins   = [t for t in pool if t["status"].startswith("win")]
    losses = [t for t in pool if t["status"] == "loss"]
    ticks  = sum(t.get("pnl_ticks", 0) or 0 for t in pool)
    open_c = len([t for t in trades if t["status"] == "open" and (not symbol or t["symbol"] == symbol)])
    return {
        "total":      len(pool),
        "wins":       len(wins),
        "losses":     len(losses),
        "win_rate":   round(len(wins) / len(pool) * 100, 1) if pool else 0,
        "total_ticks": round(ticks, 1),
        "open":       open_c,
    }

def get_adaptive_note(trades: list, symbol: str) -> str:
    """Return a short insight based on recent trade history."""
    recent = [t for t in trades if t["symbol"] == symbol and t["status"] != "open"][-10:]
    if len(recent) < 3:
        return ""
    wins = [t for t in recent if t["status"].startswith("win")]
    rate = len(wins) / len(recent)
    if rate >= 0.7:
        return f"🔥 Last {len(recent)} trades: {rate*100:.0f}% win rate — signals are working well on this instrument."
    elif rate <= 0.35:
        return f"⚠️ Last {len(recent)} trades: {rate*100:.0f}% win rate — be selective, current conditions may not suit these signals."
    return f"Last {len(recent)} trades: {rate*100:.0f}% win rate."

# ─── Data ─────────────────────────────────────────────────────────────────────
# Polygon.io real-time futures mapping
_POLYGON_MAP = {
    "MNQ=F": "MNQ",
    "MES=F": "MES",
    "MGC=F": "MGC",
}
_POLYGON_INTERVAL = {
    "1m":  (1,  "minute"), "2m":  (2,  "minute"), "5m":  (5,  "minute"),
    "15m": (15, "minute"), "30m": (30, "minute"), "1h":  (1,  "hour"),
}
_PERIOD_DAYS = {
    "7d": 7, "60d": 60, "730d": 365,
}

def _fetch_polygon(symbol: str, interval: str, period: str) -> pd.DataFrame:
    """Fetch real-time bars from Polygon.io. Returns empty df if key not set."""
    try:
        api_key = st.secrets.get("polygon", {}).get("api_key", "")
        if not api_key:
            return pd.DataFrame()
        ticker = _POLYGON_MAP.get(symbol)
        if not ticker:
            return pd.DataFrame()
        mult, timespan = _POLYGON_INTERVAL.get(interval, (5, "minute"))
        days = _PERIOD_DAYS.get(period, 7)
        end_dt   = datetime.now(ZoneInfo("UTC"))
        start_dt = end_dt - timedelta(days=days)
        url = (f"https://api.polygon.io/v2/aggs/ticker/{ticker}/range/"
               f"{mult}/{timespan}/"
               f"{start_dt.strftime('%Y-%m-%d')}/{end_dt.strftime('%Y-%m-%d')}")
        resp = requests.get(url,
                            params={"apiKey": api_key, "limit": 50000, "sort": "asc"},
                            timeout=10)
        data = resp.json()
        if data.get("status") not in ("OK", "DELAYED") or not data.get("results"):
            return pd.DataFrame()
        rows, idx = [], []
        for bar in data["results"]:
            idx.append(datetime.fromtimestamp(bar["t"] / 1000, tz=ZoneInfo("UTC")))
            rows.append({"Open": bar["o"], "High": bar["h"],
                         "Low":  bar["l"], "Close": bar["c"],
                         "Volume": bar.get("v", 0)})
        df = pd.DataFrame(rows, index=pd.DatetimeIndex(idx))
        return df.dropna()
    except Exception:
        return pd.DataFrame()

def _fetch_raw(symbol: str, interval: str, period: str) -> pd.DataFrame:
    """Try Polygon.io first (real-time). Fall back to yfinance (15-min delayed)."""
    df = _fetch_polygon(symbol, interval, period)
    if not df.empty:
        return df
    try:
        df = yf.download(symbol, period=period, interval=interval, progress=False, auto_adjust=True)
        if df.empty:
            return pd.DataFrame()
        df.columns = [c[0] if isinstance(c, tuple) else c for c in df.columns]
        return df.dropna()
    except Exception:
        return pd.DataFrame()

@st.cache_data(ttl=15, show_spinner=False)
def fetch_data(symbol: str, interval: str, period: str) -> pd.DataFrame:
    return _fetch_raw(symbol, interval, period)

def _snap(price: float, tick: float) -> float:
    """Snap a price to the nearest valid tick increment."""
    return round(round(price / tick) * tick, 10)

# ─── Indicators ───────────────────────────────────────────────────────────────
def compute_indicators(df: pd.DataFrame) -> pd.DataFrame:
    if len(df) < 50:
        return df
    close, high, low = df["Close"], df["High"], df["Low"]
    df["EMA9"]  = ta.trend.ema_indicator(close, window=9)
    df["EMA21"] = ta.trend.ema_indicator(close, window=21)
    df["EMA50"] = ta.trend.ema_indicator(close, window=50)
    df["RSI"]   = ta.momentum.rsi(close, window=14)
    macd = ta.trend.MACD(close, window_slow=26, window_fast=12, window_sign=9)
    df["MACD"] = macd.macd(); df["MACD_signal"] = macd.macd_signal(); df["MACD_hist"] = macd.macd_diff()
    bb = ta.volatility.BollingerBands(close, window=20, window_dev=2)
    df["BB_upper"] = bb.bollinger_hband(); df["BB_lower"] = bb.bollinger_lband(); df["BB_mid"] = bb.bollinger_mavg()
    df["ATR"]  = ta.volatility.average_true_range(high, low, close, window=14)
    df["VWAP"] = (close * df["Volume"]).cumsum() / df["Volume"].cumsum()
    return df

# ─── Higher Timeframe Bias ────────────────────────────────────────────────────
def get_htf_bias(df: pd.DataFrame) -> int:
    """
    Reads 15m candles and returns +1 (bullish), -1 (bearish), or 0 (neutral).
    Used to confirm or filter 5m entry signals.
    """
    if len(df) < 50:
        return 0
    last = df.iloc[-1]
    bias = 0.0
    # EMA stack
    if   last["EMA9"] > last["EMA21"] > last["EMA50"]: bias += 2
    elif last["EMA9"] < last["EMA21"] < last["EMA50"]: bias -= 2
    elif last["EMA9"] > last["EMA21"]:                  bias += 1
    else:                                               bias -= 1
    # MACD
    if last["MACD"] > last["MACD_signal"]: bias += 1
    else:                                  bias -= 1
    # RSI
    rsi = float(last["RSI"])
    if   rsi > 55: bias += 0.5
    elif rsi < 45: bias -= 0.5
    # VWAP
    if float(last["Close"]) > float(last["VWAP"]): bias += 0.5
    else:                                           bias -= 0.5

    if   bias >= 2.0: return  1
    elif bias <= -2.0: return -1
    return 0

# ─── Signal Engine ────────────────────────────────────────────────────────────
def generate_signal(df: pd.DataFrame, symbol: str = None, news_sentiment: dict = None, htf_bias_15m: int = 0, htf_bias_1h: int = 0) -> dict:
    empty = {"direction": "NEUTRAL", "score": 0, "reasons": [],
             "entry": None, "sl": None, "tp1": None, "tp2": None, "atr": 0,
             "price": 0, "rsi": 50, "ema9": 0, "ema21": 0, "ema50": 0, "vwap": 0,
             "news_adjustment": 0.0}
    if len(df) < 50:
        return empty

    last, prev = df.iloc[-1], df.iloc[-2]
    score, reasons = 0, []

    # EMA stack
    if last["EMA9"] > last["EMA21"] > last["EMA50"]:
        score += 2; reasons.append(("bull", f"✅ {tip('EMA','EMA')} stack bullish (9 &gt; 21 &gt; 50)"))
    elif last["EMA9"] < last["EMA21"] < last["EMA50"]:
        score -= 2; reasons.append(("bear", f"🔻 {tip('EMA','EMA')} stack bearish (9 &lt; 21 &lt; 50)"))
    elif last["EMA9"] > last["EMA21"]:
        score += 1; reasons.append(("bull", f"↑ {tip('EMA9','EMA9')} above {tip('EMA21','EMA21')}"))
    else:
        score -= 1; reasons.append(("bear", f"↓ {tip('EMA9','EMA9')} below {tip('EMA21','EMA21')}"))

    # Fresh crossover
    if prev["EMA9"] <= prev["EMA21"] and last["EMA9"] > last["EMA21"]:
        score += 1; reasons.append(("bull", f"⚡ Fresh bullish {tip('crossover','Crossover')} (EMA9 crossed above EMA21)"))
    elif prev["EMA9"] >= prev["EMA21"] and last["EMA9"] < last["EMA21"]:
        score -= 1; reasons.append(("bear", f"⚡ Fresh bearish {tip('crossover','Crossover')} (EMA9 crossed below EMA21)"))

    # RSI
    rsi = float(last["RSI"])
    if rsi < 35:
        score += 1; reasons.append(("bull", f"📉 {tip('RSI','RSI')} {rsi:.1f} — {tip('oversold','Oversold')} (potential bounce)"))
    elif rsi > 65:
        score -= 1; reasons.append(("bear", f"📈 {tip('RSI','RSI')} {rsi:.1f} — {tip('overbought','Overbought')} (potential pullback)"))
    elif 48 < rsi < 62:
        score += 0.5; reasons.append(("bull", f"{tip('RSI','RSI')} {rsi:.1f} — in bullish zone"))
    elif 38 < rsi < 52:
        score -= 0.5

    # MACD
    if last["MACD"] > last["MACD_signal"] and float(last["MACD_hist"]) > float(prev["MACD_hist"]):
        score += 1; reasons.append(("bull", f"✅ {tip('MACD','MACD')} bullish & momentum expanding"))
    elif last["MACD"] < last["MACD_signal"] and float(last["MACD_hist"]) < float(prev["MACD_hist"]):
        score -= 1; reasons.append(("bear", f"🔻 {tip('MACD','MACD')} bearish & momentum dropping"))

    # VWAP
    if float(last["Close"]) > float(last["VWAP"]):
        score += 0.5; reasons.append(("bull", f"Above {tip('VWAP','VWAP')} — buyers in control"))
    else:
        score -= 0.5; reasons.append(("bear", f"Below {tip('VWAP','VWAP')} — sellers in control"))

    # Bollinger Bands
    if float(last["Close"]) < float(last["BB_lower"]):
        score += 0.5; reasons.append(("bull", f"At {tip('BB','BB')} lower band — potential bounce zone"))
    elif float(last["Close"]) > float(last["BB_upper"]):
        score -= 0.5; reasons.append(("bear", f"At {tip('BB','BB')} upper band — potential reversal zone"))

    # ── News sentiment adjustment ─────────────────────────────────────────────
    news_adj = 0.0
    if news_sentiment and news_sentiment.get("count", 0) > 0:
        news_adj  = news_sentiment["adjustment"]
        score    += news_adj
        ns_label  = news_sentiment["label"]
        ns_score  = news_sentiment["score"]
        ns_count  = news_sentiment["count"]
        if news_adj > 0.3:
            reasons.append(("bull", f"📰 News is positive right now — pushing signal up by +{news_adj:.1f} ({ns_count} articles)"))
        elif news_adj < -0.3:
            reasons.append(("bear", f"📰 News is negative right now — pulling signal down by {news_adj:.1f} ({ns_count} articles)"))
        else:
            reasons.append(("bull" if news_adj >= 0 else "bear",
                            f"📰 News is mixed — small influence on signal ({ns_count} articles, {ns_score:+.2f})"))

    # ── 1h trend filter (macro — highest weight) ─────────────────────────────
    if htf_bias_1h == 1:
        if score > 0:
            score += 1.5
            reasons.append(("bull", "✅ 1h trend confirms bullish — strong macro alignment"))
        elif score < 0:
            score += 2.0
            reasons.append(("bear", "⚠ 1h trend is bullish — counter-trend SHORT heavily penalized"))
    elif htf_bias_1h == -1:
        if score < 0:
            score -= 1.5
            reasons.append(("bear", "✅ 1h trend confirms bearish — strong macro alignment"))
        elif score > 0:
            score -= 2.0
            reasons.append(("bull", "⚠ 1h trend is bearish — counter-trend LONG heavily penalized"))

    # ── 15m trend filter (intermediate) ──────────────────────────────────────
    if htf_bias_15m == 1:
        if score > 0:
            score += 1.0
            reasons.append(("bull", "✅ 15m trend confirms bullish — aligned LONG signal"))
        elif score < 0:
            score += 1.5
            reasons.append(("bear", "⚠ 15m trend is bullish — counter-trend SHORT weakened"))
    elif htf_bias_15m == -1:
        if score < 0:
            score -= 1.0
            reasons.append(("bear", "✅ 15m trend confirms bearish — aligned SHORT signal"))
        elif score > 0:
            score -= 1.5
            reasons.append(("bull", "⚠ 15m trend is bearish — counter-trend LONG weakened"))

    score = round(score, 1)
    direction = "LONG" if score >= 2.5 else "SHORT" if score <= -2.5 else "NEUTRAL"

    atr   = float(last["ATR"])
    price = float(last["Close"])

    # Adaptive SL multiplier — learned from false stops per symbol
    sl_mult = 1.5
    if symbol:
        try:
            sl_mult = float(load_config().get("sl_multipliers", {}).get(symbol, 1.5))
            sl_mult = max(1.0, min(2.5, sl_mult))
        except Exception:
            pass

    tick = TICK_INFO.get(symbol, {}).get("tick", 0.25) if symbol else 0.25
    if direction == "LONG":
        entry = _snap(price, tick)
        sl    = _snap(price - sl_mult * atr, tick)
        tp1   = _snap(price + sl_mult * atr, tick)
        tp2   = _snap(price + sl_mult * 2 * atr, tick)
    elif direction == "SHORT":
        entry = _snap(price, tick)
        sl    = _snap(price + sl_mult * atr, tick)
        tp1   = _snap(price - sl_mult * atr, tick)
        tp2   = _snap(price - sl_mult * 2 * atr, tick)
    else:
        entry = sl = tp1 = tp2 = None

    return {
        "direction": direction, "score": score, "reasons": reasons,
        "entry": entry, "sl": sl, "tp1": tp1, "tp2": tp2,
        "atr": atr, "price": price, "rsi": rsi,
        "ema9":  float(last["EMA9"]),
        "ema21": float(last["EMA21"]),
        "ema50": float(last["EMA50"]),
        "vwap":  float(last["VWAP"]),
        "news_adjustment": news_adj,
        "sl_mult": sl_mult,
    }

# ─── Chart ────────────────────────────────────────────────────────────────────
_CHART_CFG = {
    "displayModeBar": True,
    "displaylogo": False,
    "scrollZoom": False,
    "modeBarButtonsToRemove": ["lasso2d", "select2d", "toggleSpikelines"],
}

def _to_pt(df: pd.DataFrame) -> pd.DataFrame:
    """Convert DataFrame index to Pacific Time for correct chart timestamps."""
    df = df.copy()
    if df.index.tz is None:
        df.index = df.index.tz_localize("UTC")
    df.index = df.index.tz_convert("America/Los_Angeles")
    return df

def _chart_layout(height: int, title: str, yaxis_range=None, rangeslider=False) -> dict:
    yaxis = dict(gridcolor="rgba(255,255,255,0.04)", zeroline=False,
                 tickfont=dict(size=11), side="right", fixedrange=False)
    if yaxis_range:
        yaxis["range"] = yaxis_range
    xaxis = dict(
        gridcolor="rgba(255,255,255,0.04)",
        showspikes=True, spikecolor="rgba(255,255,255,0.18)",
        spikethickness=1, spikedash="dot",
        rangeslider=dict(visible=rangeslider, thickness=0.04,
                         bgcolor="#0d1420", bordercolor="#1e293b", borderwidth=1),
    )
    return dict(
        template="plotly_dark", paper_bgcolor="#0a0f1a", plot_bgcolor="#0a0f1a",
        height=height,
        margin=dict(l=10, r=120, t=36, b=10),
        font=dict(size=11, color="#94a3b8"),
        hovermode="x unified",
        dragmode="zoom",
        xaxis=xaxis,
        yaxis=yaxis,
        legend=dict(
            x=0.01, y=0.99, xanchor="left", yanchor="top",
            orientation="h",
            bgcolor="rgba(10,15,26,0.75)",
            bordercolor="rgba(255,255,255,0.08)", borderwidth=1,
            font=dict(size=10, color="#94a3b8"),
        ),
        title=dict(text=title, font=dict(size=11, color="#475569"), x=0),
    )

def _hline_annotation(fig, y, color, dash, width, label):
    fig.add_hline(
        y=y, line_color=color, line_dash=dash, line_width=width,
        annotation_text=f"<b>{label}  {y:,.2f}</b>",
        annotation_position="right",
        annotation_font_color=color,
        annotation_font_size=11,
        annotation_bgcolor="rgba(10,15,26,0.9)",
        annotation_bordercolor=color,
        annotation_borderwidth=1,
        annotation_borderpad=3,
    )

def build_price_chart(df: pd.DataFrame, signal: dict, open_trade: dict = None) -> go.Figure:
    plot_df = _to_pt(df.tail(120))
    fig = go.Figure()

    fig.add_trace(go.Candlestick(
        x=plot_df.index,
        open=plot_df["Open"], high=plot_df["High"],
        low=plot_df["Low"],   close=plot_df["Close"],
        name="Price",
        increasing_line_color="#26a65b", decreasing_line_color="#e83030",
        increasing_fillcolor="#26a65b",  decreasing_fillcolor="#e83030",
        line=dict(width=1),
    ))
    fig.add_trace(go.Scatter(x=plot_df.index, y=plot_df["EMA9"],  name="EMA9",  line=dict(color="#f0c040", width=1.2)))
    fig.add_trace(go.Scatter(x=plot_df.index, y=plot_df["EMA21"], name="EMA21", line=dict(color="#f08030", width=1.2)))
    fig.add_trace(go.Scatter(x=plot_df.index, y=plot_df["EMA50"], name="EMA50", line=dict(color="#e05020", width=1.2)))
    fig.add_trace(go.Scatter(x=plot_df.index, y=plot_df["VWAP"],  name="VWAP",  line=dict(color="#818cf8", width=1.2, dash="dot")))
    fig.add_trace(go.Scatter(x=plot_df.index, y=plot_df["BB_upper"],
                             line=dict(color="rgba(160,160,160,0.25)", width=1), showlegend=False, name="BB"))
    fig.add_trace(go.Scatter(x=plot_df.index, y=plot_df["BB_lower"],
                             line=dict(color="rgba(160,160,160,0.25)", width=1),
                             fill="tonexty", fillcolor="rgba(160,160,160,0.04)", showlegend=False, name="BB Lo"))

    lvls = open_trade if open_trade else (signal if signal.get("entry") is not None else None)
    if lvls:
        is_long     = (lvls.get("direction", signal.get("direction", "LONG")) == "LONG")
        entry_color = "#26a65b" if is_long else "#e83030"
        _hline_annotation(fig, lvls["entry"], entry_color, "solid", 2.0, "ENTRY")
        _hline_annotation(fig, lvls["sl"],    "#e83030",   "dash",  1.5, "SL   ")
        _hline_annotation(fig, lvls["tp1"],   "#26a65b",   "dash",  1.5, "TP1  ")
        _hline_annotation(fig, lvls["tp2"],   "#34d399",   "dot",   1.2, "TP2  ")

    fig.update_layout(**_chart_layout(560, "Price  ·  EMA 9/21/50  ·  VWAP  ·  BB", rangeslider=True))
    return fig

def build_rsi_chart(df: pd.DataFrame) -> go.Figure:
    plot_df = _to_pt(df.tail(120))
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=plot_df.index, y=plot_df["RSI"], name="RSI",
                             line=dict(color="#c084fc", width=1.5), fill="tozeroy",
                             fillcolor="rgba(192,132,252,0.06)"))
    for y, clr in [(70, "rgba(232,48,48,0.35)"), (30, "rgba(38,166,91,0.35)"), (50, "rgba(255,255,255,0.08)")]:
        fig.add_hline(y=y, line_color=clr, line_dash="dot", line_width=1)
    fig.update_layout(**_chart_layout(220, "RSI  (14)", yaxis_range=[0, 100]))
    return fig

def build_macd_chart(df: pd.DataFrame) -> go.Figure:
    plot_df = _to_pt(df.tail(120))
    fig = go.Figure()
    colors_hist = ["#26a65b" if v >= 0 else "#e83030" for v in plot_df["MACD_hist"]]
    fig.add_trace(go.Bar(x=plot_df.index, y=plot_df["MACD_hist"], name="Histogram",
                         marker_color=colors_hist, opacity=0.65))
    fig.add_trace(go.Scatter(x=plot_df.index, y=plot_df["MACD"],        name="MACD",   line=dict(color="#f0c040", width=1.2)))
    fig.add_trace(go.Scatter(x=plot_df.index, y=plot_df["MACD_signal"], name="Signal", line=dict(color="#f08030", width=1.2)))
    fig.update_layout(**_chart_layout(220, "MACD  (12 / 26 / 9)"))
    return fig

# ─── Sidebar ─────────────────────────────────────────────────────────────────
def render_sidebar():
    st.sidebar.title("Eval Tracker")
    acct = st.sidebar.selectbox("Account Size", list(TOPSTEP_ACCOUNTS.keys()), key="acct")
    rules = TOPSTEP_ACCOUNTS[acct]

    st.sidebar.divider()
    st.sidebar.markdown("**Enter your P&L**")
    daily_pnl = st.sidebar.number_input("Today's P&L ($)", value=0.0, step=50.0, key="daily_pnl")
    total_pnl = st.sidebar.number_input("Total P&L ($)",   value=0.0, step=100.0, key="total_pnl")

    daily_rem = rules["daily_loss"] + daily_pnl
    dd_used   = max(0.0, -total_pnl)

    st.sidebar.divider()
    st.sidebar.markdown("**Eval Progress**")

    pct = lambda v, mx: max(0.0, min(1.0, v / mx))
    st.sidebar.progress(pct(daily_rem, rules["daily_loss"]),
                        text=f"Daily loss room: ${daily_rem:,.0f} / ${rules['daily_loss']:,.0f}")
    st.sidebar.caption("How much you can still lose today")

    st.sidebar.progress(pct(max(0, total_pnl), rules["target"]),
                        text=f"Profit target: ${total_pnl:,.0f} / ${rules['target']:,.0f}")
    st.sidebar.caption("Reach this to pass the eval")

    st.sidebar.progress(pct(dd_used, rules["max_dd"]),
                        text=f"Max drawdown: ${dd_used:,.0f} / ${rules['max_dd']:,.0f}")
    st.sidebar.caption("If account drops this far, eval ends")

    st.sidebar.divider()
    if daily_pnl <= -rules["daily_loss"]:
        st.sidebar.error("STOP — Daily loss limit hit!")
    elif dd_used >= rules["max_dd"]:
        st.sidebar.error("ACCOUNT BUSTED — Max drawdown hit!")
    elif total_pnl >= rules["target"]:
        st.sidebar.success("EVAL PASSED! Contact your platform.")
    elif daily_rem < rules["daily_loss"] * 0.25:
        st.sidebar.warning("Low daily loss buffer — trade small")
    else:
        st.sidebar.info(f"Active — max {rules['contract_limit']} contracts")

    st.sidebar.divider()
    # Overall signal stats in sidebar
    all_trades = load_trades()
    stats = get_stats(all_trades)
    if stats["total"] > 0:
        st.sidebar.markdown("**Signal Track Record (All)**")
        wr_color = "🟢" if stats["win_rate"] >= 55 else ("🟡" if stats["win_rate"] >= 40 else "🔴")
        st.sidebar.markdown(f"{wr_color} **{stats['win_rate']}% win rate** — {stats['wins']}W / {stats['losses']}L")
        tick_color = "clr-pos" if stats["total_ticks"] >= 0 else "clr-neg"
        st.sidebar.markdown(f"Total ticks: `{stats['total_ticks']:+.1f}` &nbsp;|&nbsp; Open: `{stats['open']}`",
                            unsafe_allow_html=True)

    return acct, rules

# ─── Instrument Panel ─────────────────────────────────────────────────────────
def render_instrument(symbol: str, interval: str, period: str):
    ti = TICK_INFO[symbol]
    name = ti["name"]

    df = fetch_data(symbol, interval, period)
    if df.empty:
        st.error(f"No data for {symbol}. Market may be closed.")
        return

    df = compute_indicators(df)

    # Higher timeframe bias — 15m (intermediate) + 1h (macro)
    df_15m        = fetch_data(symbol, "15m", "60d")
    df_15m        = compute_indicators(df_15m) if not df_15m.empty else df_15m
    htf_bias_15m  = get_htf_bias(df_15m) if not df_15m.empty else 0

    df_1h         = fetch_data(symbol, "1h", "730d")
    df_1h         = compute_indicators(df_1h) if not df_1h.empty else df_1h
    htf_bias_1h   = get_htf_bias(df_1h) if not df_1h.empty else 0

    articles        = fetch_news()
    news_sentiment  = get_news_sentiment(symbol, articles)
    signal          = generate_signal(df, symbol, news_sentiment, htf_bias_15m=htf_bias_15m, htf_bias_1h=htf_bias_1h)

    # ── check / update open trades ──
    trades = check_open_trades(symbol, df)

    # ── record new signal (deduplication via JSON file, not session state) ──
    if should_record_signal(signal, symbol):
        record_signal(signal, symbol, interval)  # record_signal handles the notification

    last = df.iloc[-1]
    prev = df.iloc[-2] if len(df) > 1 else last
    chg  = float(last["Close"]) - float(prev["Close"])
    pct  = chg / float(prev["Close"]) * 100
    rsi_val = signal["rsi"]
    rsi_lbl = "Overbought" if rsi_val > 65 else ("Oversold" if rsi_val < 35 else "Neutral")
    rsi_cls = "neg" if rsi_val > 65 else ("pos" if rsi_val < 35 else "neu")

    # ── Metric grid (CSS grid — no Streamlit columns so widths are always equal) ──
    price_cls = "pos" if chg >= 0 else "neg"
    st.markdown(f"""
<div class="metrics-grid">
  <div class="mc">
    <div class="mc-label">{tip(name, name)}</div>
    <div class="mc-value">{float(last['Close']):,.2f}</div>
    <div class="mc-delta {price_cls}">{chg:+.2f} &nbsp;({pct:+.2f}%)</div>
  </div>
  <div class="mc">
    <div class="mc-label">{tip('RSI','RSI')}</div>
    <div class="mc-value">{rsi_val:.1f}</div>
    <div class="mc-delta {rsi_cls}">{rsi_lbl}</div>
  </div>
  <div class="mc">
    <div class="mc-label">{tip('EMA 9','EMA9')}</div>
    <div class="mc-value">{signal['ema9']:,.2f}</div>
    <div class="mc-delta neu">9-bar avg</div>
  </div>
  <div class="mc">
    <div class="mc-label">{tip('EMA 21','EMA21')}</div>
    <div class="mc-value">{signal['ema21']:,.2f}</div>
    <div class="mc-delta neu">21-bar avg</div>
  </div>
  <div class="mc">
    <div class="mc-label">{tip('ATR','ATR')}</div>
    <div class="mc-value">{signal['atr']:.2f}</div>
    <div class="mc-delta neu">volatility</div>
  </div>
</div>
""", unsafe_allow_html=True)

    # ── Session status banner ────────────────────────────────────────────────
    sess_active, sess_reason, sess_next = trading_session_active(symbol)
    if symbol in _SESSION_GATE_EXEMPT:
        # MGC trades 24h — always show as active, no paused state
        st.markdown("""
<div style="background:rgba(52,211,153,0.06);border:1px solid rgba(52,211,153,0.2);
     border-radius:10px;padding:9px 16px;margin-bottom:12px;
     display:flex;align-items:center;gap:10px;font-family:'Inter',sans-serif">
  <span style="width:8px;height:8px;border-radius:50%;background:#34d399;flex-shrink:0;
       box-shadow:0 0 6px rgba(52,211,153,0.6)"></span>
  <span style="font-size:12px;font-weight:600;color:#34d399">ACTIVE 24H</span>
  <span style="font-size:12px;color:#64748b">— Gold trades around the clock</span>
</div>""", unsafe_allow_html=True)
    elif sess_active:
        st.markdown(f"""
<div style="background:rgba(52,211,153,0.06);border:1px solid rgba(52,211,153,0.2);
     border-radius:10px;padding:9px 16px;margin-bottom:12px;
     display:flex;align-items:center;gap:10px;font-family:'Inter',sans-serif">
  <span style="width:8px;height:8px;border-radius:50%;background:#34d399;flex-shrink:0;
       box-shadow:0 0 6px rgba(52,211,153,0.6)"></span>
  <span style="font-size:12px;font-weight:600;color:#34d399">ACTIVE SESSION</span>
  <span style="font-size:12px;color:#64748b">— {sess_reason}</span>
</div>""", unsafe_allow_html=True)
    else:
        next_str = f" &nbsp;·&nbsp; Next window: <b style='color:#94a3b8'>{sess_next}</b>" if sess_next else ""
        st.markdown(f"""
<div style="background:rgba(251,191,36,0.05);border:1px solid rgba(251,191,36,0.18);
     border-radius:10px;padding:9px 16px;margin-bottom:12px;
     display:flex;align-items:center;gap:10px;font-family:'Inter',sans-serif">
  <span style="width:8px;height:8px;border-radius:50%;background:#fbbf24;flex-shrink:0"></span>
  <span style="font-size:12px;font-weight:600;color:#fbbf24">SIGNALS PAUSED</span>
  <span style="font-size:12px;color:#64748b">— {sess_reason}{next_str}</span>
</div>""", unsafe_allow_html=True)

    # ── Signal banner (full-width) ────────────────────────────────────────────
    d        = signal["direction"]
    score    = signal["score"]
    strength = abs(score) / 6.0 * 100

    if d == "LONG":
        banner_cls, icon, dir_color, dir_html = "sig-banner-long",  "📈", "#30d158", tip("LONG","Long")
        subtitle = "Conditions favor a buy — price likely moving up"
    elif d == "SHORT":
        banner_cls, icon, dir_color, dir_html = "sig-banner-short", "📉", "#ff375f", tip("SHORT","Short")
        subtitle = "Conditions favor a sell — price likely moving down"
    else:
        banner_cls, icon, dir_color, dir_html = "sig-banner-neu",   "◯",  "#8e8e93", "WAITING"
        subtitle = "No clear setup yet — stay out until signal confirms"

    bar_color = "#30d158" if d == "LONG" else ("#ff375f" if d == "SHORT" else "#48484a")

    st.markdown(f"""
<div class="sig-banner {banner_cls}">
  <div class="sig-icon">{icon}</div>
  <div class="sig-center">
    <div class="sig-dir" style="color:{dir_color}">{dir_html}</div>
    <div class="sig-sub">{subtitle}</div>
    <div class="strength-bar-wrap">
      <div class="strength-bar-fill" style="width:{strength:.0f}%;background:{bar_color}"></div>
    </div>
  </div>
  <div class="sig-right">
    <div class="sig-score-num" style="color:{dir_color}">{score:+.1f}</div>
    <div class="sig-score-lbl">score / 6.0</div>
    <div class="sig-score-lbl" style="margin-top:4px">{strength:.0f}% strength</div>
  </div>
</div>
""", unsafe_allow_html=True)

    # ── Two-panel row: Reasons | Trade Levels ────────────────────────────────
    reasons_html = ""
    for polarity, text in signal["reasons"]:
        cls = "reason-bull" if polarity == "bull" else "reason-bear"
        reasons_html += f'<div class="reason-item {cls}">{text}</div>'
    if not reasons_html:
        reasons_html = '<div class="reason-item">Not enough candle data yet</div>'

    # ── Use open trade levels if one exists (keeps tab consistent with log) ──
    all_trades   = load_trades()
    open_trade   = next((t for t in all_trades if t["symbol"] == symbol and t["status"] == "open"), None)
    display_lvls = None
    if open_trade:
        display_lvls = {
            "entry": open_trade["entry"],
            "sl":    open_trade["sl"],
            "tp1":   open_trade["tp1"],
            "tp2":   open_trade["tp2"],
        }
    elif signal["entry"] is not None:
        display_lvls = {
            "entry": signal["entry"],
            "sl":    signal["sl"],
            "tp1":   signal["tp1"],
            "tp2":   signal["tp2"],
        }

    if display_lvls:
        tick_sz   = ti["tick"]
        tick_val  = ti["value"]
        sl_ticks  = abs(display_lvls["entry"] - display_lvls["sl"])  / tick_sz
        tp1_ticks = abs(display_lvls["entry"] - display_lvls["tp1"]) / tick_sz
        tp2_ticks = abs(display_lvls["entry"] - display_lvls["tp2"]) / tick_sz
        locked_note = (' <span style="font-size:10px;color:#fbbf24;font-weight:600">● LIVE TRADE</span>' if open_trade else "")
        levels_html = f"""
<table class="tl-table">
  <tr>
    <td class="tl-label">Entry{locked_note}</td>
    <td class="tl-price mono">{display_lvls['entry']:,.2f}</td>
    <td class="tl-meta">—</td>
  </tr>
  <tr>
    <td class="tl-label" style="color:#ff375f">{tip('Stop','SL')}</td>
    <td class="tl-price mono" style="color:#ff375f">{display_lvls['sl']:,.2f}</td>
    <td class="tl-meta" style="color:#ff375f">{sl_ticks:.0f} {tip('ticks','Tick')} &nbsp;· &nbsp;${sl_ticks*tick_val:,.0f}</td>
  </tr>
  <tr>
    <td class="tl-label" style="color:#30d158">{tip('TP1','TP1')}</td>
    <td class="tl-price mono" style="color:#30d158">{display_lvls['tp1']:,.2f}</td>
    <td class="tl-meta" style="color:#30d158">{tp1_ticks:.0f} ticks &nbsp;· &nbsp;${tp1_ticks*tick_val:,.0f}</td>
  </tr>
  <tr>
    <td class="tl-label" style="color:#34c759">{tip('TP2','TP2')}</td>
    <td class="tl-price mono" style="color:#34c759">{display_lvls['tp2']:,.2f}</td>
    <td class="tl-meta" style="color:#34c759">{tp2_ticks:.0f} ticks &nbsp;· &nbsp;${tp2_ticks*tick_val:,.0f}</td>
  </tr>
  <tr>
    <td class="tl-label" style="color:#8e8e93">{tip('R:R','R:R')}</td>
    <td colspan="2" class="tl-price" style="color:#8e8e93">1:1 to TP1 &nbsp;/&nbsp; 1:2 to TP2</td>
  </tr>
</table>"""
        levels_panel = f'<div class="panel-card"><div class="panel-title">Trade Levels</div>{levels_html}</div>'
    else:
        levels_panel = f'<div class="panel-card"><div class="panel-title">Trade Levels</div><div style="color:#8e8e93;font-size:13px;padding-top:8px">No active signal — levels appear when direction confirms.</div></div>'

    st.markdown(f"""
<div class="panel-grid">
  <div class="panel-card">
    <div class="panel-title">Why This Signal</div>
    {reasons_html}
  </div>
  {levels_panel}
</div>
""", unsafe_allow_html=True)

    # ── Charts ────────────────────────────────────────────────────────────────
    st.markdown("<div style='height:8px'></div>", unsafe_allow_html=True)
    st.plotly_chart(build_price_chart(df, signal, open_trade=open_trade),
                    use_container_width=True, config=_CHART_CFG)
    st.plotly_chart(build_rsi_chart(df),  use_container_width=True, config=_CHART_CFG)
    st.plotly_chart(build_macd_chart(df), use_container_width=True, config=_CHART_CFG)


    # ── Row 5: Trade History ──────────────────────────────────────────────────
    sym_trades = [t for t in trades if t["symbol"] == symbol]
    stats = get_stats(trades, symbol)
    note  = get_adaptive_note(trades, symbol)

    with st.expander(f"Signal Track Record — {name}  ({stats['wins']}W / {stats['losses']}L / {stats['open']} open)", expanded=False):
        if stats["total"] > 0 or stats["open"] > 0:
            wr_icon = "🟢" if stats["win_rate"] >= 55 else ("🟡" if stats["win_rate"] >= 40 else "🔴")
            ticks_cls = "pos" if stats["total_ticks"] >= 0 else "neg"
            st.markdown(f"""
<div class="stats-row">
  <div class="stat-pill"><span>Win Rate</span><b>{wr_icon} {stats['win_rate']}%</b></div>
  <div class="stat-pill"><span>Closed</span><b>{stats['total']}</b></div>
  <div class="stat-pill"><span>Wins</span><b class="pos">{stats['wins']}</b></div>
  <div class="stat-pill"><span>Losses</span><b class="neg">{stats['losses']}</b></div>
  <div class="stat-pill"><span>{tip('Ticks','Tick')}</span><b class="{ticks_cls}">{stats['total_ticks']:+.1f}</b></div>
  <div class="stat-pill"><span>Open</span><b style="color:#ffd60a">{stats['open']}</b></div>
</div>""", unsafe_allow_html=True)

            if note:
                st.info(note)

            # Table — most recent first
            rows = ""
            for t in reversed(sym_trades[-30:]):
                s = t["status"]
                if s == "win_tp2":
                    badge = '<span class="badge badge-win2">WIN TP2</span>'
                elif s == "win_tp1":
                    badge = '<span class="badge badge-win1">WIN TP1</span>'
                elif s == "loss":
                    badge = '<span class="badge badge-loss">LOSS</span>'
                else:
                    badge = '<span class="badge badge-open">OPEN</span>'

                dir_badge = (f'<span class="badge badge-long">LONG</span>'
                             if t["direction"] == "LONG"
                             else '<span class="badge badge-short">SHORT</span>')
                ticks_str = f'{t["pnl_ticks"]:+.1f}' if t.get("pnl_ticks") is not None else "—"
                ticks_color_inline = "#00cc66" if (t.get("pnl_ticks") or 0) > 0 else ("#ff5050" if (t.get("pnl_ticks") or 0) < 0 else "#888")

                try:
                    ts = datetime.fromisoformat(t["timestamp"]).astimezone(PT).strftime("%m/%d %I:%M %p")
                except Exception:
                    ts = t["timestamp"][:16]

                rows += f"""
<tr>
  <td style="color:#888">{ts} PT</td>
  <td>{dir_badge}</td>
  <td class="mono">{t['entry']:,.2f}</td>
  <td class="mono" style="color:#ff5050">{t['sl']:,.2f}</td>
  <td class="mono" style="color:#00aa55">{t['tp1']:,.2f}</td>
  <td class="mono" style="color:#00ff88">{t['tp2']:,.2f}</td>
  <td style="color:{ticks_color_inline}">{ticks_str}</td>
  <td>{badge}</td>
</tr>"""

            st.markdown(f"""
<div style="overflow-x:auto">
<table class="th-table">
  <thead><tr>
    <th>Time (PT)</th><th>Dir</th><th>Entry</th>
    <th>{tip('SL','SL')}</th><th>{tip('TP1','TP1')}</th><th>{tip('TP2','TP2')}</th>
    <th>{tip('Ticks','Tick')}</th><th>Result</th>
  </tr></thead>
  <tbody>{rows}</tbody>
</table>
</div>""", unsafe_allow_html=True)

        else:
            st.info("No signals recorded yet. Signals are captured automatically when direction changes.")

# ─── News Tab ─────────────────────────────────────────────────────────────────
def render_news_tab():
    st.markdown("<div style='height:8px'></div>", unsafe_allow_html=True)

    with st.spinner("Loading latest news..."):
        articles = fetch_news()

    if not articles:
        st.warning("Could not load news feeds. Check your internet connection.")
        return

    # ── Overall market sentiment across all symbols ───────────────────────────
    def overall_sent(groups):
        rel = [a for a in articles if any(g in a["groups"] for g in groups)][:20]
        if not rel: return 0.0
        return sum(a["compound"] for a in rel) / len(rel)

    nasdaq_sent = overall_sent(["nasdaq", "macro"])
    sp500_sent  = overall_sent(["sp500",  "macro"])
    gold_sent   = overall_sent(["gold",   "macro"])

    def sent_html(score):
        if score >= 0.2:   return f'<span class="pos">▲ Positive news ({score:+.2f})</span>'
        elif score <= -0.2: return f'<span class="neg">▼ Negative news ({score:+.2f})</span>'
        else:               return f'<span class="neu">● Mixed news ({score:+.2f})</span>'

    st.markdown(f"""
<div class="panel-grid" style="grid-template-columns:repeat(3,1fr)">
  <div class="mc" style="height:auto;padding:16px 18px">
    <div class="mc-label">Nasdaq / Tech Sentiment</div>
    <div class="mc-value" style="font-size:1.1em;margin:8px 0">{sent_html(nasdaq_sent)}</div>
    <div class="mc-delta neu">MNQ · NQ signals</div>
  </div>
  <div class="mc" style="height:auto;padding:16px 18px">
    <div class="mc-label">S&P 500 Sentiment</div>
    <div class="mc-value" style="font-size:1.1em;margin:8px 0">{sent_html(sp500_sent)}</div>
    <div class="mc-delta neu">MES · ES signals</div>
  </div>
  <div class="mc" style="height:auto;padding:16px 18px">
    <div class="mc-label">Gold / Macro Sentiment</div>
    <div class="mc-value" style="font-size:1.1em;margin:8px 0">{sent_html(gold_sent)}</div>
    <div class="mc-delta neu">GC · MGC signals</div>
  </div>
</div>
""", unsafe_allow_html=True)

    st.markdown("""
<div style="background:rgba(255,255,255,0.04);border:1px solid rgba(255,255,255,0.08);
     border-radius:10px;padding:12px 16px;margin:12px 0;font-size:12px;color:#8e8e93">
  📰 <b style="color:#ebebf5">How news affects your signals:</b>
  Positive news can add up to +1.5 points to a signal. Negative news can subtract up to 1.5 points.
  Big events like Fed meetings, CPI, and jobs reports are counted twice as heavily.
  News older than 4 hours fades out. Check the "Why This Signal" section on each tab to see the news impact.
</div>
""", unsafe_allow_html=True)

    st.markdown("<div style='height:4px'></div>", unsafe_allow_html=True)

    # ── Filters ──────────────────────────────────────────────────────────────
    f1, f2 = st.columns([2, 1])
    with f1:
        filter_group = st.selectbox("Filter by market", ["All", "Nasdaq/Tech", "S&P 500", "Gold", "Big Events Only"],
                                    key="news_filter")
    with f2:
        filter_sent = st.selectbox("News tone", ["All", "Positive", "Negative", "Mixed"], key="news_sent_filter")

    group_map = {"Nasdaq/Tech": "nasdaq", "S&P 500": "sp500", "Gold": "gold"}

    filtered = articles
    if filter_group == "Big Events Only":
        filtered = [a for a in filtered if a["high_impact"]]
    elif filter_group in group_map:
        filtered = [a for a in filtered if group_map[filter_group] in a["groups"]
                    or "macro" in a["groups"]]
    if filter_sent == "Positive":
        filtered = [a for a in filtered if a["compound"] >= 0.1]
    elif filter_sent == "Negative":
        filtered = [a for a in filtered if a["compound"] <= -0.1]
    elif filter_sent == "Mixed":
        filtered = [a for a in filtered if -0.1 < a["compound"] < 0.1]

    st.markdown(f"<div style='color:#8e8e93;font-size:12px;margin:8px 0'>{len(filtered)} articles</div>",
                unsafe_allow_html=True)

    # ── Article cards ─────────────────────────────────────────────────────────
    cards_html = ""
    for a in filtered[:25]:
        label, color, emoji = sentiment_label(a["compound"])

        age_min = max(0, a["age_min"])  # clamp negative (future-dated RSS items)
        if age_min < 60:
            age_str = f"{int(age_min)}m ago" if age_min >= 1 else "just now"
        elif age_min < 1440:
            age_str = f"{int(age_min/60)}h ago"
        else:
            age_str = f"{int(age_min/1440)}d ago"

        tag_map  = {"nasdaq": "MNQ", "sp500": "MES", "gold": "MGC", "macro": "ALL"}
        tags_str = " · ".join(tag_map[g] for g in a["groups"] if g in tag_map)
        mover_html = '&nbsp;<span style="color:#fbbf24;font-size:10px;font-weight:700">⚡ MOVER</span>' if a["high_impact"] else ""

        # Safely escape user-supplied text before embedding in HTML
        title_s   = html.escape(a["title"])
        summary_s = html.escape(a["summary"][:200]) + ("…" if len(a["summary"]) > 200 else "")
        source_s  = html.escape(a["source"])

        score_str = f"{a['compound']:+.2f}"

        meta_parts = [source_s, age_str]
        if tags_str:
            meta_parts.append(tags_str)
        meta_html = " &nbsp;·&nbsp; ".join(meta_parts)

        link_s = html.escape(a.get("link", ""))
        title_el = (f'<a href="{link_s}" target="_blank" rel="noopener noreferrer" '
                    f'style="color:#f1f5f9;text-decoration:none;'
                    f'border-bottom:1px solid rgba(129,140,248,0.35)">{title_s}</a>'
                    if link_s else title_s)

        cards_html += f"""
<div style="display:flex;gap:14px;padding:16px 0;border-bottom:1px solid rgba(255,255,255,0.05)">
  <div style="width:3px;min-height:60px;background:{color};border-radius:2px;flex-shrink:0"></div>
  <div style="flex:1;min-width:0">
    <div style="font-family:'Inter',sans-serif;font-size:13px;font-weight:600;
         line-height:1.45;margin-bottom:5px">{title_el}{mover_html}</div>
    <div style="font-family:'Inter',sans-serif;font-size:12px;color:#64748b;
         line-height:1.5;margin-bottom:6px">{summary_s}</div>
    <div style="font-family:'Inter',sans-serif;font-size:11px;color:#475569">
      <span style="color:{color}">{label}</span> &nbsp;·&nbsp; {meta_html}
    </div>
  </div>
  <div style="flex-shrink:0;text-align:right;min-width:48px">
    <div style="font-family:'Inter',sans-serif;font-size:15px;font-weight:700;color:{color}">{score_str}</div>
    <div style="font-family:'Inter',sans-serif;font-size:10px;color:#475569;margin-top:2px;letter-spacing:0.04em">score</div>
  </div>
</div>"""

    st.markdown(f'<div style="margin-top:4px">{cards_html}</div>', unsafe_allow_html=True)

    # ── Economic event reminder ───────────────────────────────────────────────
    st.markdown("<div style='height:16px'></div>", unsafe_allow_html=True)
    with st.expander("Key Economic Events to Watch", expanded=False):
        for ev in ECON_EVENTS:
            impact_color = "#ff375f" if ev["impact"] == "HIGH" else "#ffd60a"
            st.markdown(f"""
<div style="background:#1c1c1e;border:1px solid rgba(255,255,255,0.07);border-radius:8px;
     padding:12px 14px;margin:5px 0;display:flex;gap:12px;align-items:flex-start">
  <span style="background:rgba(255,255,255,0.06);color:{impact_color};font-size:10px;
        font-weight:700;border-radius:4px;padding:2px 7px;white-space:nowrap;margin-top:2px">{ev['impact']}</span>
  <div>
    <div style="font-size:13px;font-weight:600;color:#f5f5f7;margin-bottom:3px">{ev['name']}</div>
    <div style="font-size:12px;color:#8e8e93">{ev['desc']}</div>
  </div>
</div>
""", unsafe_allow_html=True)

# ─── Trade Log Tab ────────────────────────────────────────────────────────────
def render_trade_log():
    st.markdown("<div style='height:8px'></div>", unsafe_allow_html=True)

    # Show DB error if any
    if "_db_error" in st.session_state:
        st.error(f"Database error: {st.session_state['_db_error']}")

    all_trades = load_trades()

    # Only show MNQ=F, MES=F, MGC=F — exclude legacy GC=F, NQ=F, ES=F
    all_trades = [t for t in all_trades if t["symbol"] in TICK_INFO]

    if not all_trades:
        st.info("No trades recorded yet. Signals are saved automatically when a direction is detected on any tab.")
        return

    # ── Overall stats across all instruments ─────────────────────────────────
    stats_all = get_stats(all_trades)
    by_sym = {}
    for sym in TICK_INFO:
        s = get_stats(all_trades, sym)
        if s["total"] > 0 or s["open"] > 0:
            by_sym[sym] = s

    wr_color = "#30d158" if stats_all["win_rate"] >= 55 else ("#ffd60a" if stats_all["win_rate"] >= 40 else "#ff375f")
    ticks_cls = "pos" if stats_all["total_ticks"] >= 0 else "neg"

    st.markdown(f"""
<div style="background:#1c1c1e;border:1px solid rgba(255,255,255,0.08);border-radius:14px;padding:20px 24px;margin-bottom:16px">
  <div style="font-size:11px;font-weight:600;letter-spacing:0.06em;text-transform:uppercase;color:#8e8e93;margin-bottom:14px">Overall Record — All Instruments</div>
  <div style="display:grid;grid-template-columns:repeat(5,1fr);gap:12px">
    <div><div style="font-size:10px;color:#8e8e93;margin-bottom:4px">Win Rate</div>
         <div style="font-size:1.6em;font-weight:700;color:{wr_color}">{stats_all['win_rate']}%</div></div>
    <div><div style="font-size:10px;color:#8e8e93;margin-bottom:4px">Total Trades</div>
         <div style="font-size:1.6em;font-weight:700;color:#f5f5f7">{stats_all['total']}</div></div>
    <div><div style="font-size:10px;color:#8e8e93;margin-bottom:4px">Wins</div>
         <div style="font-size:1.6em;font-weight:700;color:#30d158">{stats_all['wins']}</div></div>
    <div><div style="font-size:10px;color:#8e8e93;margin-bottom:4px">Losses</div>
         <div style="font-size:1.6em;font-weight:700;color:#ff375f">{stats_all['losses']}</div></div>
    <div><div style="font-size:10px;color:#8e8e93;margin-bottom:4px">Total Ticks</div>
         <div style="font-size:1.6em;font-weight:700" class="{ticks_cls}">{stats_all['total_ticks']:+.1f}</div></div>
  </div>
</div>
""", unsafe_allow_html=True)

    # ── In-hours vs Off-hours win rates ──────────────────────────────────────
    closed = [t for t in all_trades if t["status"] != "open"]
    in_h   = [t for t in closed if t["symbol"] in _SESSION_GATE_EXEMPT or trade_in_session(t["symbol"], t["timestamp"])]
    off_h  = [t for t in closed if t["symbol"] not in _SESSION_GATE_EXEMPT and not trade_in_session(t["symbol"], t["timestamp"])]

    def _wr(trades_subset):
        if not trades_subset:
            return 0, 0, 0
        wins = sum(1 for t in trades_subset if t["status"].startswith("win"))
        return round(wins / len(trades_subset) * 100), wins, len(trades_subset) - wins

    in_wr,  in_w,  in_l  = _wr(in_h)
    off_wr, off_w, off_l = _wr(off_h)
    in_color  = "#30d158" if in_wr  >= 55 else ("#ffd60a" if in_wr  >= 40 else "#ff375f")
    off_color = "#30d158" if off_wr >= 55 else ("#ffd60a" if off_wr >= 40 else "#ff375f")

    col_in, col_off = st.columns(2)
    col_in.markdown(f"""
<div style="background:#1c1c1e;border:1px solid rgba(52,211,153,0.2);border-radius:12px;padding:16px 20px">
  <div style="font-size:10px;font-weight:700;letter-spacing:0.07em;color:#34d399;margin-bottom:10px">IN HOURS</div>
  <div style="font-size:2em;font-weight:800;color:{in_color}">{in_wr}%</div>
  <div style="font-size:11px;color:#8e8e93;margin-top:4px">{in_w}W / {in_l}L &nbsp;·&nbsp; {len(in_h)} closed trades</div>
</div>""", unsafe_allow_html=True)

    col_off.markdown(f"""
<div style="background:#1c1c1e;border:1px solid rgba(255,255,255,0.07);border-radius:12px;padding:16px 20px">
  <div style="font-size:10px;font-weight:700;letter-spacing:0.07em;color:#636366;margin-bottom:10px">OFF HOURS</div>
  <div style="font-size:2em;font-weight:800;color:{off_color}">{off_wr}%</div>
  <div style="font-size:11px;color:#8e8e93;margin-top:4px">{off_w}W / {off_l}L &nbsp;·&nbsp; {len(off_h)} closed trades</div>
</div>""", unsafe_allow_html=True)

    st.markdown("<div style='height:8px'></div>", unsafe_allow_html=True)

    # ── Per-instrument mini stats ─────────────────────────────────────────────
    if by_sym:
        cols = st.columns(len(by_sym))
        for i, (sym, s) in enumerate(by_sym.items()):
            name = _ti(sym)["name"]
            wrc  = "#30d158" if s["win_rate"] >= 55 else ("#ffd60a" if s["win_rate"] >= 40 else "#ff375f")
            tc   = "pos" if s["total_ticks"] >= 0 else "neg"
            cols[i].markdown(f"""
<div class="mc" style="height:auto;padding:14px 16px">
  <div class="mc-label">{name}</div>
  <div style="font-size:1.2em;font-weight:700;color:{wrc};margin:6px 0">{s['win_rate']}% wins</div>
  <div style="font-size:11px;color:#8e8e93">{s['wins']}W / {s['losses']}L / <span style="color:#ffd60a">{s['open']} open</span></div>
  <div style="font-size:11px;margin-top:4px" class="{tc}">{s['total_ticks']:+.1f} ticks</div>
</div>""", unsafe_allow_html=True)

    st.markdown("<div style='height:14px'></div>", unsafe_allow_html=True)

    # ── Filters ───────────────────────────────────────────────────────────────
    fc1, fc2, fc3 = st.columns(3)
    sym_options = ["All"] + [TICK_INFO[s]["name"] for s in TICK_INFO
                             if any(t["symbol"] == s for t in all_trades)]
    with fc1:
        f_sym = st.selectbox("Instrument", sym_options, key="log_sym")
    with fc2:
        f_dir = st.selectbox("Direction", ["All", "LONG", "SHORT"], key="log_dir")
    with fc3:
        f_res = st.selectbox("Result", ["All", "Open", "Win", "Loss"], key="log_res")

    # Apply filters
    filtered = list(reversed(all_trades))  # newest first
    if f_sym != "All":
        filtered = [t for t in filtered if _ti(t["symbol"])["name"] == f_sym]
    if f_dir != "All":
        filtered = [t for t in filtered if t["direction"] == f_dir]
    if f_res == "Open":
        filtered = [t for t in filtered if t["status"] == "open"]
    elif f_res == "Win":
        filtered = [t for t in filtered if t["status"].startswith("win")]
    elif f_res == "Loss":
        filtered = [t for t in filtered if t["status"] == "loss"]

    st.markdown(f"<div style='color:#8e8e93;font-size:12px;margin:4px 0 10px'>{len(filtered)} trades</div>",
                unsafe_allow_html=True)

    # ── Table ─────────────────────────────────────────────────────────────────
    if filtered:
        rows = ""
        for t in filtered[:50]:
            s = t["status"]
            if s == "win_tp2":
                result_badge = '<span class="badge badge-win2">WIN — Hit TP2</span>'
            elif s == "win_tp1":
                result_badge = '<span class="badge badge-win1">WIN — Hit TP1</span>'
            elif s == "loss":
                result_badge = '<span class="badge badge-loss">LOSS — Hit Stop</span>'
            else:
                result_badge = '<span class="badge badge-open">OPEN</span>'

            dir_badge = (f'<span class="badge badge-long">BUY</span>'
                         if t["direction"] == "LONG"
                         else '<span class="badge badge-short">SELL</span>')

            _tinfo     = _ti(t["symbol"])
            name = _tinfo["name"]
            ticks_str  = f'{t["pnl_ticks"]:+.1f}' if t.get("pnl_ticks") is not None else "—"
            ticks_color = "#30d158" if (t.get("pnl_ticks") or 0) > 0 else ("#ff375f" if (t.get("pnl_ticks") or 0) < 0 else "#8e8e93")
            tick_val   = _tinfo["value"]
            dollar_str = f'${abs((t.get("pnl_ticks") or 0) * tick_val):,.0f}' if t.get("pnl_ticks") is not None else "—"
            dollar_color = ticks_color

            # Signal strength at entry
            score = t.get("score", None)
            if score is not None:
                strength = min(int(abs(score) / 6.0 * 100), 100)
                if strength >= 70:
                    strength_color = "#30d158"
                elif strength >= 45:
                    strength_color = "#ffd60a"
                else:
                    strength_color = "#ff375f"
                strength_str = f"{strength}%"
            else:
                strength_str  = "—"
                strength_color = "#8e8e93"

            try:
                ts = datetime.fromisoformat(t["timestamp"]).astimezone(PT).strftime("%m/%d  %I:%M %p")
            except Exception:
                ts = t["timestamp"][:16]

            in_sess = t["symbol"] in _SESSION_GATE_EXEMPT or trade_in_session(t["symbol"], t["timestamp"])
            sess_badge = ('<span style="font-size:10px;font-weight:700;color:#34d399">IN HOURS</span>'
                         if in_sess else
                         '<span style="font-size:10px;font-weight:700;color:#636366">OFF HOURS</span>')

            rows += f"""
<tr>
  <td style="color:#636366;white-space:nowrap">{ts}</td>
  <td><b style="color:#f5f5f7">{name}</b></td>
  <td>{dir_badge}</td>
  <td>{sess_badge}</td>
  <td class="mono" style="color:#f5f5f7">{t['entry']:,.2f}</td>
  <td class="mono" style="color:#ff375f">{t['sl']:,.2f}</td>
  <td class="mono" style="color:#30d158">{t['tp1']:,.2f}</td>
  <td class="mono" style="color:#34c759">{t['tp2']:,.2f}</td>
  <td style="color:{ticks_color};font-weight:600">{ticks_str}</td>
  <td style="color:{strength_color};font-weight:600">{strength_str}</td>
  <td style="color:{dollar_color};font-weight:600">{dollar_str}</td>
  <td>{result_badge}</td>
</tr>"""

        st.markdown(f"""
<div style="overflow-x:auto;border-radius:10px;border:1px solid rgba(255,255,255,0.07)">
<table class="th-table">
  <thead><tr>
    <th>Date / Time (PT)</th>
    <th>Market</th>
    <th>Trade</th>
    <th>Session</th>
    <th>Entry Price</th>
    <th>Stop</th>
    <th>Target 1</th>
    <th>Target 2</th>
    <th>Ticks</th>
    <th>Strength</th>
    <th>P&amp;L</th>
    <th>Result</th>
  </tr></thead>
  <tbody>{rows}</tbody>
</table>
</div>""", unsafe_allow_html=True)

        st.markdown("<div style='height:12px'></div>", unsafe_allow_html=True)
    else:
        st.info("No trades match your filters.")

# ─── Scaling Guide ────────────────────────────────────────────────────────────
def render_scale_guide(rules: dict):
    with st.expander("Scaling Strategy & Session Guide", expanded=False):
        st.markdown(f"""
<h4>How to {tip('Scale','Scaling')} Into Signals</h4>
<ol>
  <li><b>Entry 1</b> — 1 {tip('contract','Contract')} when signal fires ({tip('EMA','EMA')} stack + {tip('MACD','MACD')} agree)</li>
  <li><b>Add</b> — 1 more {tip('contract','Contract')} if price moves 8–10 {tip('ticks','Tick')} in your favor. Move {tip('stop loss','SL')} on position 1 to break-even</li>
  <li><b>Exit {tip('TP1','TP1')}</b> — Close 60% of position, slide remaining {tip('stop loss','SL')} to entry</li>
  <li><b>Exit {tip('TP2','TP2')}</b> — Let the last 40% run to TP2</li>
</ol>

<h4>Risk Rules</h4>
<ul>
  <li>Never risk more than 30% of your {tip('daily loss limit','Daily Loss')} on one trade</li>
  <li>Your max risk per trade right now: <b>${rules['daily_loss'] * 0.3:,.0f}</b></li>
  <li>Always ensure {tip('R:R','R:R')} is at least 1:1.5 before entering</li>
  <li>After 2 losses in a row — stop trading for the day</li>
</ul>

<h4>Best Times (California / PT)</h4>
<table style="font-size:13px; border-collapse:collapse; width:100%">
  <tr style="border-bottom:1px solid #333"><th style="text-align:left;padding:5px 8px;color:#888">Market</th><th style="text-align:left;padding:5px 8px;color:#888">Window</th><th style="text-align:left;padding:5px 8px;color:#888">Notes</th></tr>
  <tr><td style="padding:5px 8px">{tip('MNQ','MNQ')} / {tip('MES','MES')}</td><td style="padding:5px 8px"><b>6:30–8:30 AM PT</b></td><td style="padding:5px 8px;color:#888">Market open — highest {tip('momentum','Momentum')}. Bot active.</td></tr>
  <tr><td style="padding:5px 8px">{tip('MNQ','MNQ')} / {tip('MES','MES')}</td><td style="padding:5px 8px"><b>11:00 AM–1:00 PM PT</b></td><td style="padding:5px 8px;color:#888">Market close — second best window. Bot active.</td></tr>
  <tr style="color:#ff8080"><td style="padding:5px 8px">❌ {tip('MNQ','MNQ')} / {tip('MES','MES')}</td><td style="padding:5px 8px"><b>All other hours</b></td><td style="padding:5px 8px">Signals paused — equity futures go dead overnight</td></tr>
  <tr><td style="padding:5px 8px">{tip('MGC','MGC')}</td><td style="padding:5px 8px"><b>12:00–2:00 AM PT</b></td><td style="padding:5px 8px;color:#888">London session — strong gold moves</td></tr>
  <tr><td style="padding:5px 8px">{tip('MGC','MGC')}</td><td style="padding:5px 8px"><b>5:00–9:00 AM PT</b></td><td style="padding:5px 8px;color:#888">COMEX open — highest gold volume</td></tr>
  <tr><td style="padding:5px 8px">{tip('MGC','MGC')}</td><td style="padding:5px 8px"><b>All hours</b></td><td style="padding:5px 8px;color:#888">Gold trades 24h — bot records signals around the clock</td></tr>
  <tr style="color:#ff8080"><td style="padding:5px 8px">❌ All</td><td style="padding:5px 8px"><b>8:30–11:00 AM PT</b></td><td style="padding:5px 8px">Midday chop — slow, unpredictable for MNQ/MES</td></tr>
</table>

<br><h4>{tip('Eval','Eval')} Tips</h4>
<ul>
  <li>Hit <b>65% of profit target</b> then go small — protect the pass</li>
  <li>Never revenge trade. Down 50% of {tip('daily loss','Daily Loss')}? Log off</li>
  <li>Watch your {tip('drawdown','Drawdown')} — never let a bad streak snowball</li>
  <li>3–5% account growth per day is the sweet spot for passing consistently</li>
</ul>
""", unsafe_allow_html=True)

# ─── Settings Tab (replaces sidebar entirely) ─────────────────────────────────
def render_settings_tab():
    cfg = load_config()

    # ── Chart Timeframe ───────────────────────────────────────────────────────
    st.markdown("### Chart Timeframe")
    st.selectbox("Timeframe", ["1m","2m","5m","15m","30m","1h"], index=2, key="tf")
    st.caption("Controls the candle interval used on the MNQ, MES, and MGC chart tabs.")
    st.divider()

    # ── Eval Tracker ──────────────────────────────────────────────────────────
    st.markdown("### Eval Tracker")
    acct = st.selectbox("Account Size", list(TOPSTEP_ACCOUNTS.keys()), key="acct")
    rules = TOPSTEP_ACCOUNTS[acct]
    # store for use by scale guide in main()
    st.session_state["rules"] = rules

    st.markdown("**Enter your P&L**")
    col1, col2 = st.columns(2)
    with col1:
        daily_pnl = st.number_input("Today's P&L ($)", value=0.0, step=50.0, key="daily_pnl")
    with col2:
        total_pnl = st.number_input("Total P&L ($)", value=0.0, step=100.0, key="total_pnl")

    daily_rem = rules["daily_loss"] + daily_pnl
    dd_used   = max(0.0, -total_pnl)
    pct = lambda v, mx: max(0.0, min(1.0, v / mx))

    st.markdown("**Eval Progress**")
    st.progress(pct(daily_rem, rules["daily_loss"]),
                text=f"Daily loss room: ${daily_rem:,.0f} / ${rules['daily_loss']:,.0f}")
    st.caption("How much you can still lose today")
    st.progress(pct(max(0, total_pnl), rules["target"]),
                text=f"Profit target: ${total_pnl:,.0f} / ${rules['target']:,.0f}")
    st.caption("Reach this to pass the eval")
    st.progress(pct(dd_used, rules["max_dd"]),
                text=f"Max drawdown: ${dd_used:,.0f} / ${rules['max_dd']:,.0f}")
    st.caption("If account drops this far, eval ends")

    if daily_pnl <= -rules["daily_loss"]:
        st.error("STOP — Daily loss limit hit!")
    elif dd_used >= rules["max_dd"]:
        st.error("ACCOUNT BUSTED — Max drawdown hit!")
    elif total_pnl >= rules["target"]:
        st.success("EVAL PASSED! Contact your platform.")
    elif daily_rem < rules["daily_loss"] * 0.25:
        st.warning("Low daily loss buffer — trade small")
    else:
        st.info(f"Active — max {rules['contract_limit']} contracts")

    # ── Phone Alerts ──────────────────────────────────────────────────────────
    st.divider()
    st.markdown("### Phone Alerts")
    st.caption("Get a notification on your phone when a strong signal fires.")

    notify_on = st.toggle("Send alerts to my phone", value=cfg.get("notify_enabled", False), key="notify_toggle")
    topic_val = st.text_input(
        "Your alert topic name",
        value=cfg.get("ntfy_topic", "topstepdraco42"),
        key="ntfy_topic_input",
        help="This is the topic name you subscribed to in the ntfy app."
    )
    min_score = st.slider(
        "Only alert me when signal strength is at least",
        min_value=2.5, max_value=6.0, step=0.5,
        value=float(cfg.get("min_score", 3.0)),
        key="ntfy_min_score",
        help="Higher = fewer but stronger alerts. Max possible score is 6.0."
    )

    new_cfg = {"ntfy_topic": topic_val, "notify_enabled": notify_on, "min_score": min_score}
    if new_cfg != cfg:
        save_config(new_cfg)

    if notify_on and topic_val:
        st.success(f"Active — alerts going to: `{topic_val}`")
        if st.button("Send test alert now", key="test_notif"):
            try:
                requests.post(
                    f"https://ntfy.sh/{topic_val}",
                    data="PaperTrail is connected. You will get alerts here when a signal fires.".encode("utf-8"),
                    headers={"Title": "PaperTrail - Test Alert", "Priority": "default"},
                    timeout=5,
                )
                st.success("Sent! Check your phone.")
            except Exception as e:
                st.error(f"Failed: {e}")
    elif notify_on and not topic_val:
        st.warning("Enter a topic name to activate.")
    else:
        st.markdown("""
**Quick setup (2 minutes):**

1. Download the **ntfy** app — free, no account needed
   - iPhone: search "ntfy" in the App Store
   - Android: search "ntfy" in the Play Store
2. Open the app, tap **+**, subscribe to topic: `topstepdraco42`
3. Toggle on alerts above, then tap **Send test alert now**
""")


# ─── Dashboard Tab ────────────────────────────────────────────────────────────
@st.cache_data(ttl=15, show_spinner=False)
def _quick_signal(symbol: str, interval: str, period: str) -> dict:
    """Fetch full signal for dashboard — cached 30s."""
    try:
        df = fetch_data(symbol, interval, period)
        if df.empty:
            return {"direction": "NEUTRAL", "score": 0, "price": None, "_full": None}
        df = compute_indicators(df)

        # Higher timeframe bias — 15m (intermediate) + 1h (macro)
        df_15m       = fetch_data(symbol, "15m", "60d")
        df_15m       = compute_indicators(df_15m) if not df_15m.empty else df_15m
        htf_bias_15m = get_htf_bias(df_15m) if not df_15m.empty else 0

        df_1h        = fetch_data(symbol, "1h", "730d")
        df_1h        = compute_indicators(df_1h) if not df_1h.empty else df_1h
        htf_bias_1h  = get_htf_bias(df_1h) if not df_1h.empty else 0

        articles       = fetch_news()
        news_sentiment = get_news_sentiment(symbol, articles)
        sig            = generate_signal(df, symbol, news_sentiment, htf_bias_15m=htf_bias_15m, htf_bias_1h=htf_bias_1h)
        check_open_trades(symbol, df)
        price = float(df.iloc[-1]["Close"])
        return {"direction": sig["direction"], "score": sig["score"], "price": price, "_full": sig,
                "htf_bias_15m": htf_bias_15m, "htf_bias_1h": htf_bias_1h}
    except Exception:
        return {"direction": "NEUTRAL", "score": 0, "price": None, "_full": None,
                "htf_bias_15m": 0, "htf_bias_1h": 0}

def render_dashboard(interval: str, period: str):
    st.markdown("### Market Overview")

    groups = [
        {"label": "Nasdaq",  "symbols": ["MNQ=F"]},
        {"label": "S&P 500", "symbols": ["MES=F"]},
        {"label": "Gold",    "symbols": ["MGC=F"]},
    ]

    for group in groups:
        st.markdown(f"**{group['label']}**")
        cols = st.columns(len(group["symbols"]))
        for col, symbol in zip(cols, group["symbols"]):
            sig = _quick_signal(symbol, interval, period)
            d = sig["direction"]
            score = sig["score"]
            price = sig["price"]
            name = TICK_INFO[symbol]["name"]

            # Close any open trades first (outside cache), then record new signal if conditions met
            full_sig = sig.get("_full")
            if full_sig:
                _df_live = fetch_data(symbol, interval, period)
                if not _df_live.empty:
                    check_open_trades(symbol, _df_live)
                if should_record_signal(full_sig, symbol):
                    record_signal(full_sig, symbol, interval)  # record_signal handles the notification

            if d == "LONG":
                bg      = "linear-gradient(135deg,#0d2b1a,#0a1f12)"
                border  = "rgba(48,209,88,0.5)"
                color   = "#30d158"
                icon    = "▲"
                label   = "LONG"
            elif d == "SHORT":
                bg      = "linear-gradient(135deg,#2b0d12,#1f0a0d)"
                border  = "rgba(255,55,95,0.5)"
                color   = "#ff375f"
                icon    = "▼"
                label   = "SHORT"
            else:
                bg      = "#1c1c1e"
                border  = "rgba(255,255,255,0.1)"
                color   = "#8e8e93"
                icon    = "●"
                label   = "WAITING"

            strength = int(abs(score) / 6.0 * 100)
            price_str = f"{price:,.2f}" if price else "—"
            sess_on, sess_reason, _ = trading_session_active(symbol)
            if symbol in _SESSION_GATE_EXEMPT:
                sess_badge = ('<div style="margin-top:10px;font-size:9px;font-weight:700;letter-spacing:0.07em;'
                              'color:#34d399">● ACTIVE 24H</div>')
            else:
                sess_badge = (
                    '<div style="margin-top:10px;font-size:9px;font-weight:700;letter-spacing:0.07em;'
                    'color:#34d399">● ACTIVE SESSION</div>'
                    if sess_on else
                    '<div style="margin-top:10px;font-size:9px;font-weight:700;letter-spacing:0.07em;'
                    'color:#fbbf24">⏸ PAUSED</div>'
                )

            def _bias_pill(label: str, bias: int) -> str:
                if bias == 1:
                    return (f'<span style="font-size:9px;font-weight:700;letter-spacing:0.05em;'
                            f'color:#30d158">▲ {label}</span>')
                elif bias == -1:
                    return (f'<span style="font-size:9px;font-weight:700;letter-spacing:0.05em;'
                            f'color:#ff375f">▼ {label}</span>')
                else:
                    return (f'<span style="font-size:9px;font-weight:700;letter-spacing:0.05em;'
                            f'color:#636366">— {label}</span>')

            b15 = sig.get("htf_bias_15m", 0)
            b1h = sig.get("htf_bias_1h",  0)
            htf_badge = (
                f'<div style="margin-top:6px;display:flex;justify-content:center;gap:10px">'
                f'{_bias_pill("1h", b1h)}'
                f'{_bias_pill("15m", b15)}'
                f'</div>'
            )

            with col:
                st.markdown(f"""
<div style="background:{bg};border:2px solid {border};border-radius:14px;
     padding:16px;text-align:center;margin-bottom:12px">
  <div style="font-size:11px;font-weight:700;letter-spacing:0.08em;
       color:#8e8e93;text-transform:uppercase;margin-bottom:6px">{name}</div>
  <div style="font-size:2em;line-height:1;color:{color};margin-bottom:4px">{icon}</div>
  <div style="font-size:14px;font-weight:800;color:{color};letter-spacing:0.04em">{label}</div>
  <div style="font-size:12px;color:#8e8e93;margin-top:4px">{price_str}</div>
  <div style="background:rgba(255,255,255,0.08);border-radius:4px;height:4px;margin-top:10px;overflow:hidden">
    <div style="width:{strength}%;height:100%;background:{color};border-radius:4px"></div>
  </div>
  <div style="font-size:10px;color:#636366;margin-top:4px">{strength}% strength</div>
  {sess_badge}
  {htf_badge}
</div>
""", unsafe_allow_html=True)
        st.markdown("<div style='height:4px'></div>", unsafe_allow_html=True)

    # ── TradingView chart ─────────────────────────────────────────────────────
    st.markdown("<div style='height:8px'></div>", unsafe_allow_html=True)
    st.markdown("### Chart")
    _poly_key = st.secrets.get("polygon", {}).get("api_key", "")
    if _poly_key:
        st.markdown("""
<div style="background:rgba(52,211,153,0.06);border:1px solid rgba(52,211,153,0.2);
     border-radius:10px;padding:10px 16px;margin-bottom:14px;
     font-family:'Inter',sans-serif;font-size:12px;color:#94a3b8;display:flex;gap:10px;align-items:center">
  <span style="color:#34d399;font-weight:700;flex-shrink:0">● Live Data</span>
  Real-time CME futures prices via Polygon.io — signals and TP/SL tracking are as accurate as possible.
</div>""", unsafe_allow_html=True)
    else:
        st.markdown("""
<div style="background:rgba(251,191,36,0.06);border:1px solid rgba(251,191,36,0.2);
     border-radius:10px;padding:10px 16px;margin-bottom:14px;
     font-family:'Inter',sans-serif;font-size:12px;color:#94a3b8;display:flex;gap:10px;align-items:center">
  <span style="color:#fbbf24;font-weight:700;flex-shrink:0">⚠ Data Notice</span>
  Prices shown are ~15 min delayed (yfinance free tier). Use this app for signal direction only — always enter at the live price on your trading platform.
</div>""", unsafe_allow_html=True)

    _TV_SYMBOLS = {
        "MNQ": "CME_MINI:MNQ1!",
        "MES": "CME_MINI:MES1!",
        "MGC": "COMEX:MGC1!",
    }
    _TV_INTERVALS = {"1m":"1","2m":"2","5m":"5","15m":"15","30m":"30","1h":"60"}
    tv_iv = _TV_INTERVALS.get(interval, "5")

    # Quick-select buttons — locked to our 3 symbols only
    qc1, qc2, qc3 = st.columns(3)
    with qc1:
        if st.button("MNQ", use_container_width=True, key="tv_mnq"):
            st.session_state["tv_sym"] = "NASDAQ:QQQ"
    with qc2:
        if st.button("MES", use_container_width=True, key="tv_mes"):
            st.session_state["tv_sym"] = "AMEX:SPY"
    with qc3:
        if st.button("MGC", use_container_width=True, key="tv_mgc"):
            st.session_state["tv_sym"] = "TVC:GOLD"

    tv_sym = st.session_state.get("tv_sym", "NASDAQ:QQQ")

    components.html(f"""
<div class="tradingview-widget-container" style="height:560px;width:100%">
  <div id="tv_dash_chart" style="height:100%;width:100%"></div>
  <script src="https://s3.tradingview.com/tv.js"></script>
  <script>
  new TradingView.widget({{
    "container_id":      "tv_dash_chart",
    "width":             "100%",
    "height":            560,
    "symbol":            "{tv_sym}",
    "interval":          "{tv_iv}",
    "timezone":          "America/Los_Angeles",
    "theme":             "dark",
    "style":             "1",
    "locale":            "en",
    "toolbar_bg":        "#0f1520",
    "hide_top_toolbar":  false,
    "hide_legend":       false,
    "save_image":        false,
    "enable_publishing": false,
    "allow_symbol_change": false,
    "withdateranges":    true
  }});
  </script>
</div>""", height=580)


# ─── Main ─────────────────────────────────────────────────────────────────────
# ─── Auth ─────────────────────────────────────────────────────────────────────
def _auth_token() -> str:
    pw = st.secrets.get("app_password", "topstep2024")
    return hashlib.sha256(pw.encode()).hexdigest()[:28]

def check_auth() -> bool:
    tok = _auth_token()

    # Auto-auth if valid token is in URL (survives refresh as long as URL is kept)
    if st.query_params.get("auth") == tok:
        st.session_state["_auth"] = True

    # Stale token in URL (password changed) — clear it and force re-login
    elif "auth" in st.query_params:
        st.query_params.clear()
        st.session_state.pop("_auth", None)
        st.rerun()

    if st.session_state.get("_auth"):
        return True

    # ── Login form ────────────────────────────────────────────────────────────
    st.markdown("""
<div style="max-width:380px;margin:80px auto;font-family:'Inter',sans-serif">
  <div style="font-size:22px;font-weight:700;color:#f1f5f9;margin-bottom:6px">PaperTrail</div>
  <div style="font-size:13px;color:#64748b;margin-bottom:28px">Enter your password to continue</div>
</div>""", unsafe_allow_html=True)

    col, _ = st.columns([1.4, 1])
    with col:
        pw       = st.text_input("Password", type="password", placeholder="Enter password",
                                 key="_pw", label_visibility="collapsed")
        remember = st.checkbox("Remember me on this browser", value=True, key="_remember")

        if st.button("Log in", use_container_width=True, type="primary"):
            if hashlib.sha256(pw.encode()).hexdigest()[:28] == tok:
                st.session_state["_auth"] = True
                if remember:
                    # Embed token in URL — browser keeps it on refresh, bookmark it to persist
                    st.query_params["auth"] = tok
                st.rerun()
            else:
                st.error("Incorrect password — try again")

        if remember:
            st.caption("Tip: bookmark the page after logging in to stay signed in permanently.")

    return False


def main():
    if not check_auth():
        st.stop()

    st.markdown("""
<link href="https://fonts.googleapis.com/css2?family=Sora:wght@700&display=swap" rel="stylesheet">
<div style="font-family:'Sora',sans-serif;font-size:42px;font-weight:700;
     letter-spacing:-0.02em;color:#f1f5f9;line-height:1;margin-bottom:4px">
  Paper<span style="color:#6366f1">Trail</span>
</div>""", unsafe_allow_html=True)
    st.caption(f"Live signals for MNQ, ES & Gold &nbsp;|&nbsp; {now_pt().strftime('%I:%M:%S %p %Z')}",
               unsafe_allow_html=True)
    st.markdown("""
<div style="background:rgba(255,255,255,0.03);border:1px solid rgba(255,255,255,0.07);
     border-radius:8px;padding:8px 14px;margin-top:4px;margin-bottom:2px;
     font-family:'Inter',sans-serif;font-size:12px;color:#64748b;line-height:1.5">
  ⚠️ <b style="color:#94a3b8">Not financial advice.</b>
  PaperTrail generates automated signals for informational purposes only.
  All trades are taken at your own discretion and risk. Past signal performance does not guarantee future results.
</div>""", unsafe_allow_html=True)

    from streamlit_autorefresh import st_autorefresh
    st_autorefresh(interval=30000, debounce=True, key="autorefresh")

    # Timeframe lives in Settings — read from session state with default
    period_map = {"1m":"7d","2m":"60d","5m":"60d","15m":"60d","30m":"60d","1h":"730d"}
    interval   = st.session_state.get("tf", "5m")
    period     = period_map.get(interval, "60d")

    st.markdown("<div style='height:28px'></div>", unsafe_allow_html=True)
    st.divider()

    tab_home, tab_mnq, tab_mes, tab_mgc, tab_news, tab_log, tab_settings = st.tabs([
        "Dashboard",
        "MNQ — Nasdaq",
        "MES — S&P 500",
        "MGC — Gold",
        "News",
        "Trade Log",
        "Settings",
    ])

    with tab_home:
        render_dashboard(interval, period)
        st.divider()
        render_scale_guide(st.session_state.get("rules", TOPSTEP_ACCOUNTS["$50K"]))

    with tab_mnq:
        st.divider()
        render_instrument("MNQ=F", interval, period)

    with tab_mes:
        st.divider()
        render_instrument("MES=F", interval, period)

    with tab_mgc:
        st.divider()
        render_instrument("MGC=F", interval, period)

    with tab_news:
        render_news_tab()

    with tab_log:
        render_trade_log()

    with tab_settings:
        render_settings_tab()

if __name__ == "__main__":
    main()
