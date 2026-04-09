"""
Signal Worker — runs 24/7 independent of the Streamlit app.
Checks MGC (Micro Gold) every 5 minutes, closes open trades when TP/SL hit,
records new signals, sends ntfy push notifications.
"""

import os, time, uuid, logging, requests, feedparser, re
from datetime import datetime, timedelta
from zoneinfo import ZoneInfo

import yfinance as yf
import pandas as pd
import ta
from supabase import create_client
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

# ─── Config ───────────────────────────────────────────────────────────────────
PT     = ZoneInfo("America/Los_Angeles")
_vader = SentimentIntensityAnalyzer()

SUPABASE_URL       = os.environ["SUPABASE_URL"]
SUPABASE_KEY       = os.environ["SUPABASE_KEY"]
NTFY_TOPIC         = "topstepnotis"
CHECK_INTERVAL_SEC = int(os.environ.get("CHECK_INTERVAL_SEC", "300"))  # 5 min

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)s  %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S"
)
log = logging.getLogger("worker")

TICK_INFO = {
    "MNQ=F": {"tick": 0.25, "value": 0.50,  "name": "MNQ"},
    "MES=F": {"tick": 0.25, "value": 1.25,  "name": "MES"},
    "MGC=F": {"tick": 0.10, "value": 1.00,  "name": "MGC"},
}

# ── Active symbols ────────────────────────────────────────────────────────────
ACTIVE_SYMBOLS = {"MGC=F"}

# Minimum score to record a signal per symbol (tuned from trade log)
_MIN_SCORE = {
    "MGC=F": 3.0,
}

# Max ticks the signal entry can differ from the live 1m price before rejecting
_MAX_ENTRY_STALENESS_TICKS = {
    "MGC=F": 15,   # 15 × $0.10 = 1.5 pts
}

def _get_live_price(symbol: str) -> float | None:
    """Fetch the most recent 1m close as a proxy for live price."""
    try:
        df1m = fetch_data(symbol, "1m", "1d")
        if not df1m.empty:
            return float(df1m["Close"].iloc[-1])
    except Exception:
        pass
    return None

def _entry_is_fresh(signal: dict, symbol: str) -> bool:
    """Reject signal if live 1m price has drifted too far from the entry."""
    entry = signal.get("entry")
    if entry is None:
        return True
    try:
        live_price = _get_live_price(symbol)
        if live_price is None:
            return True
        tick_sz    = TICK_INFO[symbol]["tick"]
        max_ticks  = _MAX_ENTRY_STALENESS_TICKS.get(symbol, 20)
        diff_ticks = abs(entry - live_price) / tick_sz
        if diff_ticks > max_ticks:
            log.warning(f"[staleness] {symbol} entry {entry} vs live {live_price:.2f} = {diff_ticks:.0f} ticks — REJECTED")
            return False
    except Exception as e:
        log.warning(f"[staleness] check failed for {symbol}: {e}")
    return True

_MAX_PERIOD = {
    "1m": "7d", "2m": "60d", "5m": "60d",
    "15m": "60d", "30m": "60d", "1h": "730d",
}

SYMBOL_GROUPS = {
    "MGC=F": ["gold", "macro"],
}

INSTRUMENT_KEYWORDS = {
    "nasdaq": ["nasdaq","tech","technology","apple","aapl","nvidia","nvda","microsoft","msft",
               "meta","amazon","amzn","alphabet","google","growth stocks","mnq","nq","qqq"],
    "sp500":  ["s&p","sp500","s&p 500","dow","russell","equities","stocks","wall street",
               "mes","es","spy","market rally","market selloff"],
    "gold":   ["gold","xau","precious metal","safe haven","bullion","gc","mgc","silver","commodities"],
    "macro":  ["fed","federal reserve","fomc","interest rate","rate hike","rate cut","inflation",
               "cpi","pce","nfp","jobs report","gdp","recession","jerome powell","treasury",
               "yield","dollar","dxy","debt ceiling","banking crisis","tariff","trade war",
               "geopolitical","ukraine","china","earnings"],
}

NEWS_FEEDS = [
    ("Yahoo Finance", "https://finance.yahoo.com/news/rssindex"),
    ("Reuters",       "https://feeds.reuters.com/reuters/businessNews"),
    ("CNBC",          "https://www.cnbc.com/id/100003114/device/rss/rss.html"),
    ("MarketWatch",   "https://feeds.content.dowjones.io/public/rss/mw_realtimeheadlines"),
]

# ─── Helpers ──────────────────────────────────────────────────────────────────
def now_pt():
    return datetime.now(PT)

def db():
    return create_client(SUPABASE_URL, SUPABASE_KEY)

def _snap(price: float, tick: float) -> float:
    return round(round(price / tick) * tick, 10)

# ─── Supabase ─────────────────────────────────────────────────────────────────
def load_trades() -> list:
    try:
        res = db().table("trades").select("data").order("created_at").execute()
        return [r["data"] for r in res.data] if res.data else []
    except Exception as e:
        log.error(f"load_trades: {e}")
        return []

def save_single_trade(trade: dict):
    try:
        db().table("trades").upsert({"id": trade["id"], "data": trade}).execute()
    except Exception as e:
        log.error(f"save_single_trade: {e}")

def load_config() -> dict:
    try:
        res = db().table("config").select("data").eq("id", "default").execute()
        if res.data:
            return res.data[0]["data"]
    except Exception as e:
        log.error(f"load_config: {e}")
    return {"ntfy_topic": "topstepnotis", "notify_enabled": True, "min_score": 3.5}

# ─── Notifications ────────────────────────────────────────────────────────────
def send_notification(symbol: str, trade: dict):
    cfg = load_config()
    if not cfg.get("notify_enabled"):
        return
    topic = cfg.get("ntfy_topic", NTFY_TOPIC).strip()
    if not topic:
        return
    min_sc = float(cfg.get("min_score", 3.5))
    if abs(trade.get("score", 0)) < min_sc:
        return

    ti       = TICK_INFO[symbol]
    d        = trade["direction"]
    strength = min(int(abs(trade["score"]) / 12.0 * 100), 99)
    sl_ticks = abs(trade["entry"] - trade["sl"]) / ti["tick"]
    tp1_ticks= abs(trade["entry"] - trade["tp1"]) / ti["tick"]

    title = f"{d} - {ti['name']} | Score {trade['score']:+.1f} ({strength}% strength)"
    body  = (f"Entry: {trade['entry']:,.2f}\n"
             f"Stop:  {trade['sl']:,.2f}  ({sl_ticks:.0f} ticks)\n"
             f"TP1:   {trade['tp1']:,.2f}  ({tp1_ticks:.0f} ticks)\n"
             f"TP2:   {trade['tp2']:,.2f}\n"
             f"Time:  {now_pt().strftime('%I:%M %p PT')}")
    priority = "urgent" if abs(trade["score"]) >= 4.5 else "high" if abs(trade["score"]) >= 3.5 else "default"
    tags     = "chart_with_upwards_trend" if d == "LONG" else "chart_with_downwards_trend"
    try:
        requests.post(
            f"https://ntfy.sh/{topic}",
            data=body.encode("utf-8"),
            headers={"Title": title, "Priority": priority, "Tags": tags},
            timeout=5,
        )
        log.info(f"Notified: {title}")
    except Exception as e:
        log.error(f"Notification failed: {e}")

# ─── News ─────────────────────────────────────────────────────────────────────
_news_cache: dict = {"articles": [], "fetched_at": None}

def fetch_news() -> list:
    now = now_pt()
    if _news_cache["fetched_at"] and (now - _news_cache["fetched_at"]).total_seconds() < 300:
        return _news_cache["articles"]
    articles, seen = [], set()
    for source, url in NEWS_FEEDS:
        try:
            for entry in feedparser.parse(url).entries[:15]:
                title = entry.get("title", "").strip()
                if not title or title in seen:
                    continue
                seen.add(title)
                summary  = re.sub(r"<[^>]+>", " ", entry.get("summary", ""))
                summary  = re.sub(r"\s+", " ", summary).strip()[:300]
                text     = f"{title}. {summary}"
                compound = _vader.polarity_scores(text)["compound"]
                low_text = text.lower()
                groups   = [g for g, kws in INSTRUMENT_KEYWORDS.items()
                            if any(k in low_text for k in kws)]
                high_impact = any(k in low_text for k in
                                  ["fed","fomc","cpi","nfp","gdp","rate","inflation",
                                   "jobs","recession","earnings","tariff"])
                articles.append({"title": title, "summary": summary, "source": source,
                                  "compound": compound, "groups": groups,
                                  "high_impact": high_impact})
        except Exception:
            pass
    _news_cache["articles"]   = articles
    _news_cache["fetched_at"] = now
    return articles

def get_news_sentiment(symbol: str, articles: list) -> dict:
    groups   = SYMBOL_GROUPS.get(symbol, [])
    relevant = [a for a in articles
                if any(g in a["groups"] for g in groups) or a["high_impact"]]
    if not relevant:
        return {"score": 0, "adjustment": 0, "count": 0}
    avg = sum(a["compound"] for a in relevant) / len(relevant)
    adj = max(-1.5, min(1.5, avg * 3))
    return {"score": avg, "adjustment": adj, "count": len(relevant)}

# ─── Data & Indicators ────────────────────────────────────────────────────────
def fetch_data(symbol: str, interval: str = "5m", period: str = "60d") -> pd.DataFrame:
    for attempt in range(3):
        try:
            df = yf.download(symbol, period=period, interval=interval,
                             progress=False, auto_adjust=True)
            if df.empty:
                return pd.DataFrame()
            df.columns = [c[0] if isinstance(c, tuple) else c for c in df.columns]
            return df.dropna()
        except Exception as e:
            if "Rate" in str(e) or "Too Many" in str(e):
                wait = 20 * (attempt + 1)
                log.warning(f"{symbol}: rate limited, retrying in {wait}s")
                time.sleep(wait)
            else:
                log.error(f"fetch_data {symbol}: {e}")
                return pd.DataFrame()
    log.error(f"fetch_data {symbol}: all retries failed")
    return pd.DataFrame()

def compute_indicators(df: pd.DataFrame) -> pd.DataFrame:
    if len(df) < 50:
        return df
    close, high, low = df["Close"], df["High"], df["Low"]
    df["EMA9"]  = ta.trend.ema_indicator(close, window=9)
    df["EMA21"] = ta.trend.ema_indicator(close, window=21)
    df["EMA50"] = ta.trend.ema_indicator(close, window=50)
    df["RSI"]   = ta.momentum.rsi(close, window=14)
    macd = ta.trend.MACD(close, window_slow=26, window_fast=12, window_sign=9)
    df["MACD"]        = macd.macd()
    df["MACD_signal"] = macd.macd_signal()
    df["MACD_hist"]   = macd.macd_diff()
    bb = ta.volatility.BollingerBands(close, window=20, window_dev=2)
    df["BB_upper"] = bb.bollinger_hband()
    df["BB_lower"] = bb.bollinger_lband()
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
    if   last["EMA9"] > last["EMA21"] > last["EMA50"]: bias += 2
    elif last["EMA9"] < last["EMA21"] < last["EMA50"]: bias -= 2
    elif last["EMA9"] > last["EMA21"]:                  bias += 1
    else:                                               bias -= 1
    if last["MACD"] > last["MACD_signal"]: bias += 1
    else:                                  bias -= 1
    rsi = float(last["RSI"])
    if   rsi > 55: bias += 0.5
    elif rsi < 45: bias -= 0.5
    if float(last["Close"]) > float(last["VWAP"]): bias += 0.5
    else:                                           bias -= 0.5
    if   bias >= 2.0: return  1
    elif bias <= -2.0: return -1
    return 0

# ─── Fair Value Gap Detection ─────────────────────────────────────────────────
def detect_fvg(df: pd.DataFrame) -> dict:
    """
    Scan the last 50 candles for unfilled Fair Value Gaps (FVGs).

    Bullish FVG: candle[i-2].High < candle[i].Low  — fast up-move left a gap; acts as support on retrace
    Bearish FVG: candle[i-2].Low  > candle[i].High — fast down-move left a gap; acts as resistance on retrace

    Returns:
      bullish        — list of {"top", "bottom", "ts"} for unfilled bullish FVGs
      bearish        — list of {"top", "bottom", "ts"} for unfilled bearish FVGs
      in_bullish_fvg — True if current price is inside a bullish FVG (buy zone)
      in_bearish_fvg — True if current price is inside a bearish FVG (sell zone)
      nearest_bull   — closest bullish FVG or None
      nearest_bear   — closest bearish FVG or None
    """
    result = {
        "bullish": [], "bearish": [],
        "in_bullish_fvg": False, "in_bearish_fvg": False,
        "nearest_bull": None, "nearest_bear": None,
    }
    if len(df) < 5:
        return result

    current_price = float(df["Close"].iloc[-2])  # use last closed candle
    lookback      = min(50, len(df) - 2)

    bullish_fvgs: list = []
    bearish_fvgs: list = []

    for i in range(2, lookback + 2):
        idx = len(df) - i
        if idx < 2:
            break

        c1_high = float(df["High"].iloc[idx - 2])
        c1_low  = float(df["Low"].iloc[idx - 2])
        c3_low  = float(df["Low"].iloc[idx])
        c3_high = float(df["High"].iloc[idx])
        ts      = df.index[idx]

        # ── Bullish FVG ────────────────────────────────────────────────────────
        if c1_high < c3_low:
            fvg_bottom, fvg_top = c1_high, c3_low
            # Filled if any later candle overlaps the gap
            filled = any(
                float(df["Low"].iloc[j])  <= fvg_top and
                float(df["High"].iloc[j]) >= fvg_bottom
                for j in range(idx + 1, len(df))
            )
            if not filled:
                bullish_fvgs.append({"top": fvg_top, "bottom": fvg_bottom, "ts": ts})

        # ── Bearish FVG ────────────────────────────────────────────────────────
        if c1_low > c3_high:
            fvg_bottom, fvg_top = c3_high, c1_low
            filled = any(
                float(df["Low"].iloc[j])  <= fvg_top and
                float(df["High"].iloc[j]) >= fvg_bottom
                for j in range(idx + 1, len(df))
            )
            if not filled:
                bearish_fvgs.append({"top": fvg_top, "bottom": fvg_bottom, "ts": ts})

    result["bullish"] = bullish_fvgs
    result["bearish"] = bearish_fvgs

    for fvg in bullish_fvgs:
        if fvg["bottom"] <= current_price <= fvg["top"]:
            result["in_bullish_fvg"] = True
            break

    for fvg in bearish_fvgs:
        if fvg["bottom"] <= current_price <= fvg["top"]:
            result["in_bearish_fvg"] = True
            break

    if bullish_fvgs:
        result["nearest_bull"] = min(bullish_fvgs,
            key=lambda f: abs(current_price - (f["top"] + f["bottom"]) / 2))
    if bearish_fvgs:
        result["nearest_bear"] = min(bearish_fvgs,
            key=lambda f: abs(current_price - (f["top"] + f["bottom"]) / 2))

    return result

# ─── Candlestick Pattern Detection ────────────────────────────────────────────
def detect_candle_patterns(df: pd.DataFrame) -> dict:
    """
    Detect key candlestick patterns on the last few closed candles.
    Returns a score adjustment and list of detected patterns.

    Patterns detected:
      Bullish: hammer, bullish engulfing, morning star, pin bar (long lower wick)
      Bearish: shooting star, bearish engulfing, evening star, pin bar (long upper wick)
    """
    result = {"score": 0.0, "patterns": []}
    if len(df) < 5:
        return result

    # Use closed candles (iloc[-2] is last closed, matching signal engine)
    c1 = df.iloc[-2]  # most recent closed
    c2 = df.iloc[-3]  # one before
    c3 = df.iloc[-4]  # two before

    o1, h1, l1, cl1 = float(c1["Open"]), float(c1["High"]), float(c1["Low"]), float(c1["Close"])
    o2, h2, l2, cl2 = float(c2["Open"]), float(c2["High"]), float(c2["Low"]), float(c2["Close"])
    o3, h3, l3, cl3 = float(c3["Open"]), float(c3["High"]), float(c3["Low"]), float(c3["Close"])

    body1 = abs(cl1 - o1)
    body2 = abs(cl2 - o2)
    range1 = h1 - l1 if h1 != l1 else 0.001
    range2 = h2 - l2 if h2 != l2 else 0.001
    upper_wick1 = h1 - max(o1, cl1)
    lower_wick1 = min(o1, cl1) - l1

    # ── Bullish engulfing: previous red candle fully engulfed by current green ──
    if cl2 < o2 and cl1 > o1 and cl1 > o2 and o1 < cl2:
        result["score"] += 1.0
        result["patterns"].append("bullish engulfing")

    # ── Bearish engulfing: previous green candle fully engulfed by current red ──
    if cl2 > o2 and cl1 < o1 and cl1 < o2 and o1 > cl2:
        result["score"] -= 1.0
        result["patterns"].append("bearish engulfing")

    # ── Hammer (bullish): small body at top, long lower wick ≥ 2× body ──
    if body1 > 0 and lower_wick1 >= 2.0 * body1 and upper_wick1 < body1 and cl1 > o1:
        result["score"] += 0.75
        result["patterns"].append("hammer")

    # ── Shooting star (bearish): small body at bottom, long upper wick ≥ 2× body ──
    if body1 > 0 and upper_wick1 >= 2.0 * body1 and lower_wick1 < body1 and cl1 < o1:
        result["score"] -= 0.75
        result["patterns"].append("shooting star")

    # ── Bullish pin bar: very long lower wick ≥ 2/3 of total range ──
    if lower_wick1 >= 0.66 * range1 and cl1 > o1:
        if "hammer" not in result["patterns"]:  # don't double-count
            result["score"] += 0.5
            result["patterns"].append("bullish pin bar")

    # ── Bearish pin bar: very long upper wick ≥ 2/3 of total range ──
    if upper_wick1 >= 0.66 * range1 and cl1 < o1:
        if "shooting star" not in result["patterns"]:
            result["score"] -= 0.5
            result["patterns"].append("bearish pin bar")

    # ── Morning star (3-candle bullish reversal) ──
    # Candle 3: big red, Candle 2: small body (indecision), Candle 1: big green closing above c3 midpoint
    c3_mid = (o3 + cl3) / 2
    small_body2 = body2 < 0.4 * range2  # indecision candle
    if cl3 < o3 and small_body2 and cl1 > o1 and cl1 > c3_mid and body1 > body2 * 2:
        result["score"] += 1.0
        result["patterns"].append("morning star")

    # ── Evening star (3-candle bearish reversal) ──
    if cl3 > o3 and small_body2 and cl1 < o1 and cl1 < c3_mid and body1 > body2 * 2:
        result["score"] -= 1.0
        result["patterns"].append("evening star")

    # ── Doji: tiny body relative to range (indecision) ──
    if body1 < 0.1 * range1:
        result["patterns"].append("doji")
        # Doji doesn't add score by itself — it's a warning of potential reversal

    return result

# ─── Support & Resistance Detection ──────────────────────────────────────────
def detect_sr_levels(df: pd.DataFrame, lookback: int = 80) -> dict:
    """
    Detect support & resistance from recent swing highs/lows.

    A swing high: candle's High is higher than the 2 candles on each side.
    A swing low:  candle's Low is lower than the 2 candles on each side.

    Clusters nearby swings into zones. Returns:
      support    — list of price levels acting as floors
      resistance — list of price levels acting as ceilings
      nearest_support    — closest support below current price (or None)
      nearest_resistance — closest resistance above current price (or None)
      at_support    — True if price is within 0.3% of a support level
      at_resistance — True if price is within 0.3% of a resistance level
    """
    result = {
        "support": [], "resistance": [],
        "nearest_support": None, "nearest_resistance": None,
        "at_support": False, "at_resistance": False,
    }
    if len(df) < lookback:
        lookback = len(df) - 4
    if lookback < 10:
        return result

    highs = df["High"].values
    lows  = df["Low"].values
    price = float(df["Close"].iloc[-2])  # match signal engine's closed candle

    swing_highs = []
    swing_lows  = []

    start = len(df) - lookback
    for i in range(start + 2, len(df) - 2):
        # Swing high: higher than 2 neighbors on each side
        if (highs[i] > highs[i-1] and highs[i] > highs[i-2]
                and highs[i] > highs[i+1] and highs[i] > highs[i+2]):
            swing_highs.append(float(highs[i]))
        # Swing low: lower than 2 neighbors on each side
        if (lows[i] < lows[i-1] and lows[i] < lows[i-2]
                and lows[i] < lows[i+1] and lows[i] < lows[i+2]):
            swing_lows.append(float(lows[i]))

    # Cluster nearby levels (within 0.15% of each other)
    def cluster(levels, threshold_pct=0.0015):
        if not levels:
            return []
        levels = sorted(levels)
        clusters = []
        group = [levels[0]]
        for lv in levels[1:]:
            if abs(lv - group[0]) / group[0] < threshold_pct:
                group.append(lv)
            else:
                clusters.append(sum(group) / len(group))
                group = [lv]
        clusters.append(sum(group) / len(group))
        return clusters

    result["resistance"] = cluster(swing_highs)
    result["support"]    = cluster(swing_lows)

    # Find nearest levels
    supports_below = [s for s in result["support"] if s < price]
    resist_above   = [r for r in result["resistance"] if r > price]

    if supports_below:
        result["nearest_support"] = max(supports_below)
    if resist_above:
        result["nearest_resistance"] = min(resist_above)

    # Check if price is near S/R (within 0.3%)
    threshold = price * 0.003
    if result["nearest_support"] and abs(price - result["nearest_support"]) < threshold:
        result["at_support"] = True
    if result["nearest_resistance"] and abs(price - result["nearest_resistance"]) < threshold:
        result["at_resistance"] = True

    return result

# ─── Signal Engine ────────────────────────────────────────────────────────────
def generate_signal(df: pd.DataFrame, symbol: str, ns: dict, htf_bias_15m: int = 0, htf_bias_1h: int = 0) -> dict:
    empty = {"direction": "NEUTRAL", "score": 0, "reasons": [],
             "entry": None, "sl": None, "tp1": None, "tp2": None, "atr": 0}
    if len(df) < 50:
        return empty

    last, prev = df.iloc[-2], df.iloc[-3]  # use last CLOSED candle, not the live open one
    score = 0.0

    # EMA stack
    if   last["EMA9"] > last["EMA21"] > last["EMA50"]: score += 2
    elif last["EMA9"] < last["EMA21"] < last["EMA50"]: score -= 2
    elif last["EMA9"] > last["EMA21"]:                  score += 1
    else:                                               score -= 1

    # Fresh crossover
    if   prev["EMA9"] <= prev["EMA21"] and last["EMA9"] > last["EMA21"]: score += 1
    elif prev["EMA9"] >= prev["EMA21"] and last["EMA9"] < last["EMA21"]: score -= 1

    # RSI
    rsi = float(last["RSI"])
    if   rsi < 35:          score += 1
    elif rsi > 65:          score -= 1
    elif 48 < rsi < 62:     score += 0.5
    elif 38 < rsi < 52:     score -= 0.5

    # MACD
    if   last["MACD"] > last["MACD_signal"] and float(last["MACD_hist"]) > float(prev["MACD_hist"]): score += 1
    elif last["MACD"] < last["MACD_signal"] and float(last["MACD_hist"]) < float(prev["MACD_hist"]): score -= 1

    # VWAP
    if float(last["Close"]) > float(last["VWAP"]): score += 0.5
    else:                                           score -= 0.5

    # Bollinger
    if   float(last["Close"]) < float(last["BB_lower"]): score += 0.5
    elif float(last["Close"]) > float(last["BB_upper"]): score -= 0.5

    # News
    score += ns.get("adjustment", 0)

    # 1h trend filter (macro — highest weight)
    if htf_bias_1h == 1:
        if score > 0:   score += 1.5  # aligned LONG
        elif score < 0: score += 2.0  # counter-trend SHORT heavily penalized
    elif htf_bias_1h == -1:
        if score < 0:   score -= 1.5  # aligned SHORT
        elif score > 0: score -= 2.0  # counter-trend LONG heavily penalized

    # 15m trend filter (intermediate)
    if htf_bias_15m == 1:
        if score > 0:   score += 1.0  # aligned LONG
        elif score < 0: score += 1.5  # counter-trend SHORT weakened
    elif htf_bias_15m == -1:
        if score < 0:   score -= 1.0  # aligned SHORT
        elif score > 0: score -= 1.5  # counter-trend LONG weakened

    # ── Fair Value Gap scoring ────────────────────────────────────────────────
    fvg = detect_fvg(df)
    if fvg["in_bullish_fvg"]:
        score += 1.5
    elif fvg["in_bearish_fvg"]:
        score -= 1.5
    else:
        nb = fvg["nearest_bull"]
        nd = fvg["nearest_bear"]
        price = float(last["Close"])
        if nb and nb["bottom"] < price:
            score += 0.5
        elif nd and nd["top"] > price:
            score -= 0.5

    # ── Candlestick pattern scoring ──────────────────────────────────────────
    candles = detect_candle_patterns(df)
    score += candles["score"]

    # ── Support & Resistance scoring ─────────────────────────────────────────
    sr = detect_sr_levels(df)
    if sr["at_support"]:
        score += 1.0   # price bouncing off support = bullish
    if sr["at_resistance"]:
        score -= 1.0   # price hitting resistance = bearish
    # Reward trades with clear room to run (nearest target not blocked by S/R)
    if score > 0 and sr["nearest_resistance"]:
        room = sr["nearest_resistance"] - float(last["Close"])
        atr_val = float(last["ATR"]) if pd.notna(last["ATR"]) else 1.0
        if room < 0.5 * atr_val:
            score -= 0.75  # resistance too close for a LONG to work
    if score < 0 and sr["nearest_support"]:
        room = float(last["Close"]) - sr["nearest_support"]
        atr_val = float(last["ATR"]) if pd.notna(last["ATR"]) else 1.0
        if room < 0.5 * atr_val:
            score += 0.75  # support too close for a SHORT to work

    score  = round(score, 2)

    if   score >= 2.5: direction = "LONG"
    elif score <= -2.5: direction = "SHORT"
    else: return {**empty, "score": score, "fvg": fvg}

    atr   = float(last["ATR"]) if pd.notna(last["ATR"]) else 1.0
    price = float(last["Close"])
    tick  = TICK_INFO.get(symbol, {}).get("tick", 0.25)

    # Adaptive SL multiplier from config
    sl_mult = float(load_config().get("sl_multipliers", {}).get(symbol, 1.5))
    sl_mult = max(1.0, min(2.5, sl_mult))

    if direction == "LONG":
        entry = _snap(price, tick)
        sl    = _snap(price - sl_mult * atr, tick)
        tp1   = _snap(price + sl_mult * atr, tick)
        tp2   = _snap(price + sl_mult * 2 * atr, tick)
    else:
        entry = _snap(price, tick)
        sl    = _snap(price + sl_mult * atr, tick)
        tp1   = _snap(price - sl_mult * atr, tick)
        tp2   = _snap(price - sl_mult * 2 * atr, tick)

    return {"direction": direction, "score": score, "entry": entry,
            "sl": sl, "tp1": tp1, "tp2": tp2, "atr": atr, "reasons": [], "fvg": fvg}

# ─── Session Gate ─────────────────────────────────────────────────────────────
def trading_session_active(symbol: str) -> tuple:
    """
    MGC trades nearly 24h Sun-Fri, with two blackout windows:
      - Daily maintenance: 2:00 PM – 3:00 PM PT (5-6 PM ET) every weekday
      - Weekend: Saturday all day + Sunday before 3:00 PM PT
    During blackouts the bot must NOT generate signals or check TP/SL,
    because yfinance returns stale pre-close data that causes false closes.
    """
    now     = now_pt()
    weekday = now.weekday()
    h       = now.hour + now.minute / 60.0
    # Weekend
    if weekday == 5:
        return False, "Market closed — Saturday", "3:00 PM PT Sunday"
    if weekday == 6 and h < 15.0:
        return False, "Market closed — Sunday pre-open", "3:00 PM PT"
    # Daily maintenance window: 2:00 PM – 3:00 PM PT (CME 5-6 PM ET)
    if 14.0 <= h < 15.0:
        return False, "Daily maintenance — market closed", "3:00 PM PT"
    # Buffer after reopen: skip first 5 min so fresh candles populate
    if 15.0 <= h < 15.083:  # 3:00 - 3:05 PM PT
        return False, "Market reopening — waiting for fresh data", "3:05 PM PT"
    return True, "Gold futures active 24h", ""

# ─── TP/SL Checker ────────────────────────────────────────────────────────────
def check_open_trades(symbol: str, df: pd.DataFrame):
    trades      = load_trades()
    open_trades = [t for t in trades if t["status"] == "open" and t["symbol"] == symbol]
    if not open_trades:
        return

    # Always use 1m candles for TP/SL — catches every wick, most precise
    try:
        df_1m = fetch_data(symbol, "1m", "7d")
        if not df_1m.empty:
            df = df_1m
    except Exception:
        pass

    if df.empty:
        return

    ti_sz = TICK_INFO[symbol]["tick"]

    for trade in open_trades:
        try:
            sig_time = datetime.fromisoformat(trade["timestamp"])
            if sig_time.tzinfo is None:
                sig_time = sig_time.replace(tzinfo=PT)
            if df.index.tz is not None:
                sig_time = sig_time.astimezone(df.index.tz)
                after = df[df.index > sig_time]
            else:
                sig_utc = sig_time.astimezone(ZoneInfo("UTC")).replace(tzinfo=None)
                after   = df[df.index > sig_utc]
        except Exception:
            continue

        after = after.iloc[:-1]  # drop last incomplete candle — its wick isn't final

        if after.empty:
            continue

        d, sl, tp1, tp2, entry = (trade["direction"], trade["sl"],
                                   trade["tp1"], trade["tp2"], trade["entry"])

        for idx, candle in after.iterrows():
            hi, lo = float(candle["High"]), float(candle["Low"])
            if d == "LONG":
                tp2_hit = hi >= tp2
                tp1_hit = hi >= tp1
                sl_hit  = lo <= sl
                if tp2_hit:
                    trade.update(status="win_tp2", pnl_ticks=round(abs(tp2-entry)/ti_sz,1), closed_at=idx.isoformat())
                elif tp1_hit:
                    trade.update(status="win_tp1", pnl_ticks=round(abs(tp1-entry)/ti_sz,1), closed_at=idx.isoformat())
                elif sl_hit:
                    trade.update(status="loss",    pnl_ticks=round(-abs(entry-sl)/ti_sz,1), closed_at=idx.isoformat())
                else:
                    continue
            else:
                tp2_hit = lo <= tp2
                tp1_hit = lo <= tp1
                sl_hit  = hi >= sl
                if tp2_hit:
                    trade.update(status="win_tp2", pnl_ticks=round(abs(entry-tp2)/ti_sz,1), closed_at=idx.isoformat())
                elif tp1_hit:
                    trade.update(status="win_tp1", pnl_ticks=round(abs(entry-tp1)/ti_sz,1), closed_at=idx.isoformat())
                elif sl_hit:
                    trade.update(status="loss",    pnl_ticks=round(-abs(sl-entry)/ti_sz,1), closed_at=idx.isoformat())
                else:
                    continue

            save_single_trade(trade)
            log.info(f"Closed {d} {symbol} → {trade['status']} ({trade['pnl_ticks']:+.1f} ticks)")
            break

# ─── Signal Recording ─────────────────────────────────────────────────────────
def should_record(signal: dict, symbol: str) -> bool:
    if signal["direction"] == "NEUTRAL":
        return False
    if abs(signal.get("score", 0)) < _MIN_SCORE.get(symbol, 2.5):
        return False
    # Staleness gate — reject if live price has moved too far from entry
    if not _entry_is_fresh(signal, symbol):
        return False
    trades     = load_trades()
    sym_trades = [t for t in trades if t["symbol"] == symbol]
    # Block if trade already open
    if any(t["status"] == "open" for t in sym_trades):
        return False
    if not sym_trades:
        return True
    last = sym_trades[-1]
    try:
        last_time = datetime.fromisoformat(last["timestamp"])
        if last_time.tzinfo is None:
            last_time = last_time.replace(tzinfo=PT)
        elapsed = (now_pt() - last_time).total_seconds()
        # Hard cooldown — never record within 20 min of last trade
        if elapsed < 1200:
            return False
        # Direction flip cooldown — wait 45 min before reversing
        if elapsed < 2700 and last.get("direction") != signal["direction"]:
            log.info(f"[cooldown] {symbol} direction flip blocked — only {elapsed/60:.0f}m since last trade")
            return False
        # Same direction + similar score within 60 min = duplicate
        if elapsed < 3600:
            if (last.get("direction") == signal["direction"] and
                abs(last.get("score", 0) - signal.get("score", 0)) < 1.0):
                return False
        # Same entry price within 2 hours = stale repeat
        if elapsed < 7200:
            tick = TICK_INFO.get(symbol, {}).get("tick", 0.10)
            if abs(float(last.get("entry", 0)) - float(signal.get("entry", 0))) / tick < 15:
                log.info(f"[cooldown] {symbol} similar entry blocked — {elapsed/60:.0f}m, entry diff < 15 ticks")
                return False
    except Exception:
        pass
    return True

def _verify_and_correct_signal(signal: dict, symbol: str) -> dict | None:
    """
    Cross-check signal entry/TP/SL against live 1m price.
    - Rewrites entry to actual live price (not stale 5m candle)
    - Recalculates TP/SL from that corrected entry
    - Rejects if the direction no longer makes sense at live price
    Returns corrected signal or None if it fails verification.
    """
    live_price = _get_live_price(symbol)
    if live_price is None:
        log.warning(f"[verify] {symbol} — can't fetch live price, rejecting signal")
        return None

    tick = TICK_INFO[symbol]["tick"]
    old_entry = float(signal["entry"])
    drift_ticks = abs(old_entry - live_price) / tick

    # If entry is more than 10 ticks from live, rewrite entry to live price
    if drift_ticks > 10:
        log.info(f"[verify] {symbol} correcting entry {old_entry:.2f} → {live_price:.2f} (drift={drift_ticks:.0f} ticks)")

    # Use live price as the real entry
    entry = _snap(live_price, tick)
    atr = float(signal.get("atr", 1.0))
    if atr < tick:
        atr = tick * 10  # safety floor

    sl_mult = float(load_config().get("sl_multipliers", {}).get(symbol, 1.5))
    sl_mult = max(1.0, min(2.5, sl_mult))

    if signal["direction"] == "LONG":
        sl  = _snap(entry - sl_mult * atr, tick)
        tp1 = _snap(entry + sl_mult * atr, tick)
        tp2 = _snap(entry + sl_mult * 2 * atr, tick)
        # Sanity: SL must be below entry, TPs above
        if sl >= entry or tp1 <= entry:
            log.warning(f"[verify] {symbol} LONG failed sanity — sl={sl} entry={entry} tp1={tp1}")
            return None
    else:
        sl  = _snap(entry + sl_mult * atr, tick)
        tp1 = _snap(entry - sl_mult * atr, tick)
        tp2 = _snap(entry - sl_mult * 2 * atr, tick)
        if sl <= entry or tp1 >= entry:
            log.warning(f"[verify] {symbol} SHORT failed sanity — sl={sl} entry={entry} tp1={tp1}")
            return None

    # Reward/risk ratio check — TP1 must be at least 0.8× the SL distance
    risk = abs(entry - sl)
    reward = abs(tp1 - entry)
    if risk == 0 or reward / risk < 0.8:
        log.warning(f"[verify] {symbol} bad R:R — risk={risk:.2f} reward={reward:.2f}")
        return None

    signal = dict(signal)
    signal.update(entry=entry, sl=sl, tp1=tp1, tp2=tp2)
    log.info(f"[verify] {symbol} PASS — entry={entry:.2f} sl={sl:.2f} tp1={tp1:.2f} tp2={tp2:.2f}")
    return signal

def record_signal(signal: dict, symbol: str):
    # Final verification — correct entry to live price, sanity-check TP/SL
    verified = _verify_and_correct_signal(signal, symbol)
    if verified is None:
        log.warning(f"[record] {symbol} signal FAILED verification — not recording")
        return
    signal = verified

    trade = {
        "id":        str(uuid.uuid4())[:8],
        "symbol":    symbol,
        "name":      TICK_INFO[symbol]["name"],
        "interval":  "5m",
        "direction": signal["direction"],
        "entry":     float(signal["entry"]),
        "sl":        float(signal["sl"]),
        "tp1":       float(signal["tp1"]),
        "tp2":       float(signal["tp2"]),
        "score":     signal["score"],
        "reasons":   signal.get("reasons", []),
        "timestamp": now_pt().isoformat(),
        "status":    "open",
        "closed_at": None,
        "pnl_ticks": None,
    }
    save_single_trade(trade)
    log.info(f"Recorded {signal['direction']} {symbol} score={signal['score']:+.1f}")
    send_notification(symbol, trade)

# ─── Main Loop ────────────────────────────────────────────────────────────────
def _has_open_trade(symbol: str) -> bool:
    """Check Supabase for any open trade on this symbol."""
    trades = load_trades()
    return any(t["status"] == "open" and t["symbol"] == symbol for t in trades)

def run_once():
    log.info("── tick ──")

    # ── Session gate — pause during maintenance & weekends ───────────────
    # Check once for all symbols (they share the same CME schedule)
    active, reason, reopens = trading_session_active("MGC=F")
    if not active:
        log.info(f"PAUSED — {reason} (reopens {reopens})")
        return

    articles = fetch_news()
    for symbol in ACTIVE_SYMBOLS:
        time.sleep(15)  # avoid yfinance rate limiting
        try:
            # ── If a trade is open: ONLY check TP/SL, skip everything else ──
            if _has_open_trade(symbol):
                log.info(f"{TICK_INFO[symbol]['name']}  IN TRADE — checking TP/SL only")
                df = fetch_data(symbol, "5m", "60d")
                if not df.empty:
                    check_open_trades(symbol, df)
                continue

            # ── No open trade: generate signals ─────────────────────────────
            df = fetch_data(symbol, "5m", "60d")
            if df.empty:
                log.warning(f"{symbol}: no data")
                continue
            df = compute_indicators(df)

            # Higher timeframe confirmation — 15m (intermediate) + 1h (macro)
            time.sleep(5)
            df_15m       = fetch_data(symbol, "15m", "60d")
            df_15m       = compute_indicators(df_15m) if not df_15m.empty else df_15m
            htf_bias_15m = get_htf_bias(df_15m) if not df_15m.empty else 0

            time.sleep(5)
            df_1h        = fetch_data(symbol, "1h", "730d")
            df_1h        = compute_indicators(df_1h) if not df_1h.empty else df_1h
            htf_bias_1h  = get_htf_bias(df_1h) if not df_1h.empty else 0

            ns  = get_news_sentiment(symbol, articles)
            sig = generate_signal(df, symbol, ns, htf_bias_15m=htf_bias_15m, htf_bias_1h=htf_bias_1h)

            def _bl(b): return "BUL" if b == 1 else "BEA" if b == -1 else "NEU"
            log.info(f"{TICK_INFO[symbol]['name']}  {sig['direction']:7}  score={sig['score']:+.2f}  1h={_bl(htf_bias_1h)}  15m={_bl(htf_bias_15m)}")

            if should_record(sig, symbol):
                record_signal(sig, symbol)

        except Exception as e:
            log.error(f"{symbol}: {e}")

if __name__ == "__main__":
    log.info("Worker started — interval %ds", CHECK_INTERVAL_SEC)
    while True:
        try:
            run_once()
        except Exception as e:
            log.error(f"run_once: {e}")
        time.sleep(CHECK_INTERVAL_SEC)
