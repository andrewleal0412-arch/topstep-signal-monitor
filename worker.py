"""
TopStep Signal Worker — runs 24/7 independent of the Streamlit app.
Checks all 6 symbols every 5 minutes, records signals to Supabase,
sends ntfy push notifications when strong setups fire.
"""

import os, time, uuid, json, logging, requests, feedparser, re
from datetime import datetime, timedelta
from zoneinfo import ZoneInfo

import yfinance as yf
import pandas as pd
import ta
from supabase import create_client
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

# ─── Config ───────────────────────────────────────────────────────────────────
PT      = ZoneInfo("America/Los_Angeles")
_vader  = SentimentIntensityAnalyzer()

SUPABASE_URL = os.environ["SUPABASE_URL"]
SUPABASE_KEY = os.environ["SUPABASE_KEY"]
NTFY_TOPIC   = os.environ.get("NTFY_TOPIC", "topstepdraco42")
MIN_SCORE    = float(os.environ.get("MIN_SCORE", "3.0"))
CHECK_INTERVAL_SEC = int(os.environ.get("CHECK_INTERVAL_SEC", "300"))  # 5 min

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)s  %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S"
)
log = logging.getLogger("topstep-worker")

TICK_INFO = {
    "MNQ=F": {"tick": 0.25, "value": 0.50, "name": "MNQ"},
    "MES=F": {"tick": 0.25, "value": 1.25, "name": "MES"},
    "MGC=F": {"tick": 0.10, "value": 1.00, "name": "MGC"},
}

SYMBOL_GROUPS = {
    "MNQ=F": ["nasdaq", "macro"], "NQ=F":  ["nasdaq", "macro"],
    "MES=F": ["sp500",  "macro"], "ES=F":  ["sp500",  "macro"],
    "GC=F":  ["gold",   "macro"], "MGC=F": ["gold",   "macro"],
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
    ("Yahoo Finance",  "https://finance.yahoo.com/news/rssindex"),
    ("Reuters",        "https://feeds.reuters.com/reuters/businessNews"),
    ("CNBC",           "https://www.cnbc.com/id/100003114/device/rss/rss.html"),
    ("MarketWatch",    "https://feeds.content.dowjones.io/public/rss/mw_realtimeheadlines"),
    ("Investing.com",  "https://www.investing.com/rss/news.rss"),
]

# ─── Helpers ──────────────────────────────────────────────────────────────────
def now_pt():
    return datetime.now(PT)

def db():
    return create_client(SUPABASE_URL, SUPABASE_KEY)

# ─── Supabase persistence ─────────────────────────────────────────────────────
def load_trades() -> list:
    try:
        res = db().table("trades").select("data").order("created_at").execute()
        return [r["data"] for r in res.data] if res.data else []
    except Exception as e:
        log.error(f"load_trades error: {e}")
        return []

def save_single_trade(trade: dict):
    try:
        db().table("trades").upsert({"id": trade["id"], "data": trade}).execute()
    except Exception as e:
        log.error(f"save_single_trade error: {e}")

def load_config() -> dict:
    try:
        res = db().table("config").select("data").eq("id", "default").execute()
        if res.data:
            return res.data[0]["data"]
    except Exception as e:
        log.error(f"load_config error: {e}")
    return {"ntfy_topic": NTFY_TOPIC, "notify_enabled": True, "min_score": MIN_SCORE}

# ─── Notifications ────────────────────────────────────────────────────────────
def send_notification(symbol: str, signal: dict):
    cfg = load_config()
    topic = cfg.get("ntfy_topic", NTFY_TOPIC).strip()
    if not topic:
        return
    min_sc = float(cfg.get("min_score", MIN_SCORE))
    if abs(signal["score"]) < min_sc:
        return

    ti       = TICK_INFO[symbol]
    d        = signal["direction"]
    name     = ti["name"]
    score    = signal["score"]
    strength = int(abs(score) / 6.0 * 100)
    tick_sz  = ti["tick"]
    sl_ticks = abs(signal["entry"] - signal["sl"]) / tick_sz
    tp1_ticks= abs(signal["entry"] - signal["tp1"]) / tick_sz

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
        log.info(f"Notification sent: {title}")
    except Exception as e:
        log.error(f"Notification failed: {e}")

# ─── News ─────────────────────────────────────────────────────────────────────
_news_cache = {"articles": [], "fetched_at": None}

def fetch_news() -> list:
    now = now_pt()
    if _news_cache["fetched_at"] and (now - _news_cache["fetched_at"]).total_seconds() < 300:
        return _news_cache["articles"]

    articles, seen = [], set()
    for source, url in NEWS_FEEDS:
        try:
            feed = feedparser.parse(url)
            for entry in feed.entries[:15]:
                title   = entry.get("title", "").strip()
                summary = re.sub(r"<[^>]+>", " ", entry.get("summary", ""))
                summary = re.sub(r"\s+", " ", summary).strip()[:300]
                if not title or title in seen:
                    continue
                seen.add(title)
                text      = f"{title}. {summary}"
                compound  = _vader.polarity_scores(text)["compound"]
                low_text  = text.lower()
                groups    = [g for g, kws in INSTRUMENT_KEYWORDS.items()
                             if any(k in low_text for k in kws)]
                high_impact = any(k in low_text for k in
                                  ["fed","fomc","cpi","nfp","gdp","rate","inflation",
                                   "jobs","recession","earnings","tariff"])
                articles.append({
                    "title": title, "summary": summary, "source": source,
                    "compound": compound, "groups": groups, "high_impact": high_impact,
                })
        except Exception:
            pass

    _news_cache["articles"]   = articles
    _news_cache["fetched_at"] = now
    return articles

def get_news_sentiment(symbol: str, articles: list) -> dict:
    sym_groups = SYMBOL_GROUPS.get(symbol, [])
    relevant   = [a for a in articles
                  if any(g in a["groups"] for g in sym_groups) or a["high_impact"]]
    if not relevant:
        return {"score": 0, "adjustment": 0, "label": "Neutral", "count": 0}
    avg = sum(a["compound"] for a in relevant) / len(relevant)
    adj = max(-1.5, min(1.5, avg * 3))
    return {"score": avg, "adjustment": adj, "label": "Positive" if avg > 0.05 else "Negative" if avg < -0.05 else "Neutral", "count": len(relevant)}

# ─── Data & Indicators ────────────────────────────────────────────────────────
def fetch_data(symbol: str, interval: str = "5m", period: str = "60d") -> pd.DataFrame:
    try:
        df = yf.download(symbol, period=period, interval=interval, progress=False, auto_adjust=True)
        if df.empty:
            return pd.DataFrame()
        df.columns = [c[0] if isinstance(c, tuple) else c for c in df.columns]
        return df.dropna()
    except Exception as e:
        log.error(f"fetch_data {symbol}: {e}")
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
    df["MACD"] = macd.macd(); df["MACD_signal"] = macd.macd_signal(); df["MACD_hist"] = macd.macd_diff()
    bb = ta.volatility.BollingerBands(close, window=20, window_dev=2)
    df["BB_upper"] = bb.bollinger_hband(); df["BB_lower"] = bb.bollinger_lband()
    df["ATR"]  = ta.volatility.average_true_range(high, low, close, window=14)
    df["VWAP"] = (close * df["Volume"]).cumsum() / df["Volume"].cumsum()
    return df

# ─── Signal Engine ────────────────────────────────────────────────────────────
def generate_signal(df: pd.DataFrame, symbol: str, news_sentiment: dict) -> dict:
    empty = {"direction": "NEUTRAL", "score": 0, "reasons": [],
             "entry": None, "sl": None, "tp1": None, "tp2": None, "atr": 0}
    if len(df) < 50:
        return empty

    last, prev = df.iloc[-1], df.iloc[-2]
    score = 0

    if last["EMA9"] > last["EMA21"] > last["EMA50"]:
        score += 2
    elif last["EMA9"] < last["EMA21"] < last["EMA50"]:
        score -= 2
    elif last["EMA9"] > last["EMA21"]:
        score += 1
    else:
        score -= 1

    if prev["EMA9"] <= prev["EMA21"] and last["EMA9"] > last["EMA21"]:
        score += 1
    elif prev["EMA9"] >= prev["EMA21"] and last["EMA9"] < last["EMA21"]:
        score -= 1

    rsi = float(last["RSI"])
    if rsi < 35:
        score += 1
    elif rsi > 65:
        score -= 1
    elif 48 < rsi < 62:
        score += 0.5
    elif 38 < rsi < 52:
        score -= 0.5

    if last["MACD"] > last["MACD_signal"] and float(last["MACD_hist"]) > float(prev["MACD_hist"]):
        score += 1
    elif last["MACD"] < last["MACD_signal"] and float(last["MACD_hist"]) < float(prev["MACD_hist"]):
        score -= 1

    if float(last["Close"]) > float(last["VWAP"]):
        score += 0.5
    else:
        score -= 0.5

    if float(last["Close"]) < float(last["BB_lower"]):
        score += 0.5
    elif float(last["Close"]) > float(last["BB_upper"]):
        score -= 0.5

    score += news_sentiment.get("adjustment", 0)

    atr   = float(last["ATR"]) if pd.notna(last["ATR"]) else 1.0
    entry = float(last["Close"])

    if score >= 2.5:
        direction = "LONG"
        sl  = entry - atr * 1.5
        tp1 = entry + atr * 1.5
        tp2 = entry + atr * 3.0
    elif score <= -2.5:
        direction = "SHORT"
        sl  = entry + atr * 1.5
        tp1 = entry - atr * 1.5
        tp2 = entry - atr * 3.0
    else:
        return {**empty, "score": round(score, 2)}

    return {
        "direction": direction,
        "score":     round(score, 2),
        "entry":     round(entry, 2),
        "sl":        round(sl, 2),
        "tp1":       round(tp1, 2),
        "tp2":       round(tp2, 2),
        "atr":       round(atr, 2),
        "reasons":   [],
    }

# ─── Trading Session Gate ─────────────────────────────────────────────────────
def trading_session_active(symbol: str) -> bool:
    now     = now_pt()
    weekday = now.weekday()
    h       = now.hour + now.minute / 60.0

    if weekday == 5:                       # Saturday — closed
        return False
    if weekday == 6 and h < 15.0:         # Sunday before 3 PM PT
        return False

    if symbol in ("MNQ=F", "MES=F"):
        if 6.5 <= h < 8.5:   return True  # NYSE open
        if 11.0 <= h < 13.0: return True  # power hour / close
        return False                       # midday chop or overnight

    if symbol == "MGC=F":
        if h < 2.0:          return True  # London session
        if 5.0 <= h < 9.0:  return True  # COMEX / NY morning
        return False

    return True

# ─── Trade logic ─────────────────────────────────────────────────────────────
def should_record(signal: dict, symbol: str) -> bool:
    if signal["direction"] == "NEUTRAL":
        return False
    if not trading_session_active(symbol):
        return False
    trades     = load_trades()
    sym_trades = [t for t in trades if t["symbol"] == symbol]
    # Max 1 open trade per symbol at a time
    if any(t["status"] == "open" for t in sym_trades):
        return False
    if not sym_trades:
        return True
    last = sym_trades[-1]
    if last["direction"] == signal["direction"]:
        return False
    try:
        last_time = datetime.fromisoformat(last["timestamp"])
        if last_time.tzinfo is None:
            last_time = last_time.replace(tzinfo=PT)
        if (now_pt() - last_time).total_seconds() < 900:
            return False
    except Exception:
        pass
    return True

def record_signal(signal: dict, symbol: str) -> dict:
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
    log.info(f"Recorded {signal['direction']} on {symbol} | score {signal['score']:+.1f}")
    # Notify with exact saved trade values so TP numbers match the log
    send_notification(symbol, trade)
    return trade

def check_open_trades(symbol: str, df: pd.DataFrame):
    trades      = load_trades()
    open_trades = [t for t in trades if t["status"] == "open" and t["symbol"] == symbol]
    if not open_trades or df.empty:
        return

    ti_sz = TICK_INFO[symbol]["tick"]
    for trade in open_trades:
        try:
            sig_time = datetime.fromisoformat(trade["timestamp"])
            if sig_time.tzinfo is None:
                sig_time = sig_time.replace(tzinfo=PT)
            sig_time = sig_time.astimezone(df.index.tz) if df.index.tz else sig_time.replace(tzinfo=None)
            after = df[df.index > sig_time]
        except Exception:
            continue

        if after.empty:
            continue

        d     = trade["direction"]
        sl    = trade["sl"]
        tp1   = trade["tp1"]
        tp2   = trade["tp2"]
        entry = trade["entry"]

        for idx, candle in after.iterrows():
            hi, lo = float(candle["High"]), float(candle["Low"])
            if d == "LONG":
                sl_hit, tp1_hit, tp2_hit = lo <= sl, hi >= tp1, hi >= tp2
                if sl_hit and tp1_hit:
                    trade.update(status="loss", pnl_ticks=round(-abs(entry-sl)/ti_sz,1), closed_at=idx.isoformat())
                elif sl_hit:
                    trade.update(status="loss", pnl_ticks=round(-abs(entry-sl)/ti_sz,1), closed_at=idx.isoformat())
                elif tp2_hit:
                    trade.update(status="win_tp2", pnl_ticks=round(abs(tp2-entry)/ti_sz,1), closed_at=idx.isoformat())
                elif tp1_hit:
                    trade.update(status="win_tp1", pnl_ticks=round(abs(tp1-entry)/ti_sz,1), closed_at=idx.isoformat())
                else:
                    continue
            else:
                sl_hit, tp1_hit, tp2_hit = hi >= sl, lo <= tp1, lo <= tp2
                if sl_hit and tp1_hit:
                    trade.update(status="loss", pnl_ticks=round(-abs(sl-entry)/ti_sz,1), closed_at=idx.isoformat())
                elif sl_hit:
                    trade.update(status="loss", pnl_ticks=round(-abs(sl-entry)/ti_sz,1), closed_at=idx.isoformat())
                elif tp2_hit:
                    trade.update(status="win_tp2", pnl_ticks=round(abs(entry-tp2)/ti_sz,1), closed_at=idx.isoformat())
                elif tp1_hit:
                    trade.update(status="win_tp1", pnl_ticks=round(abs(entry-tp1)/ti_sz,1), closed_at=idx.isoformat())
                else:
                    continue

            save_single_trade(trade)
            result = trade["status"]
            log.info(f"Closed {trade['direction']} {symbol} → {result} ({trade['pnl_ticks']:+.1f} ticks)")
            break

# ─── Notification cooldown (in-memory) ───────────────────────────────────────
_last_notif: dict = {}

def maybe_notify(symbol: str, signal: dict):
    last = _last_notif.get(symbol)
    if last and (now_pt() - last).total_seconds() < 900:
        return
    send_notification(symbol, signal)
    _last_notif[symbol] = now_pt()

# ─── Main loop ────────────────────────────────────────────────────────────────
SYMBOLS   = list(TICK_INFO.keys())
INTERVAL  = "5m"
PERIOD    = "60d"

def run_once():
    log.info("── Checking all symbols ──")
    articles = fetch_news()
    for symbol in SYMBOLS:
        time.sleep(8)  # avoid Yahoo Finance rate limiting
        try:
            df = fetch_data(symbol, INTERVAL, PERIOD)
            if df.empty:
                log.warning(f"{symbol}: no data")
                continue
            df  = compute_indicators(df)
            ns  = get_news_sentiment(symbol, articles)
            sig = generate_signal(df, symbol, ns)

            sess = trading_session_active(symbol)
            log.info(f"{TICK_INFO[symbol]['name']:5s}  {sig['direction']:7s}  score {sig['score']:+.2f}  session={'ON' if sess else 'OFF'}")

            check_open_trades(symbol, df)

            if should_record(sig, symbol):
                record_signal(sig, symbol)

        except Exception as e:
            log.error(f"{symbol} error: {e}")

if __name__ == "__main__":
    log.info("TopStep worker started — checking every %ds", CHECK_INTERVAL_SEC)
    while True:
        try:
            run_once()
        except Exception as e:
            log.error(f"run_once error: {e}")
        log.info(f"Sleeping {CHECK_INTERVAL_SEC}s...")
        time.sleep(CHECK_INTERVAL_SEC)
