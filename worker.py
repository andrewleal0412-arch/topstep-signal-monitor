"""
Signal Worker — runs 24/7 independent of the Streamlit app.
Checks MNQ, MES, MGC every 5 minutes, closes open trades when TP/SL hit,
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
NTFY_TOPIC         = os.environ.get("NTFY_TOPIC", "")
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

# MGC trades 24h — no session gate
_SESSION_GATE_EXEMPT = {"MGC=F"}

# Minimum score to record a signal per symbol (tuned from trade log)
_MIN_SCORE = {
    "MNQ=F": 3.5,
    "MES=F": 2.5,
    "MGC=F": 3.0,
}

_MAX_PERIOD = {
    "1m": "7d", "2m": "60d", "5m": "60d",
    "15m": "60d", "30m": "60d", "1h": "730d",
}

SYMBOL_GROUPS = {
    "MNQ=F": ["nasdaq", "macro"],
    "MES=F": ["sp500",  "macro"],
    "MGC=F": ["gold",   "macro"],
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
    return {"ntfy_topic": NTFY_TOPIC, "notify_enabled": True, "min_score": 3.5}

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
    strength = int(abs(trade["score"]) / 6.0 * 100)
    sl_ticks = abs(trade["entry"] - trade["sl"]) / ti["tick"]
    tp1_ticks= abs(trade["entry"] - trade["tp1"]) / ti["tick"]

    title = f"{d} — {ti['name']} | Score {trade['score']:+.1f} ({strength}% strength)"
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

# ─── Signal Engine ────────────────────────────────────────────────────────────
def generate_signal(df: pd.DataFrame, symbol: str, ns: dict, htf_bias: int = 0) -> dict:
    empty = {"direction": "NEUTRAL", "score": 0, "reasons": [],
             "entry": None, "sl": None, "tp1": None, "tp2": None, "atr": 0}
    if len(df) < 50:
        return empty

    last, prev = df.iloc[-1], df.iloc[-2]
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

    # Higher timeframe (15m) trend filter
    if htf_bias == 1:
        if score > 0:  score += 1.0   # aligned LONG
        elif score < 0: score += 1.5  # counter-trend SHORT weakened
    elif htf_bias == -1:
        if score < 0:  score -= 1.0   # aligned SHORT
        elif score > 0: score -= 1.5  # counter-trend LONG weakened

    score  = round(score, 2)

    if   score >= 2.5: direction = "LONG"
    elif score <= -2.5: direction = "SHORT"
    else: return {**empty, "score": score}

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
            "sl": sl, "tp1": tp1, "tp2": tp2, "atr": atr, "reasons": []}

# ─── Session Gate ─────────────────────────────────────────────────────────────
def trading_session_active(symbol: str) -> bool:
    now     = now_pt()
    weekday = now.weekday()
    h       = now.hour + now.minute / 60.0
    if weekday == 5: return False
    if weekday == 6 and h < 15.0: return False
    if symbol in ("MNQ=F", "MES=F"):
        return (6.5 <= h < 8.5) or (11.0 <= h < 13.0)
    if symbol == "MGC=F":
        return True   # 24h market
    return True

# ─── TP/SL Checker ────────────────────────────────────────────────────────────
def check_open_trades(symbol: str, df: pd.DataFrame):
    trades      = load_trades()
    open_trades = [t for t in trades if t["status"] == "open" and t["symbol"] == symbol]
    if not open_trades or df.empty:
        return

    # Always fetch max history so old open trades aren't stuck
    try:
        spacing_min = (df.index[-1] - df.index[-2]).total_seconds() / 60
        if   spacing_min <= 1:  iv = "1m"
        elif spacing_min <= 2:  iv = "2m"
        elif spacing_min <= 5:  iv = "5m"
        elif spacing_min <= 15: iv = "15m"
        elif spacing_min <= 30: iv = "30m"
        else:                   iv = "1h"
        ext = fetch_data(symbol, iv, _MAX_PERIOD.get(iv, "60d"))
        if not ext.empty and len(ext) > len(df):
            df = ext
    except Exception:
        pass

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
    if symbol not in _SESSION_GATE_EXEMPT and not trading_session_active(symbol):
        return False
    trades     = load_trades()
    sym_trades = [t for t in trades if t["symbol"] == symbol]
    if any(t["status"] == "open" for t in sym_trades):
        return False
    if not sym_trades:
        return True
    last = sym_trades[-1]
    try:
        last_time = datetime.fromisoformat(last["timestamp"])
        if last_time.tzinfo is None:
            last_time = last_time.replace(tzinfo=PT)
        if (now_pt() - last_time).total_seconds() < 1800:  # 30 min cooldown
            return False
    except Exception:
        pass
    return True

def record_signal(signal: dict, symbol: str):
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
def run_once():
    log.info("── tick ──")
    articles = fetch_news()
    for symbol in TICK_INFO:
        time.sleep(15)  # avoid yfinance rate limiting
        try:
            df = fetch_data(symbol, "5m", "60d")
            if df.empty:
                log.warning(f"{symbol}: no data")
                continue
            df = compute_indicators(df)

            # Higher timeframe confirmation — fetch 15m separately
            time.sleep(5)
            df_15m   = fetch_data(symbol, "15m", "60d")
            df_15m   = compute_indicators(df_15m) if not df_15m.empty else df_15m
            htf_bias = get_htf_bias(df_15m) if not df_15m.empty else 0

            ns  = get_news_sentiment(symbol, articles)
            sig = generate_signal(df, symbol, ns, htf_bias=htf_bias)

            log.info(f"{TICK_INFO[symbol]['name']}  {sig['direction']:7}  score={sig['score']:+.2f}  htf={'BUL' if htf_bias==1 else 'BEA' if htf_bias==-1 else 'NEU'}")

            # Always check open trades first
            check_open_trades(symbol, df)

            # Then record new signal if conditions met
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
