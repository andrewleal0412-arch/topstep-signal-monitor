# PaperTrail — Claude Project Memory

## What This Project Is
A trading signal monitor for TopStep futures trading. Monitors **MGC=F (Micro Gold)** only.
- **Signal bot** runs 24/7, scores market conditions, records trades to Supabase
- **Dashboard** shows trade log, live signal, settings
- **TradingView indicator** mirrors the signal engine visually on charts

---

## Deployment
| Service | Platform | Trigger |
|---------|----------|---------|
| `worker.py` | Render (24/7 background) | Push to GitHub main |
| `app.py` | Streamlit Community Cloud | Push to GitHub main |

- **GitHub:** https://github.com/andrewleal0412-arch/topstep-signal-monitor
- **App URL:** https://topstep-signal-monitor-ruhonqhnvvtwbodnjycc8n.streamlit.app/
- **Supabase:** https://lzsgiqrwkukpyfokiebm.supabase.co — tables: `trades`, `config`
- **To deploy:** `git add . && git commit -m "message" && git push`

---

## File Map
| File | Purpose |
|------|---------|
| `worker.py` | 24/7 signal bot — runs on Render |
| `app.py` | Streamlit dashboard UI |
| `PaperTrail_Indicator.pine` | TradingView Pine Script v5 indicator |
| `requirements.txt` | Python deps |
| `.streamlit/secrets.toml` | Local secrets (never commit) |
| `cleanup_trades.py` | One-time DB cleanup script |

---

## Signal Scoring Engine (Max Score: 15.0)

| Component | Points | Notes |
|-----------|--------|-------|
| EMA stack (9/21/50) | ±1.5 | Full bull/bear stack (reduced from ±2.0 to fire earlier) |
| EMA crossover | ±1.0 | Fresh 9/21 cross |
| RSI | ±1.0 | <35 bull, >65 bear |
| MACD | ±1.0 | Line vs signal + histogram |
| VWAP | ±0.5 | Price above/below |
| Bollinger Bands | ±0.5 | Outside bands |
| Momentum (ROC) | ±1.0 | 5-bar rate of change — leading indicator |
| 15m HTF bias | ±1.0–1.5 | Confirming or contrarian |
| 1h HTF bias | ±1.5–2.0 | Confirming or contrarian |
| Candlestick patterns | ±0.5–1.0 | Engulfing, hammer, star, pin bar |
| Support/Resistance | ±1.0 | At level = +1, blocked = -0.75 |
| FVG (Fair Value Gap) | ±1.5 | Price inside FVG zone |

**Strength = abs(score) / 15.0 × 100%**

---

## Key Constants (worker.py)
```python
ACTIVE_SYMBOLS = {"MGC=F"}
_MIN_SCORE = {"MGC=F": 3.0}
_MAX_ENTRY_STALENESS_TICKS = {"MGC=F": 15}  # 15 × $0.10 = 1.5 pts
CHECK_INTERVAL_SEC = 120  # 2 min
NTFY_TOPIC = "topstepnotis"
```

---

## Architecture Decisions

### Data
- **yfinance** for market data — uses `df.iloc[-2]` (last CLOSED candle) to avoid stale/incomplete data
- `check_open_trades()` uses 1m candles — keeps ALL candles (already closed), drops last only for 5m+

### Trade Lifecycle
1. `run_once()` checks session gate first → if paused, return
2. If open trade exists → ONLY check TP/SL, skip signal generation
3. New signal → `should_record()` cooldown check → `_verify_and_correct_signal()` → save to Supabase

### Cooldowns (`should_record()`)
- Hard cooldown: 10 min
- Direction flip: 20 min
- Same direction: 30 min
- Stale entry: 1hr or 15 ticks
- **Consecutive loss breaker:** 2 same-direction losses → block that direction 1 hour

### Session Gate (`trading_session_active()`)
- Weekends: PAUSED
- CME maintenance: 2:00–3:00 PM PT daily: PAUSED
- Buffer after maintenance: 3:00–3:05 PM PT: PAUSED

### Entry Verification (`_verify_and_correct_signal()`)
- Fetches live 1m price
- Rewrites entry price to live price
- Recalculates TP/SL with ATR
- Sanity checks SL/TP sides and R:R ≥ 0.8

---

## Supabase Schema

### `trades` table
```
id (uuid), symbol, direction, entry, sl, tp, outcome,
created_at, closed_at, duration_min, pnl_ticks, score,
strength, reasons (jsonb), score_detail (jsonb)
```

### `config` table
```
id (text, "default"), data (jsonb)
```

---

## TradingView Pine Script (`PaperTrail_Indicator.pine`)
- Pine Script **v5** — line 4 must be exactly: `//@version=5`
- Mirrors the signal engine: EMAs, RSI, MACD, VWAP, BB, HTF bias (15m/1h), candlestick patterns, S/R, FVG
- BUY/SELL arrows on fresh signals only
- Entry/TP1/TP2/SL lines drawn on chart
- Score panel table (top right by default)
- Alert conditions for BUY and SELL
- **Known issue:** multi-line function calls inside `if` blocks cause syntax errors — keep on single lines
- To load: copy file contents → TradingView Pine Editor → Cmd+A → Delete → Cmd+V → Add to chart

---

## Common Tasks

### Push a fix to both services
```bash
cd ~/Desktop/TOPSTEP
git add -A
git commit -m "your message"
git push
```

### Copy Pine Script to clipboard
```bash
cat ~/Desktop/TOPSTEP/PaperTrail_Indicator.pine | pbcopy
```

### Run cleanup script (dry run)
```bash
SUPABASE_URL="https://lzsgiqrwkukpyfokiebm.supabase.co" \
SUPABASE_KEY="<key from .streamlit/secrets.toml>" \
python ~/Desktop/TOPSTEP/cleanup_trades.py
```

### Check Render logs
Go to https://dashboard.render.com → topstep-signal-monitor → Logs

---

## Session Tips
- Use `/clear` to reset context after finishing a task
- Each session: focus on ONE thing
- The bot only trades MGC=F — ignore MNQ/MES/ES references (legacy)
- App password: topstep2024
