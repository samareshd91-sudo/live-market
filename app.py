
import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
from datetime import datetime, timezone

# ============================================================
# 🇮🇳 INDIA PRO SCALPING ENGINE — STEP 3
# Live technical dashboard + prominent signal alert layer
#
# IMPORTANT:
# - No fake live BUY/SELL signal is generated.
# - The alert system is wired to the final signal gate.
# - Current market source: yfinance (provider availability varies).
# - Options/OI and News remain hard-gated until validated adapters
#   are connected.
# ============================================================

st.set_page_config(
    page_title="India Pro Scalping Engine",
    page_icon="🚨",
    layout="wide",
    initial_sidebar_state="expanded",
)

# -----------------------------
# CSS
# -----------------------------
st.markdown("""
<style>
.block-container {
    padding-top: 0.9rem;
    padding-bottom: 2rem;
    max-width: 1400px;
}
.signal-alert {
    padding: 24px;
    border-radius: 18px;
    border: 2px solid currentColor;
    text-align: center;
    margin: 10px 0 22px 0;
}
.signal-title {
    font-size: 2.0rem;
    font-weight: 800;
    margin-bottom: 4px;
}
.signal-score {
    font-size: 1.35rem;
    font-weight: 700;
}
.alert-row {
    display: flex;
    justify-content: space-around;
    gap: 10px;
    flex-wrap: wrap;
    margin-top: 16px;
}
.alert-chip {
    padding: 9px 13px;
    border-radius: 10px;
    background: rgba(128,128,128,.12);
    min-width: 120px;
}
.wait-box {
    padding: 18px;
    border-radius: 14px;
    border: 1px solid rgba(128,128,128,.30);
    background: rgba(128,128,128,.06);
    text-align: center;
}
.warning-box {
    padding: 12px 15px;
    border-radius: 10px;
    border: 1px solid rgba(255,165,0,.35);
    background: rgba(255,165,0,.08);
}
.small {
    opacity: .72;
    font-size: .82rem;
}
@media (max-width: 700px) {
    .signal-title {font-size: 1.45rem;}
    .signal-score {font-size: 1.05rem;}
}
</style>
""", unsafe_allow_html=True)

# ============================================================
# CONFIG
# ============================================================

SYMBOLS = {
    "NIFTY 50": "^NSEI",
    "BANK NIFTY": "^NSEBANK",
}

TIMEFRAME_MAP = {
    "1M": "1m",
    "3M": "5m",   # provider-safe fallback
    "5M": "5m",
    "15M": "15m",
}

SIGNAL_THRESHOLD = 80

WEIGHTS = {
    "Market Structure": 15,
    "Trend & Momentum": 15,
    "Price + Volume": 10,
    "Options / OI": 15,
    "Key Levels": 10,
    "Risk / Liquidity": 10,
    "News & Macro": 25,
}

# ============================================================
# SESSION STATE
# ============================================================

if "last_alert_key" not in st.session_state:
    st.session_state.last_alert_key = None

if "alert_history" not in st.session_state:
    st.session_state.alert_history = []

# ============================================================
# DATA
# ============================================================

@st.cache_data(ttl=15, show_spinner=False)
def fetch_market(symbol: str, interval: str):
    try:
        df = yf.download(
            symbol,
            period="5d",
            interval=interval,
            auto_adjust=False,
            progress=False,
            threads=False,
        )

        if df is None or df.empty:
            return None, "No market data returned."

        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)

        df = df.rename(columns={
            "Open": "open",
            "High": "high",
            "Low": "low",
            "Close": "close",
            "Volume": "volume",
        })

        required = ["open", "high", "low", "close"]
        if any(c not in df.columns for c in required):
            return None, "OHLC columns unavailable."

        if "volume" not in df.columns:
            df["volume"] = np.nan

        df = df[["open", "high", "low", "close", "volume"]]
        df = df.dropna(subset=["open", "high", "low", "close"])
        df = df[~df.index.duplicated(keep="last")]

        return df, None

    except Exception as exc:
        return None, str(exc)

# ============================================================
# INDICATORS
# ============================================================

def ema(s, n):
    return s.ewm(span=n, adjust=False).mean()

def rsi(s, n=14):
    d = s.diff()
    gain = d.clip(lower=0)
    loss = -d.clip(upper=0)
    ag = gain.ewm(alpha=1/n, adjust=False).mean()
    al = loss.ewm(alpha=1/n, adjust=False).mean()
    rs = ag / al.replace(0, np.nan)
    return 100 - 100 / (1 + rs)

def atr(df, n=14):
    prev = df["close"].shift(1)
    tr = pd.concat([
        df["high"] - df["low"],
        (df["high"] - prev).abs(),
        (df["low"] - prev).abs(),
    ], axis=1).max(axis=1)
    return tr.ewm(alpha=1/n, adjust=False).mean()

def adx(df, n=14):
    high, low, close = df["high"], df["low"], df["close"]

    up = high.diff()
    down = -low.diff()

    plus_dm = pd.Series(
        np.where((up > down) & (up > 0), up, 0.0),
        index=df.index,
    )
    minus_dm = pd.Series(
        np.where((down > up) & (down > 0), down, 0.0),
        index=df.index,
    )

    prev = close.shift(1)
    tr = pd.concat([
        high - low,
        (high - prev).abs(),
        (low - prev).abs(),
    ], axis=1).max(axis=1)

    atr_rma = tr.ewm(alpha=1/n, adjust=False).mean()
    plus_di = 100 * plus_dm.ewm(alpha=1/n, adjust=False).mean() / atr_rma
    minus_di = 100 * minus_dm.ewm(alpha=1/n, adjust=False).mean() / atr_rma

    dx = 100 * (plus_di - minus_di).abs() / (
        plus_di + minus_di
    ).replace(0, np.nan)

    return (
        dx.ewm(alpha=1/n, adjust=False).mean(),
        plus_di,
        minus_di,
    )

def vwap(df):
    # Gracefully falls back when volume is unavailable/zero.
    vol = pd.to_numeric(df["volume"], errors="coerce").fillna(0)
    typical = (df["high"] + df["low"] + df["close"]) / 3
    day = pd.Series(df.index.date, index=df.index)

    pv = typical * vol
    cpv = pv.groupby(day).cumsum()
    cv = vol.groupby(day).cumsum()

    result = cpv / cv.replace(0, np.nan)

    # If provider does not supply useful volume, use typical price
    # as a clearly labelled fallback rather than returning None.
    if result.notna().sum() < 3:
        result = typical

    return result

def add_indicators(df):
    x = df.copy()

    x["ema20"] = ema(x["close"], 20)
    x["ema50"] = ema(x["close"], 50)
    x["ema200"] = ema(x["close"], 200)
    x["rsi14"] = rsi(x["close"])
    x["atr14"] = atr(x)
    x["adx14"], x["plus_di"], x["minus_di"] = adx(x)
    x["vwap"] = vwap(x)

    vol = pd.to_numeric(x["volume"], errors="coerce")
    x["vol_sma20"] = vol.rolling(20, min_periods=5).mean()
    x["rvol"] = vol / x["vol_sma20"].replace(0, np.nan)

    x["body"] = (x["close"] - x["open"]).abs()
    x["range"] = (x["high"] - x["low"]).replace(0, np.nan)
    x["body_pct"] = x["body"] / x["range"]

    return x

# ============================================================
# STRUCTURE / TECHNICAL SCORE
# ============================================================

def structure_state(df):
    if len(df) < 10:
        return "DATA PENDING", 0

    r = df.tail(8)

    hh = r["high"].iloc[-1] > r["high"].iloc[-5]
    hl = r["low"].iloc[-1] > r["low"].iloc[-5]
    lh = r["high"].iloc[-1] < r["high"].iloc[-5]
    ll = r["low"].iloc[-1] < r["low"].iloc[-5]

    if hh and hl:
        return "BULLISH", 1
    if lh and ll:
        return "BEARISH", -1
    return "TRANSITION", 0

def technical_components(df):
    row = df.iloc[-1]
    structure, direction = structure_state(df)

    structure_score = {
        "BULLISH": 12,
        "BEARISH": 12,
        "TRANSITION": 6,
        "DATA PENDING": 0,
    }.get(structure, 0)

    trend = 0
    if row["close"] > row["ema20"]:
        trend += 3
    if row["ema20"] > row["ema50"]:
        trend += 3
    if row["close"] > row["vwap"]:
        trend += 3

    if pd.notna(row["adx14"]):
        trend += 3 if row["adx14"] >= 25 else 2 if row["adx14"] >= 18 else 0

    if pd.notna(row["rsi14"]):
        trend += 3 if 55 <= row["rsi14"] <= 70 else 1 if 45 <= row["rsi14"] < 55 else 0

    trend = min(15, trend)

    pv = 0
    if pd.notna(row["rvol"]):
        pv += 4 if row["rvol"] >= 2 else 3 if row["rvol"] >= 1.2 else 1 if row["rvol"] >= .8 else 0
    if pd.notna(row["body_pct"]) and row["body_pct"] >= .60:
        pv += 3
    if row["close"] != row["open"]:
        pv += 1
    pv = min(10, pv)

    key = 5
    risk = 0
    if pd.notna(row["atr14"]) and row["atr14"] > 0:
        risk += 5
    if pd.notna(row["rvol"]):
        risk += 3
    risk = min(10, risk)

    return {
        "Market Structure": structure_score,
        "Trend & Momentum": trend,
        "Price + Volume": pv,
        "Options / OI": None,
        "Key Levels": key,
        "Risk / Liquidity": risk,
        "News & Macro": None,
        "structure": structure,
        "direction": direction,
    }

# ============================================================
# FINAL SIGNAL GATE
# ============================================================

def build_final_signal(components, row):
    """
    HARD GATE:
    A final signal cannot exist unless every critical module is available.
    Current Step 3 has Options/OI and News intentionally unavailable.
    """
    critical = [
        components["Market Structure"],
        components["Trend & Momentum"],
        components["Price + Volume"],
        components["Options / OI"],
        components["Key Levels"],
        components["Risk / Liquidity"],
        components["News & Macro"],
    ]

    if any(v is None for v in critical):
        return {
            "status": "WAIT",
            "reason": "DATA INCOMPLETE",
            "score": None,
            "direction": None,
        }

    score = sum(critical)

    if score < SIGNAL_THRESHOLD:
        return {
            "status": "WAIT",
            "reason": f"SCORE {score}/100 < {SIGNAL_THRESHOLD}",
            "score": score,
            "direction": None,
        }

    # Direction is only decided after all modules pass.
    if components["direction"] > 0:
        direction = "BUY CALL"
    elif components["direction"] < 0:
        direction = "BUY PUT"
    else:
        return {
            "status": "WAIT",
            "reason": "NO DIRECTIONAL CONFLUENCE",
            "score": score,
            "direction": None,
        }

    return {
        "status": "CONFIRMED",
        "reason": "ALL HARD GATES PASSED",
        "score": score,
        "direction": direction,
    }

# ============================================================
# ALERT SYSTEM
# ============================================================

def render_signal_alert(signal, symbol_name, row):
    """
    Visual alert + browser speech attempt.

    Browsers may block autoplay audio/speech until the user has
    interacted with the page. The visual alert remains reliable.
    """
    if signal["status"] != "CONFIRMED":
        st.markdown(
            f"""
            <div class="wait-box">
                <div style="font-size:1.35rem;font-weight:800;">⚪ WAIT</div>
                <div>{signal["reason"]}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
        return

    direction = signal["direction"]
    score = signal["score"]

    if "CALL" in direction:
        icon = "🟢"
    else:
        icon = "🟣"

    ts = datetime.now().strftime("%H:%M:%S")

    st.markdown(
        f"""
        <div class="signal-alert">
            <div class="signal-title">{icon} 🚨 SIGNAL CONFIRMED</div>
            <div style="font-size:1.55rem;font-weight:800;">{direction}</div>
            <div class="signal-score">{symbol_name} • SCORE {score}/100</div>

            <div class="alert-row">
                <div class="alert-chip"><b>LTP</b><br>{row["close"]:,.2f}</div>
                <div class="alert-chip"><b>RSI</b><br>{row["rsi14"]:.1f}</div>
                <div class="alert-chip"><b>ADX</b><br>{row["adx14"]:.1f}</div>
                <div class="alert-chip"><b>TIME</b><br>{ts}</div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # Browser notification / speech. This is best-effort because
    # mobile browsers can block unsolicited autoplay.
    speech_text = f"{direction} signal confirmed for {symbol_name}. Score {score}."
    st.components.v1.html(
        f"""
        <script>
        (function() {{
            const text = {speech_text!r};

            try {{
                if ("Notification" in window && Notification.permission === "granted") {{
                    new Notification("🚨 Trading Signal", {{body: text}});
                }}
            }} catch(e) {{}}

            try {{
                if ("speechSynthesis" in window) {{
                    window.speechSynthesis.cancel();
                    const u = new SpeechSynthesisUtterance(text);
                    u.rate = 0.95;
                    window.speechSynthesis.speak(u);
                }}
            }} catch(e) {{}}
        }})();
        </script>
        """,
        height=0,
    )

# ============================================================
# SIDEBAR
# ============================================================

with st.sidebar:
    st.header("⚙️ Controls")

    symbol_name = st.selectbox(
        "Market",
        list(SYMBOLS.keys()),
        index=0,
    )

    timeframe = st.selectbox(
        "Primary Timeframe",
        ["1M", "3M", "5M", "15M"],
        index=2,
    )

    auto_refresh = st.checkbox("Auto refresh", value=True)

    st.divider()
    st.caption("Signal threshold: 80/100")
    st.caption("Options/OI: HARD GATE")
    st.caption("News: HARD GATE")
    st.caption("Telegram: OFF")
    st.caption("Orders: OFF")

# ============================================================
# FETCH
# ============================================================

df, error = fetch_market(
    SYMBOLS[symbol_name],
    TIMEFRAME_MAP[timeframe],
)

if df is None:
    st.error(f"Market data unavailable: {error}")
    st.stop()

if len(df) < 50:
    st.warning(f"Only {len(df)} candles available.")

df = add_indicators(df)
row = df.iloc[-1]
components = technical_components(df)
final_signal = build_final_signal(components, row)

# ============================================================
# SIGNAL ALERT — FIRST THING USER SEES
# ============================================================

st.title("🇮🇳 India Pro Scalping Engine")
st.caption("STEP 3 • Signal Alert + Live Technical Verification")

render_signal_alert(final_signal, symbol_name, row)

# Alert event deduplication.
if final_signal["status"] == "CONFIRMED":
    candle_id = str(df.index[-1])
    alert_key = f"{symbol_name}|{TIMEFRAME_MAP[timeframe]}|{candle_id}|{final_signal['direction']}|{final_signal['score']}"

    if st.session_state.last_alert_key != alert_key:
        st.session_state.last_alert_key = alert_key
        st.session_state.alert_history.insert(
            0,
            {
                "Time": datetime.now().strftime("%H:%M:%S"),
                "Market": symbol_name,
                "Signal": final_signal["direction"],
                "Score": final_signal["score"],
            },
        )
        st.session_state.alert_history = st.session_state.alert_history[:20]

# ============================================================
# MARKET STRIP
# ============================================================

c1, c2, c3, c4, c5 = st.columns(5)

c1.metric("MARKET", symbol_name)
c2.metric("LTP", f"{row['close']:,.2f}")

prev = df["close"].iloc[-2]
change = row["close"] - prev
pct = change / prev * 100 if prev else 0

c3.metric("CHANGE", f"{change:+,.2f}", f"{pct:+.2f}%")
c4.metric("STRUCTURE", components["structure"])
c5.metric("BIAS", "BULLISH" if components["direction"] > 0 else "BEARISH" if components["direction"] < 0 else "NEUTRAL")

st.caption(f"Data: Yahoo Finance / yfinance • Last candle: {df.index[-1]}")

# ============================================================
# TECHNICAL
# ============================================================

st.subheader("⚡ Professional Technical Snapshot")

metric_data = [
    ("RSI 14", row["rsi14"]),
    ("ADX 14", row["adx14"]),
    ("ATR 14", row["atr14"]),
    ("EMA 20", row["ema20"]),
    ("EMA 50", row["ema50"]),
    ("EMA 200", row["ema200"]),
    ("VWAP", row["vwap"]),
    ("RVOL", row["rvol"]),
]

for start in range(0, len(metric_data), 4):
    cols = st.columns(4)
    for col, (label, value) in zip(cols, metric_data[start:start+4]):
        col.metric(
            label,
            "PENDING" if pd.isna(value) else f"{value:,.2f}",
        )

# ============================================================
# PRESSURE
# ============================================================

st.subheader("🐂🐻 Buyer vs Seller Pressure")

if pd.notna(row["body_pct"]):
    body_strength = float(np.clip(row["body_pct"], 0, 1))
    buyer = 50 + (40 * body_strength if row["close"] >= row["open"] else -40 * body_strength)
    buyer = float(np.clip(buyer, 0, 100))
    seller = 100 - buyer

    p1, p2 = st.columns(2)
    p1.metric("Estimated Buyer Pressure", f"{buyer:.0f}%")
    p2.metric("Estimated Seller Pressure", f"{seller:.0f}%")
    st.progress(buyer / 100)

st.caption("OHLCV-based estimate only. This is not true bid/ask order flow.")

# ============================================================
# STRUCTURE
# ============================================================

st.subheader("🏗️ Multi-Factor Market Structure")

a, b, c, d = st.columns(4)
a.metric("15M Bias", "PENDING")
b.metric("Current TF", components["structure"])
c.metric("BOS", "PENDING")
d.metric("CHOCH", "PENDING")

# ============================================================
# OPTIONS/OI
# ============================================================

st.subheader("🐋 Options / OI Intelligence")

for col, label in zip(
    st.columns(6),
    ["ATM Strike", "PCR", "Call OI", "Put OI", "Call ΔOI", "Put ΔOI"],
):
    col.metric(label, "DATA PENDING")

st.warning(
    "Options/OI is a hard gate. No option data = no final BUY/SELL signal."
)

# ============================================================
# NEWS
# ============================================================

st.subheader("📰 News Intelligence")

for col, label in zip(
    st.columns(4),
    ["News Bias", "Impact", "Freshness", "India Relevance"],
):
    col.metric(label, "DATA PENDING")

st.warning(
    "News is a hard gate. No validated news confirmation = no final BUY/SELL signal."
)

# ============================================================
# SCORE
# ============================================================

st.subheader("🎯 Master 100-Point Engine")

score_rows = []
for module, weight in WEIGHTS.items():
    score_rows.append({
        "Module": module,
        "Weight": weight,
        "Status": "PENDING" if components[module] is None else "AVAILABLE",
        "Score": "—" if components[module] is None else components[module],
    })

score_df = pd.DataFrame(score_rows)
st.dataframe(score_df, use_container_width=True, hide_index=True)

st.info(
    "Final score is intentionally locked until all seven modules are available. "
    "The 80/100 threshold will be evaluated only after all hard gates pass."
)

# ============================================================
# ALERT HISTORY
# ============================================================

st.subheader("🔔 Alert History")

if st.session_state.alert_history:
    st.dataframe(
        pd.DataFrame(st.session_state.alert_history),
        use_container_width=True,
        hide_index=True,
    )
else:
    st.caption("No confirmed signals yet.")

# ============================================================
# TRADE PLAN
# ============================================================

st.subheader("🎯 Trade Plan")

trade_plan = pd.DataFrame({
    "Field": [
        "Final Decision",
        "Entry",
        "Stop Loss",
        "Target 1",
        "Target 2",
        "Target 3",
        "Risk / Reward",
    ],
    "Value": ["PENDING"] * 7,
})
st.dataframe(trade_plan, use_container_width=True, hide_index=True)

# ============================================================
# FOOTER / AUTO REFRESH
# ============================================================

st.divider()
st.caption(
    f"India Pro Scalping Engine • Step 3 • "
    f"{datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%S UTC')}"
)

if auto_refresh:
    st.markdown(
        """
        <script>
        setTimeout(function () {
            window.location.reload();
        }, 15000);
        </script>
        """,
        unsafe_allow_html=True,
    )
