
import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
from datetime import datetime, timezone

# ============================================================
# 🇮🇳 INDIA PRO SCALPING ENGINE — STEP 2
# Live market-data foundation + technical engine
#
# Current data adapter:
#   Yahoo Finance via yfinance
#
# IMPORTANT:
# - This is a market-data/verification build, NOT an order executor.
# - Yahoo/yfinance data availability and latency can vary.
# - No fake BUY/SELL values are generated.
# - Options/OI + News Engine remain pending until their data adapters
#   are connected and validated.
# ============================================================

st.set_page_config(
    page_title="India Pro Scalping Engine",
    page_icon="🇮🇳",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
.block-container {padding-top:1rem;padding-bottom:2rem}
.card {
    border:1px solid rgba(128,128,128,.25);
    border-radius:12px;padding:14px;
    background:rgba(128,128,128,.05);
}
.big-signal {
    border:1px solid rgba(128,128,128,.3);
    border-radius:14px;padding:22px;text-align:center;
}
.small {opacity:.72;font-size:.82rem}
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
    "3M": "5m",   # Provider-safe fallback; UI labels it as entry context.
    "5M": "5m",
    "15M": "15m",
}

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
# DATA
# ============================================================

@st.cache_data(ttl=15, show_spinner=False)
def fetch_market(symbol: str, interval: str):
    """
    Fetch recent candles.
    Yahoo Finance intraday availability is provider-dependent.
    """
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
            return None, "No data returned."

        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)

        df = df.rename(columns={
            "Open": "open",
            "High": "high",
            "Low": "low",
            "Close": "close",
            "Volume": "volume",
        })

        needed = ["open", "high", "low", "close"]
        if any(c not in df.columns for c in needed):
            return None, "Required OHLC columns missing."

        if "volume" not in df.columns:
            df["volume"] = np.nan

        df = df[["open", "high", "low", "close", "volume"]].copy()
        df = df.dropna(subset=["open", "high", "low", "close"])

        # Remove duplicate timestamps.
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
    delta = s.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)

    avg_gain = gain.ewm(alpha=1/n, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1/n, adjust=False).mean()

    rs = avg_gain / avg_loss.replace(0, np.nan)
    return 100 - (100 / (1 + rs))

def atr(df, n=14):
    prev = df["close"].shift(1)
    tr = pd.concat([
        df["high"] - df["low"],
        (df["high"] - prev).abs(),
        (df["low"] - prev).abs(),
    ], axis=1).max(axis=1)

    return tr.ewm(alpha=1/n, adjust=False).mean()

def adx(df, n=14):
    high = df["high"]
    low = df["low"]
    close = df["close"]

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

    dx = 100 * (plus_di - minus_di).abs() / (plus_di + minus_di).replace(0, np.nan)
    adx_line = dx.ewm(alpha=1/n, adjust=False).mean()

    return adx_line, plus_di, minus_di

def vwap(df):
    # Intraday VWAP should reset each trading day.
    typical = (df["high"] + df["low"] + df["close"]) / 3
    day = pd.Series(df.index.date, index=df.index)
    pv = typical * df["volume"].fillna(0)

    cum_pv = pv.groupby(day).cumsum()
    cum_vol = df["volume"].fillna(0).groupby(day).cumsum()

    return cum_pv / cum_vol.replace(0, np.nan)

def add_indicators(df):
    x = df.copy()

    x["ema20"] = ema(x["close"], 20)
    x["ema50"] = ema(x["close"], 50)
    x["ema200"] = ema(x["close"], 200)
    x["rsi14"] = rsi(x["close"], 14)
    x["atr14"] = atr(x, 14)

    x["adx14"], x["plus_di"], x["minus_di"] = adx(x, 14)

    x["vwap"] = vwap(x)

    x["vol_sma20"] = x["volume"].rolling(20, min_periods=5).mean()
    x["rvol"] = x["volume"] / x["vol_sma20"].replace(0, np.nan)

    x["body"] = (x["close"] - x["open"]).abs()
    x["range"] = (x["high"] - x["low"]).replace(0, np.nan)
    x["body_pct"] = x["body"] / x["range"]

    return x

# ============================================================
# MARKET STRUCTURE
# ============================================================

def structure_state(df):
    if len(df) < 10:
        return "DATA PENDING", 0

    recent = df.tail(8)

    hh = recent["high"].iloc[-1] > recent["high"].iloc[-5]
    hl = recent["low"].iloc[-1] > recent["low"].iloc[-5]

    lh = recent["high"].iloc[-1] < recent["high"].iloc[-5]
    ll = recent["low"].iloc[-1] < recent["low"].iloc[-5]

    if hh and hl:
        return "BULLISH", 1
    if lh and ll:
        return "BEARISH", -1

    return "TRANSITION", 0

def market_bias(row):
    bullish = 0
    bearish = 0

    if row["close"] > row["ema20"]:
        bullish += 1
    else:
        bearish += 1

    if row["ema20"] > row["ema50"]:
        bullish += 1
    else:
        bearish += 1

    if row["close"] > row["vwap"]:
        bullish += 1
    else:
        bearish += 1

    if row["plus_di"] > row["minus_di"]:
        bullish += 1
    else:
        bearish += 1

    if bullish >= 3:
        return "BULLISH"
    if bearish >= 3:
        return "BEARISH"
    return "NEUTRAL"

# ============================================================
# SCORE
# ============================================================

def technical_score(df):
    """
    Only scores modules that this data source can genuinely support.
    Options and News are intentionally NOT invented.
    Therefore a final 100-point score is not emitted yet.
    """
    row = df.iloc[-1]

    structure, direction = structure_state(df)

    # Structure: directional quality estimate, max 15
    structure_score = 0
    if structure == "BULLISH":
        structure_score = 12
    elif structure == "BEARISH":
        structure_score = 12
    elif structure == "TRANSITION":
        structure_score = 6

    # Trend/momentum, max 15
    trend_score = 0
    if row["close"] > row["ema20"]:
        trend_score += 3
    if row["ema20"] > row["ema50"]:
        trend_score += 3
    if row["close"] > row["vwap"]:
        trend_score += 3

    if pd.notna(row["adx14"]):
        if row["adx14"] >= 25:
            trend_score += 3
        elif row["adx14"] >= 18:
            trend_score += 2

    if pd.notna(row["rsi14"]):
        if 55 <= row["rsi14"] <= 70:
            trend_score += 3
        elif 45 <= row["rsi14"] < 55:
            trend_score += 1

    trend_score = min(15, trend_score)

    # Price + volume, max 10
    pv_score = 0
    if pd.notna(row["rvol"]):
        if row["rvol"] >= 2:
            pv_score += 4
        elif row["rvol"] >= 1.2:
            pv_score += 3
        elif row["rvol"] >= 0.8:
            pv_score += 1

    if pd.notna(row["body_pct"]) and row["body_pct"] >= 0.60:
        pv_score += 3

    if row["close"] > row["open"]:
        pv_score += 1
    elif row["close"] < row["open"]:
        pv_score += 1

    pv_score = min(10, pv_score)

    # Key levels, max 10 — basic provider-neutral implementation
    key_score = 0
    if row["close"] > row["vwap"]:
        key_score += 5
    else:
        key_score += 5

    # Risk/liquidity: only partial evidence from OHLCV
    risk_score = 0
    if pd.notna(row["atr14"]) and row["atr14"] > 0:
        risk_score += 5
    if pd.notna(row["rvol"]):
        risk_score += 3
    risk_score = min(10, risk_score)

    return {
        "Market Structure": structure_score,
        "Trend & Momentum": trend_score,
        "Price + Volume": pv_score,
        "Options / OI": None,
        "Key Levels": key_score,
        "Risk / Liquidity": risk_score,
        "News & Macro": None,
        "structure": structure,
        "direction": direction,
    }

# ============================================================
# UI
# ============================================================

st.title("🇮🇳 India Pro Scalping Engine")
st.caption("STEP 2 • Live market-data + technical verification layer")

with st.sidebar:
    st.header("⚙️ Controls")

    symbol_name = st.selectbox(
        "Market",
        list(SYMBOLS.keys()),
    )

    timeframe = st.selectbox(
        "Primary Timeframe",
        ["1M", "3M", "5M", "15M"],
        index=2,
    )

    auto_refresh = st.checkbox("Auto refresh", value=True)

    if auto_refresh:
        st.caption("Dashboard cache: 15 seconds")

    st.divider()
    st.caption("Backtesting: OFF")
    st.caption("Paper Trading: OFF")
    st.caption("Telegram: OFF")

ticker = SYMBOLS[symbol_name]
interval = TIMEFRAME_MAP[timeframe]

df, error = fetch_market(ticker, interval)

if df is None:
    st.error(f"Market data unavailable: {error}")
    st.stop()

if len(df) < 50:
    st.warning(f"Only {len(df)} candles available. Technical confidence is limited.")

df = add_indicators(df)
row = df.iloc[-1]
scores = technical_score(df)

# ============================================================
# TOP STRIP
# ============================================================

c1, c2, c3, c4, c5 = st.columns(5)

c1.metric("MARKET", symbol_name)
c2.metric("LTP", f"{row['close']:,.2f}")

prev = df["close"].iloc[-2]
change = row["close"] - prev
change_pct = (change / prev) * 100 if prev else np.nan

c3.metric(
    "CHANGE",
    f"{change:+,.2f}",
    f"{change_pct:+.2f}%",
)

c4.metric("STRUCTURE", scores["structure"])
c5.metric("BIAS", market_bias(row))

st.caption(
    f"Data source: Yahoo Finance / yfinance • Last candle: {df.index[-1]}"
)

st.divider()

# ============================================================
# TECHNICAL SNAPSHOT
# ============================================================

st.subheader("⚡ Professional Technical Snapshot")

cols = st.columns(8)

metrics = [
    ("RSI 14", row["rsi14"], "0.0"),
    ("ADX 14", row["adx14"], "0.0"),
    ("ATR 14", row["atr14"], "0.00"),
    ("EMA 20", row["ema20"], "0.00"),
    ("EMA 50", row["ema50"], "0.00"),
    ("EMA 200", row["ema200"], "0.00"),
    ("VWAP", row["vwap"], "0.00"),
    ("RVOL", row["rvol"], "0.00"),
]

for col, (label, value, fmt) in zip(cols, metrics):
    if pd.isna(value):
        col.metric(label, "PENDING")
    else:
        col.metric(label, format(value, fmt))

# ============================================================
# BUYER / SELLER PRESSURE ESTIMATE
# ============================================================

st.subheader("🐂🐻 Buyer vs Seller Pressure")

if pd.notna(row["volume"]) and pd.notna(row["rvol"]):
    # Candle-based estimate only; explicitly labelled as estimate.
    body_direction = 1 if row["close"] >= row["open"] else -1
    body_strength = float(np.clip(row["body_pct"], 0, 1)) if pd.notna(row["body_pct"]) else 0

    buy_est = 50 + (body_direction * body_strength * 40)
    buy_est = float(np.clip(buy_est, 0, 100))
    sell_est = 100 - buy_est

    b1, b2 = st.columns(2)
    b1.metric("Estimated Buyer Pressure", f"{buy_est:.0f}%")
    b2.metric("Estimated Seller Pressure", f"{sell_est:.0f}%")

    st.progress(buy_est / 100)

    st.caption(
        "⚠️ Estimated from OHLCV candle pressure. This is NOT actual bid/ask order flow. "
        "True buyer/seller strength will be enabled when a suitable order-flow source is connected."
    )
else:
    st.info("Buyer/Seller pressure: DATA PENDING")

# ============================================================
# MARKET STRUCTURE
# ============================================================

st.subheader("🏗️ Multi-Factor Market Structure")

s1, s2, s3, s4 = st.columns(4)
s1.metric("15M Bias", "PENDING")
s2.metric("Current TF", scores["structure"])
s3.metric("BOS", "PENDING")
s4.metric("CHOCH", "PENDING")

# ============================================================
# KEY LEVELS
# ============================================================

st.subheader("📍 Key Levels")

day_high = float(df["high"].max())
day_low = float(df["low"].min())

levels = pd.DataFrame({
    "Level": [
        "Current Price",
        "Session/Loaded High",
        "Session/Loaded Low",
        "VWAP",
        "EMA 20",
        "EMA 50",
        "EMA 200",
    ],
    "Value": [
        row["close"],
        day_high,
        day_low,
        row["vwap"],
        row["ema20"],
        row["ema50"],
        row["ema200"],
    ],
})

levels["Distance"] = levels["Value"] - row["close"]
st.dataframe(
    levels.style.format({"Value": "{:,.2f}", "Distance": "{:+,.2f}"}),
    use_container_width=True,
    hide_index=True,
)

# ============================================================
# OPTIONS/OI — WAITING FOR VALIDATED ADAPTER
# ============================================================

st.subheader("🐋 Options / OI Intelligence")

oi_cols = st.columns(6)
for col, label in zip(
    oi_cols,
    ["ATM Strike", "PCR", "Call OI", "Put OI", "Call ΔOI", "Put ΔOI"],
):
    col.metric(label, "DATA PENDING")

st.warning(
    "Option-chain/OI is intentionally not fabricated. "
    "The next data adapter will populate ATM, PCR, OI, ΔOI, Call Wall and Put Wall."
)

# ============================================================
# NEWS
# ============================================================

st.subheader("📰 News Intelligence")

n1, n2, n3, n4 = st.columns(4)
n1.metric("News Bias", "DATA PENDING")
n2.metric("Impact", "DATA PENDING")
n3.metric("Freshness", "DATA PENDING")
n4.metric("India Relevance", "DATA PENDING")

st.info(
    "News BUY / NEWS SELL will remain disabled until a validated news feed "
    "and confirmation engine are connected."
)

# ============================================================
# 100 POINT ENGINE
# ============================================================

st.subheader("🎯 Master 100-Point Engine")

score_table = pd.DataFrame({
    "Module": list(WEIGHTS.keys()),
    "Weight": list(WEIGHTS.values()),
    "Status": [
        "AVAILABLE",
        "AVAILABLE",
        "AVAILABLE",
        "PENDING",
        "AVAILABLE",
        "PARTIAL",
        "PENDING",
    ],
    "Score": [
        scores["Market Structure"],
        scores["Trend & Momentum"],
        scores["Price + Volume"],
        "—",
        scores["Key Levels"],
        scores["Risk / Liquidity"],
        "—",
    ],
})

st.dataframe(score_table, use_container_width=True, hide_index=True)

st.warning(
    "FINAL 100/100 score is intentionally BLOCKED until Options/OI and News data "
    "are connected. A partial score must never be presented as a final trading signal."
)

# ============================================================
# SIGNALS
# ============================================================

sig1, sig2 = st.columns(2)

with sig1:
    st.subheader("📊 Market Signal")
    st.markdown(
        '<div class="big-signal">'
        '<h2>⚪ WAIT</h2>'
        '<p>Final confirmation pending Options/OI + News.</p>'
        '</div>',
        unsafe_allow_html=True,
    )

with sig2:
    st.subheader("📰 News Signal")
    st.markdown(
        '<div class="big-signal">'
        '<h2>⚪ NEWS WAIT</h2>'
        '<p>No validated news engine connected.</p>'
        '</div>',
        unsafe_allow_html=True,
    )

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
        "Position Size",
    ],
    "Value": ["PENDING"] * 8,
})
st.dataframe(trade_plan, use_container_width=True, hide_index=True)

# ============================================================
# WARNINGS
# ============================================================

st.subheader("⚠️ Risk / Data Warnings")

warnings = []

if error:
    warnings.append(error)

if len(df) < 200:
    warnings.append("Less than 200 candles: EMA200 is not fully warmed up.")

warnings.append("Option-chain/OI adapter is not connected.")
warnings.append("News adapter is not connected.")
warnings.append("This build does not place orders.")

for w in warnings:
    st.write(f"⚠️ {w}")

# ============================================================
# FOOTER
# ============================================================

st.divider()
st.caption(
    f"India Pro Scalping Engine • Step 2 • "
    f"{datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%S UTC')}"
)

if auto_refresh:
    st.markdown(
        """
        <script>
        setTimeout(function(){ window.location.reload(); }, 15000);
        </script>
        """,
        unsafe_allow_html=True,
    )
