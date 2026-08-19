
import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
from datetime import datetime, timezone

# ============================================================
# 🇮🇳 INDIA PRO SCALPING ENGINE — STEP 4
# Technical + FREE NSE Option Chain / OI Intelligence
#
# No paid broker API.
# Option data uses nselib's public NSE-data interface.
# If NSE blocks/rate-limits the request, the app safely returns
# DATA PENDING and does NOT fabricate an options score.
# ============================================================

st.set_page_config(page_title="India Pro Scalping Engine", page_icon="🚨", layout="wide")

st.markdown("""
<style>
.block-container{padding-top:.8rem;max-width:1400px}
.signal-alert{padding:22px;border-radius:18px;border:2px solid currentColor;text-align:center;margin-bottom:18px}
.signal-title{font-size:1.8rem;font-weight:800}
.wait-box{padding:18px;border-radius:14px;border:1px solid rgba(128,128,128,.3);text-align:center}
.small{opacity:.7;font-size:.82rem}
.warning-box{padding:12px;border-radius:10px;background:rgba(255,165,0,.08)}
@media(max-width:700px){.signal-title{font-size:1.4rem}}
</style>
""", unsafe_allow_html=True)

SYMBOLS = {"NIFTY 50":"^NSEI", "BANK NIFTY":"^NSEBANK"}
NSE_SYMBOLS = {"NIFTY 50":"NIFTY", "BANK NIFTY":"BANKNIFTY"}
INTERVALS = {"1M":"1m","3M":"5m","5M":"5m","15M":"15m"}

WEIGHTS = {
    "Market Structure":15, "Trend & Momentum":15, "Price + Volume":10,
    "Options / OI":15, "Key Levels":10, "Risk / Liquidity":10, "News & Macro":25
}

if "last_alert_key" not in st.session_state: st.session_state.last_alert_key=None
if "alert_history" not in st.session_state: st.session_state.alert_history=[]

@st.cache_data(ttl=15, show_spinner=False)
def market_data(symbol, interval):
    try:
        df=yf.download(symbol,period="5d",interval=interval,auto_adjust=False,progress=False,threads=False)
        if df is None or df.empty: return None,"No market data."
        if isinstance(df.columns,pd.MultiIndex): df.columns=df.columns.get_level_values(0)
        df=df.rename(columns={"Open":"open","High":"high","Low":"low","Close":"close","Volume":"volume"})
        if not all(c in df for c in ["open","high","low","close"]): return None,"OHLC unavailable."
        if "volume" not in df: df["volume"]=np.nan
        df=df[["open","high","low","close","volume"]].dropna(subset=["open","high","low","close"])
        return df[~df.index.duplicated(keep="last")],None
    except Exception as e:
        return None,str(e)

def ema(s,n): return s.ewm(span=n,adjust=False).mean()

def rsi(s,n=14):
    d=s.diff(); g=d.clip(lower=0); l=-d.clip(upper=0)
    ag=g.ewm(alpha=1/n,adjust=False).mean(); al=l.ewm(alpha=1/n,adjust=False).mean()
    rs=ag/al.replace(0,np.nan)
    return 100-100/(1+rs)

def atr(df,n=14):
    p=df.close.shift(1)
    tr=pd.concat([df.high-df.low,(df.high-p).abs(),(df.low-p).abs()],axis=1).max(axis=1)
    return tr.ewm(alpha=1/n,adjust=False).mean()

def adx(df,n=14):
    up=df.high.diff(); dn=-df.low.diff()
    plus=pd.Series(np.where((up>dn)&(up>0),up,0.),index=df.index)
    minus=pd.Series(np.where((dn>up)&(dn>0),dn,0.),index=df.index)
    p=df.close.shift(1)
    tr=pd.concat([df.high-df.low,(df.high-p).abs(),(df.low-p).abs()],axis=1).max(axis=1)
    ar=tr.ewm(alpha=1/n,adjust=False).mean()
    pdi=100*plus.ewm(alpha=1/n,adjust=False).mean()/ar
    mdi=100*minus.ewm(alpha=1/n,adjust=False).mean()/ar
    dx=100*(pdi-mdi).abs()/(pdi+mdi).replace(0,np.nan)
    return dx.ewm(alpha=1/n,adjust=False).mean(),pdi,mdi

def vwap(df):
    vol=pd.to_numeric(df.volume,errors="coerce").fillna(0)
    tp=(df.high+df.low+df.close)/3
    day=pd.Series(df.index.date,index=df.index)
    out=(tp*vol).groupby(day).cumsum()/vol.groupby(day).cumsum().replace(0,np.nan)
    return out if out.notna().sum()>=3 else tp

def add_ind(df):
    x=df.copy()
    x["ema20"]=ema(x.close,20); x["ema50"]=ema(x.close,50); x["ema200"]=ema(x.close,200)
    x["rsi14"]=rsi(x.close); x["atr14"]=atr(x); x["adx14"],x["plus_di"],x["minus_di"]=adx(x); x["vwap"]=vwap(x)
    x["vol_sma20"]=x.volume.rolling(20,min_periods=5).mean()
    x["rvol"]=x.volume/x.vol_sma20.replace(0,np.nan)
    x["body_pct"]=(x.close-x.open).abs()/(x.high-x.low).replace(0,np.nan)
    return x

def technical(df):
    r=df.iloc[-1]
    last=df.tail(8)
    bull=last.high.iloc[-1]>last.high.iloc[-5] and last.low.iloc[-1]>last.low.iloc[-5]
    bear=last.high.iloc[-1]<last.high.iloc[-5] and last.low.iloc[-1]<last.low.iloc[-5]
    structure="BULLISH" if bull else "BEARISH" if bear else "TRANSITION"
    direction=1 if bull else -1 if bear else 0
    ms={"BULLISH":12,"BEARISH":12,"TRANSITION":6}[structure]
    trend=0
    trend += 3 if r.close>r.ema20 else 0
    trend += 3 if r.ema20>r.ema50 else 0
    trend += 3 if r.close>r.vwap else 0
    trend += 3 if r.adx14>=25 else 2 if r.adx14>=18 else 0
    trend += 3 if 55<=r.rsi14<=70 else 1 if 45<=r.rsi14<55 else 0
    pv=0
    if pd.notna(r.rvol): pv += 4 if r.rvol>=2 else 3 if r.rvol>=1.2 else 1 if r.rvol>=.8 else 0
    if pd.notna(r.body_pct) and r.body_pct>=.6: pv+=3
    pv+=1
    risk=5 if pd.notna(r.atr14) and r.atr14>0 else 0
    risk+=3 if pd.notna(r.rvol) else 0
    return {
        "Market Structure":min(ms,15),"Trend & Momentum":min(trend,15),
        "Price + Volume":min(pv,10),"Options / OI":None,
        "Key Levels":5,"Risk / Liquidity":min(risk,10),
        "News & Macro":None,"structure":structure,"direction":direction
    }

# ---------------- NSE OPTION CHAIN ----------------

@st.cache_data(ttl=20, show_spinner=False)
def fetch_nse_chain(symbol):
    try:
        from nselib import derivatives
        raw=derivatives.nse_live_option_chain(symbol=symbol, oi_mode="compact")
        if raw is None: return None,"NSE returned no option chain."
        if isinstance(raw,tuple): raw=raw[0]
        if not isinstance(raw,pd.DataFrame): raw=pd.DataFrame(raw)
        if raw.empty: return None,"Empty option chain."
        return raw,""
    except Exception as e:
        return None,str(e)

def num(df, col):
    if col not in df: return pd.Series(np.nan,index=df.index)
    return pd.to_numeric(df[col],errors="coerce").fillna(0)

def option_analysis(chain, spot):
    x=chain.copy()
    # nselib compact names
    aliases={
        "strike":["Strike Price","strikePrice","strike_price"],
        "coi":["CALLS_OI","CE_OI","call_oi"],
        "poi":["PUTS_OI","PE_OI","put_oi"],
        "cd":["CALLS_Chng in OI","CE_Chng_OI","call_change_oi"],
        "pd":["PUTS_Chng in OI","PE_Chng_OI","put_change_oi"],
        "cv":["CALLS_Volume","CE_Volume","call_volume"],
        "pv":["PUTS_Volume","PE_Volume","put_volume"],
        "cltp":["CALLS_LTP","CE_LTP","call_ltp"],
        "pltp":["PUTS_LTP","PE_LTP","put_ltp"],
    }
    def pick(names):
        for n in names:
            if n in x.columns:return n
        return None
    cols={k:pick(v) for k,v in aliases.items()}
    if cols["strike"] is None or cols["coi"] is None or cols["poi"] is None:
        return None,"Required OI columns missing."

    y=pd.DataFrame(index=x.index)
    for k,c in cols.items(): y[k]=num(x,c) if c else 0
    y["strike"]=pd.to_numeric(x[cols["strike"]],errors="coerce")
    y=y.dropna(subset=["strike"]).sort_values("strike")
    if y.empty:return None,"No strikes."

    atm=float(y.iloc[(y.strike-spot).abs().argmin()].strike)
    total_call=y.coi.sum(); total_put=y.poi.sum()
    pcr=float(total_put/total_call) if total_call else np.nan
    call_wall=float(y.loc[y.coi.idxmax(),"strike"]) if y.coi.sum()>0 else np.nan
    put_wall=float(y.loc[y.poi.idxmax(),"strike"]) if y.poi.sum()>0 else np.nan

    # Max Pain: sum option intrinsic loss at each strike.
    strikes=y.strike.to_numpy()
    pain=[]
    for k in strikes:
        call_loss=((k-y.strike).clip(lower=0)*y.coi).sum()
        put_loss=((y.strike-k).clip(lower=0)*y.poi).sum()
        pain.append(call_loss+put_loss)
    max_pain=float(strikes[int(np.argmin(pain))]) if len(pain) else np.nan

    # Option directional score, deliberately conservative.
    score=0
    if pcr>=1.20: score+=4
    elif pcr>=1.05: score+=3
    elif pcr<=0.80: score-=4
    elif pcr<=0.95: score-=2

    if spot>call_wall: score+=2
    if spot<put_wall: score-=2

    # ΔOI: rising put OI / falling call OI is mildly bullish; inverse bearish.
    cd=y.cd.sum(); pd_=y.pd.sum()
    if pd_>max(cd,0)*1.15: score+=3
    elif cd>max(pd_,0)*1.15: score-=3

    score=int(np.clip(score,-15,15))
    return {
        "chain":y,"atm":atm,"pcr":pcr,"call_oi":total_call,"put_oi":total_put,
        "call_doi":cd,"put_doi":pd_,"call_wall":call_wall,
        "put_wall":put_wall,"max_pain":max_pain,"score":score
    },""

# ---------------- SIDEBAR ----------------

with st.sidebar:
    st.header("⚙️ Controls")
    market=st.selectbox("Market",list(SYMBOLS.keys()))
    tf=st.selectbox("Primary Timeframe",["1M","3M","5M","15M"],index=2)
    auto=st.checkbox("Auto refresh",True)
    st.divider()
    st.caption("Signal threshold: 80/100")
    st.caption("Options/OI: LIVE FREE NSE LAYER")
    st.caption("News: HARD GATE")
    st.caption("Telegram: OFF")
    st.caption("Orders: OFF")

df,err=market_data(SYMBOLS[market],INTERVALS[tf])
if df is None: st.error(err); st.stop()
df=add_ind(df)
row=df.iloc[-1]
comp=technical(df)

chain,chain_err=fetch_nse_chain(NSE_SYMBOLS[market])
opt=None
if chain is not None:
    opt,opt_err=option_analysis(chain,float(row.close))
else:
    opt_err=chain_err

if opt:
    comp["Options / OI"]=float(np.clip(7.5+opt["score"]/2,0,15))

# Final hard gate remains locked because News is still missing.
critical=[comp[k] for k in WEIGHTS]
ready=all(v is not None for v in critical)
final_score=int(sum(critical)) if ready else None
confirmed=ready and final_score>=80 and comp["direction"]!=0

if confirmed:
    direction="BUY CALL" if comp["direction"]>0 else "BUY PUT"
    st.markdown(f"""
    <div class="signal-alert">
    <div class="signal-title">🚨 SIGNAL CONFIRMED</div>
    <div style="font-size:1.5rem;font-weight:800">{direction}</div>
    <div>{market} • {final_score}/100 • {datetime.now().strftime("%H:%M:%S")}</div>
    </div>""",unsafe_allow_html=True)
else:
    reason="NEWS DATA PENDING" if comp["News & Macro"] is None else f"SCORE {final_score}/100"
    st.markdown(f'<div class="wait-box"><b>⚪ WAIT</b><br>{reason}</div>',unsafe_allow_html=True)

st.title("🇮🇳 India Pro Scalping Engine")
st.caption("STEP 4 • Technical + Free NSE Option Chain / OI")

c=st.columns(5)
c[0].metric("MARKET",market)
c[1].metric("LTP",f"{row.close:,.2f}")
prev=df.close.iloc[-2]; ch=row.close-prev
c[2].metric("CHANGE",f"{ch:+,.2f}",f"{ch/prev*100:+.2f}%")
c[3].metric("STRUCTURE",comp["structure"])
c[4].metric("BIAS","BULLISH" if comp["direction"]>0 else "BEARISH" if comp["direction"]<0 else "NEUTRAL")

st.subheader("⚡ Professional Technical Snapshot")
for i in range(0,8,4):
    cc=st.columns(4)
    vals=[("RSI 14",row.rsi14),("ADX 14",row.adx14),("ATR 14",row.atr14),("EMA 20",row.ema20),
          ("EMA 50",row.ema50),("EMA 200",row.ema200),("VWAP",row.vwap),("RVOL",row.rvol)][i:i+4]
    for col,(name,val) in zip(cc,vals): col.metric(name,"PENDING" if pd.isna(val) else f"{val:,.2f}")

st.subheader("🐂🐻 Buyer vs Seller Pressure")
bp=50+40*float(np.clip(row.body_pct,0,1))*(1 if row.close>=row.open else -1)
bp=float(np.clip(bp,0,100)); sp=100-bp
a,b=st.columns(2); a.metric("Estimated Buyer Pressure",f"{bp:.0f}%"); b.metric("Estimated Seller Pressure",f"{sp:.0f}%")
st.progress(bp/100)
st.caption("OHLCV estimate only; not true bid/ask order flow.")

st.subheader("🐋 Options / OI Intelligence")
if opt:
    cols=st.columns(6)
    for col,label,val in zip(cols,["ATM Strike","PCR","Call OI","Put OI","Call ΔOI","Put ΔOI"],
                            [opt["atm"],opt["pcr"],opt["call_oi"],opt["put_oi"],opt["call_doi"],opt["put_doi"]]):
        col.metric(label,f"{val:,.2f}" if isinstance(val,float) else f"{val:,}")
    a,b,c=st.columns(3)
    a.metric("📞 Call Wall",f"{opt['call_wall']:,.0f}")
    b.metric("📍 Max Pain",f"{opt['max_pain']:,.0f}")
    c.metric("🛡️ Put Wall",f"{opt['put_wall']:,.0f}")
    st.metric("Options Score Contribution",f"{comp['Options / OI']:.1f}/15")
    show=opt["chain"].copy()
    near=show.iloc[(show.strike-float(row.close)).abs().argsort()[:15]].sort_values("strike")
    st.dataframe(near,use_container_width=True,hide_index=True)
else:
    st.warning(f"Option chain unavailable: {opt_err}")
    st.caption("No options score is fabricated. The Options/OI module remains PENDING.")

st.subheader("📰 News Intelligence")
for col,label in zip(st.columns(4),["News Bias","Impact","Freshness","India Relevance"]):
    col.metric(label,"DATA PENDING")
st.warning("News is still a hard gate. No validated news confirmation = no final BUY/SELL.")

st.subheader("🎯 Master 100-Point Engine")
rows=[]
for k,w in WEIGHTS.items():
    rows.append({"Module":k,"Weight":w,"Status":"AVAILABLE" if comp[k] is not None else "PENDING",
                 "Score":"—" if comp[k] is None else round(comp[k],1)})
st.dataframe(pd.DataFrame(rows),use_container_width=True,hide_index=True)

if not ready:
    st.info("Final 100-point score is locked until every module, including News & Macro, is validated.")

st.subheader("🎯 Trade Plan")
st.dataframe(pd.DataFrame({"Field":["Final Decision","Entry","Stop Loss","Target 1","Target 2","Target 3","Risk / Reward"],
                           "Value":["PENDING"]*7}),use_container_width=True,hide_index=True)

st.divider()
st.caption(f"India Pro Scalping Engine • Step 4 • {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%S UTC')}")

if auto:
    st.markdown('<script>setTimeout(function(){window.location.reload();},20000);</script>',unsafe_allow_html=True)
