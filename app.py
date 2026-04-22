"""
NEPSE Trading Intelligence Dashboard
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
"""

import os
import sys
import logging
from datetime import datetime

# ── Path fix — must come before ANY local imports ──────────────────────────────
_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)
# ──────────────────────────────────────────────────────────────────────────────

import numpy as np
import pandas as pd
import streamlit as st

from data_engine import get_market_data, market_summary, enrich_dataframe
from charts import (
    volume_bar_chart, pct_change_distribution, smart_money_heatmap,
    price_volume_impact_chart, market_breadth_gauge,
    top_movers_chart, smart_money_bar,
)
from utils import fmt_number, setup_logging

setup_logging(logging.WARNING)

# ── Page config ────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="NEPSE Intelligence",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── CSS ────────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@300;400;600;700&family=Syne:wght@400;600;700;800&display=swap');
:root {
  --bg:#0D1117; --card:#161B22; --border:#21262D;
  --accent:#00D4FF; --green:#00E676; --red:#FF5252;
  --amber:#FFB300; --purple:#B388FF; --text:#E6EDF3; --dim:#8B949E;
}
html,body,[class*="css"]{background:var(--bg)!important;color:var(--text);font-family:'JetBrains Mono',monospace}
.nepse-header{background:linear-gradient(135deg,#0D1117,#161B22 50%,#0D1117);
  border-bottom:1px solid var(--accent);padding:1.5rem 2rem;margin-bottom:1.5rem}
.nepse-title{font-family:'Syne',sans-serif;font-size:2.2rem;font-weight:800;
  background:linear-gradient(90deg,var(--accent),var(--purple));
  -webkit-background-clip:text;-webkit-text-fill-color:transparent;margin:0}
.nepse-sub{color:var(--dim);font-size:0.78rem;letter-spacing:.15em;text-transform:uppercase;margin-top:.25rem}
.mcard{background:var(--card);border:1px solid var(--border);border-radius:10px;
  padding:1rem 1.2rem;transition:border-color .2s}
.mcard:hover{border-color:var(--accent)}
.mlabel{font-size:.68rem;text-transform:uppercase;letter-spacing:.12em;color:var(--dim);margin-bottom:.35rem}
.mvalue{font-size:1.55rem;font-weight:700;color:var(--text);line-height:1}
.mdelta{font-size:.73rem;margin-top:.3rem}
.up{color:var(--green)} .down{color:var(--red)} .flat{color:var(--amber)}
.sec{font-family:'Syne',sans-serif;font-size:1.05rem;font-weight:700;color:var(--accent);
  letter-spacing:.05em;border-bottom:1px solid var(--border);padding-bottom:.45rem;margin:1.4rem 0 .9rem}
.badge{display:inline-block;padding:.15rem .6rem;border-radius:20px;
  font-size:.68rem;font-weight:600;letter-spacing:.08em;text-transform:uppercase}
.blive{background:rgba(0,230,118,.15);color:var(--green);border:1px solid var(--green)}
.bcached{background:rgba(255,179,0,.15);color:var(--amber);border:1px solid var(--amber)}
.bdemo{background:rgba(179,136,255,.15);color:var(--purple);border:1px solid var(--purple)}
.berror{background:rgba(255,82,82,.15);color:var(--red);border:1px solid var(--red)}
.srow{display:flex;align-items:center;gap:.7rem;padding:.55rem 0;border-bottom:1px solid var(--border);font-size:.8rem}
.dot{width:8px;height:8px;border-radius:50%}
.dg{background:var(--green);box-shadow:0 0 6px var(--green)}
.dr{background:var(--red);box-shadow:0 0 6px var(--red)}
.da{background:var(--amber);box-shadow:0 0 6px var(--amber)}
.sm-tbl{width:100%;border-collapse:collapse;font-size:.8rem}
.sm-tbl th{text-align:left;color:var(--dim);font-size:.68rem;text-transform:uppercase;
  letter-spacing:.1em;padding:.38rem .55rem;border-bottom:1px solid var(--border)}
.sm-tbl td{padding:.38rem .55rem;border-bottom:1px solid rgba(33,38,45,.5);vertical-align:middle}
.sm-tbl tr:hover td{background:rgba(0,212,255,.04)}
.sbar-o{background:var(--border);border-radius:4px;height:6px;width:75px;display:inline-block;vertical-align:middle}
.sbar-i{height:100%;border-radius:4px}
[data-testid="stSidebar"]{background:var(--card)!important;border-right:1px solid var(--border)}
.stButton>button{background:transparent;border:1px solid var(--accent);color:var(--accent);
  font-family:'JetBrains Mono',monospace;font-size:.78rem;letter-spacing:.1em;
  padding:.4rem 1.1rem;border-radius:6px;transition:all .2s}
.stButton>button:hover{background:var(--accent);color:var(--bg)}
::-webkit-scrollbar{width:5px;height:5px}
::-webkit-scrollbar-track{background:var(--bg)}
::-webkit-scrollbar-thumb{background:var(--border);border-radius:3px}
</style>
""", unsafe_allow_html=True)


# ── Demo data ──────────────────────────────────────────────────────────────────
def _demo_df() -> pd.DataFrame:
    np.random.seed(42)
    syms = ["NABIL","NICA","SBI","NMB","SANIMA","GBIME","KBL","MBL","PCBL","PRVU",
            "BOKL","CBL","CCBL","CZBIL","EBL","HBL","HIDCL","JBNL","MEGA","NBL",
            "ADBL","NGADI","NHPC","NIB","NIMB","NLG","NLIC","NLICL","NTC","ORTC",
            "PBBL","PLIC","PMHPL","RBBI","RBBL","SADBL","SCB","SIFC","SHL","SRBL",
            "TRH","UAIL","UPPER","VLBS","WOMI","YETI","SHINE","MLBSL","GILB","CITY"]
    n   = len(syms)
    ltp = np.random.uniform(100, 3000, n).round(2)
    pct = np.random.normal(0.5, 3.5, n).round(2).clip(-10, 10)
    vol = (np.random.exponential(50000, n) + 1000).astype(int)
    return pd.DataFrame({
        "symbol": syms, "ltp": ltp,
        "prev_close": (ltp / (1 + pct/100)).round(2),
        "pct_change": pct,
        "volume": vol,
        "turnover": (ltp * vol * np.random.uniform(.8, 1.2, n)).round(0),
        "transactions": (vol / np.random.uniform(10, 50, n)).astype(int),
        "open":  (ltp * np.random.uniform(.97, 1.03, n)).round(2),
        "high":  (ltp * np.random.uniform(1.0, 1.08, n)).round(2),
        "low":   (ltp * np.random.uniform(.92, 1.0,  n)).round(2),
    })


# ── Data loading (cached 5 min) ────────────────────────────────────────────────
@st.cache_data(ttl=300, show_spinner=False)
def load_data():
    df, source, status = get_market_data()
    if df is None or df.empty:
        return None, source, status
    return enrich_dataframe(df), source, status


# ── Helper renderers ───────────────────────────────────────────────────────────
def mcard(label, value, delta="", dc="flat"):
    st.markdown(
        f'<div class="mcard"><div class="mlabel">{label}</div>'
        f'<div class="mvalue">{value}</div>'
        f'{"<div class=mdelta " + dc + ">" + delta + "</div>" if delta else ""}'
        f'</div>', unsafe_allow_html=True)


def score_color(s):
    if s >= 75: return "#00E676"
    if s >= 55: return "#69F0AE"
    if s >= 35: return "#FFB300"
    return "#FF5252"


def sm_table(df, top_n=30):
    if "smart_money_score" not in df.columns:
        st.warning("Smart money scores not available.")
        return
    cols = [c for c in ["symbol","smart_money_score","smart_money_label",
                         "pct_change","volume","buy_pressure","sell_pressure"] if c in df.columns]
    top  = df[cols].dropna(subset=["smart_money_score"]).nlargest(top_n,"smart_money_score").reset_index(drop=True)
    rows = ""
    for _, r in top.iterrows():
        sc  = float(r.get("smart_money_score",0))
        col = score_color(sc)
        bw  = int(sc * 0.75)
        pct = r.get("pct_change",0)
        pc  = "up" if pct>0 else ("down" if pct<0 else "flat")
        ps  = f"+{pct:.2f}%" if pct>0 else f"{pct:.2f}%"
        rows += (
            f"<tr><td><b style='color:#E6EDF3'>{r['symbol']}</b></td>"
            f"<td><span style='color:{col};font-weight:700'>{sc:.1f}</span> "
            f"<span class='sbar-o'><span class='sbar-i' style='width:{bw}%;background:{col}'></span></span></td>"
            f"<td>{r.get('smart_money_label','—')}</td>"
            f"<td class='{pc}'>{ps}</td>"
            f"<td style='color:#8B949E'>{fmt_number(r.get('volume',0),1)}</td>"
            f"<td class='up'>{r.get('buy_pressure',0):.3f}</td>"
            f"<td class='down'>{r.get('sell_pressure',0):.3f}</td></tr>"
        )
    st.markdown(
        f"<table class='sm-tbl'><thead><tr>"
        f"<th>Symbol</th><th>Score</th><th>Signal</th>"
        f"<th>% Chg</th><th>Volume</th><th>Buy↑</th><th>Sell↓</th>"
        f"</tr></thead><tbody>{rows}</tbody></table>",
        unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
# SIDEBAR
# ══════════════════════════════════════════════════════════════════════════════
def render_sidebar(df, source, status, summary):
    with st.sidebar:
        st.markdown("""
        <div style='text-align:center;padding:1rem 0 .5rem'>
          <div style='font-family:Syne,sans-serif;font-size:1.4rem;font-weight:800;
                      background:linear-gradient(90deg,#00D4FF,#B388FF);
                      -webkit-background-clip:text;-webkit-text-fill-color:transparent'>
            NEPSE Intel
          </div>
          <div style='color:#8B949E;font-size:.63rem;letter-spacing:.2em;text-transform:uppercase;margin-top:.2rem'>
            Trading Dashboard
          </div>
        </div><hr style='border-color:#21262D;margin:.8rem 0'>
        """, unsafe_allow_html=True)

        if st.button("⟳  Refresh Data", use_container_width=True):
            st.cache_data.clear(); st.rerun()

        st.markdown("<div style='height:.6rem'></div>", unsafe_allow_html=True)
        st.markdown("<div style='font-size:.68rem;text-transform:uppercase;letter-spacing:.12em;color:#8B949E;margin-bottom:.45rem'>Data Sources</div>", unsafe_allow_html=True)

        for sname, is_src, priority in [
            ("NEPSE Official API", source=="NEPSE Official API" and status=="live", "Primary"),
            ("ShareSansar",        source=="ShareSansar"        and status=="live", "Secondary"),
            ("NepseAlpha",         source=="NepseAlpha"         and status=="live", "Tertiary"),
        ]:
            dc = "dg" if is_src else "dr"
            st.markdown(f"""
            <div class='srow'>
              <div class='dot {dc}'></div>
              <div style='flex:1;{"font-weight:600" if is_src else ""}'>{sname}</div>
              <div style='color:#8B949E;font-size:.63rem'>{priority}</div>
            </div>""", unsafe_allow_html=True)

        if summary and summary.get("total_stocks",0) > 0:
            st.markdown("<div style='height:.8rem'></div>", unsafe_allow_html=True)
            sent  = summary.get("market_sentiment","neutral")
            sc    = "#00E676" if sent=="bullish" else ("#FF5252" if sent=="bearish" else "#FFB300")
            st.markdown(f"""
            <div style='background:#161B22;border:1px solid #21262D;border-radius:8px;padding:.75rem;font-size:.78rem;margin-top:.5rem'>
              <div style='display:flex;justify-content:space-between;margin-bottom:.35rem'>
                <span style='color:#8B949E'>Sentiment</span>
                <span style='color:{sc};font-weight:700;text-transform:capitalize'>{sent}</span>
              </div>
              <div style='display:flex;justify-content:space-between;margin-bottom:.35rem'>
                <span style='color:#8B949E'>Advances</span><span style='color:#00E676'>{summary.get("advances",0)}</span>
              </div>
              <div style='display:flex;justify-content:space-between;margin-bottom:.35rem'>
                <span style='color:#8B949E'>Declines</span><span style='color:#FF5252'>{summary.get("declines",0)}</span>
              </div>
              <div style='display:flex;justify-content:space-between'>
                <span style='color:#8B949E'>Avg Chg</span>
                <span style='color:#E6EDF3'>{summary.get("avg_pct_change",0):+.2f}%</span>
              </div>
            </div>""", unsafe_allow_html=True)

        st.markdown("<div style='height:.8rem'></div>", unsafe_allow_html=True)
        st.markdown("<div style='font-size:.68rem;text-transform:uppercase;letter-spacing:.12em;color:#8B949E;margin-bottom:.45rem'>Filters</div>", unsafe_allow_html=True)
        min_vol   = st.number_input("Min Volume", min_value=0, value=0, step=1000)
        pct_range = st.slider("% Change Range", -15.0, 15.0, (-15.0, 15.0), 0.5)

        st.markdown(f"""
        <hr style='border-color:#21262D;margin:1rem 0 .5rem'>
        <div style='color:#8B949E;font-size:.62rem;text-align:center'>
          {datetime.now().strftime('%H:%M:%S')} · <span style='color:#00D4FF'>{source}</span>
        </div>""", unsafe_allow_html=True)

        return min_vol, pct_range


# ══════════════════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════════════════
def main():
    st.markdown("""
    <div class='nepse-header'>
      <h1 class='nepse-title'>NEPSE Trading Intelligence</h1>
      <div class='nepse-sub'>Real-Time Market Data · Smart Money Detection · Institutional Activity</div>
    </div>""", unsafe_allow_html=True)

    with st.spinner("Fetching market data…"):
        df, source, status = load_data()

    is_demo = False
    if df is None or df.empty:
        st.warning("⚠️ All live sources failed. Showing **demo data** for UI demonstration.")
        df      = enrich_dataframe(_demo_df())
        source  = "Demo (Synthetic)"
        status  = "demo"
        is_demo = True

    summary = market_summary(df)
    min_vol, pct_range = render_sidebar(df, source, status, summary)

    # Apply filters
    filt = df.copy()
    if "volume" in filt.columns:
        filt = filt[filt["volume"] >= min_vol]
    if "pct_change" in filt.columns:
        filt = filt[filt["pct_change"].between(pct_range[0], pct_range[1])]

    # Status bar
    bclass = {"live":"blive","demo":"bdemo"}.get(status.split()[0],"bcached")
    st.markdown(f"""
    <div style='display:flex;align-items:center;gap:.7rem;margin-bottom:1.3rem;
                padding:.55rem 1rem;background:#161B22;border-radius:8px;border:1px solid #21262D'>
      <span class='badge {bclass}'>{status}</span>
      <span style='color:#8B949E;font-size:.78rem'>Source: <b style='color:#E6EDF3'>{source}</b></span>
      <span style='color:#8B949E;font-size:.78rem'>·</span>
      <span style='color:#8B949E;font-size:.78rem'>{len(df)} stocks · {len(filt)} filtered</span>
    </div>""", unsafe_allow_html=True)

    tabs = st.tabs(["📊 Market Overview","🧠 Smart Money","📈 Top Movers",
                    "🔥 Most Active","💧 Liquidity","⚙️ Data Source"])

    # ── Tab 1: Market Overview ─────────────────────────────────────────────────
    with tabs[0]:
        st.markdown("<div class='sec'>Market Overview</div>", unsafe_allow_html=True)
        c1,c2,c3,c4,c5,c6 = st.columns(6)
        with c1: mcard("Total Stocks", str(summary.get("total_stocks",0)))
        with c2: mcard("Advances",  str(summary.get("advances",0)),  "↑ Positive","up")
        with c3: mcard("Declines",  str(summary.get("declines",0)),  "↓ Negative","down")
        with c4: mcard("Unchanged", str(summary.get("unchanged",0)), "→ Flat","flat")
        avg = summary.get("avg_pct_change",0)
        with c5: mcard("Avg % Chg", f"{avg:+.2f}%", delta_class="up" if avg>=0 else "down")
        with c6: mcard("Breadth", f"{summary.get('breadth',0):.3f}")

        st.markdown("<div style='height:.6rem'></div>", unsafe_allow_html=True)
        c1,c2,c3 = st.columns(3)
        with c1: mcard("Total Volume",   fmt_number(summary.get("total_volume",0),2))
        with c2: mcard("Total Turnover", "Rs " + fmt_number(summary.get("total_turnover",0),2))
        sent = summary.get("market_sentiment","neutral").upper()
        sc2  = "up" if sent=="BULLISH" else ("down" if sent=="BEARISH" else "flat")
        with c3: mcard("Market Sentiment", sent, delta_class=sc2)

        st.markdown("<div style='height:.8rem'></div>", unsafe_allow_html=True)
        ca,cb = st.columns(2)
        with ca:
            st.plotly_chart(market_breadth_gauge(
                summary.get("advances",0), summary.get("declines",0), summary.get("unchanged",0)
            ), use_container_width=True, config={"displayModeBar":False})
        with cb:
            st.plotly_chart(pct_change_distribution(filt),
                            use_container_width=True, config={"displayModeBar":False})

        st.markdown("<div class='sec'>Market Highlights</div>", unsafe_allow_html=True)
        h1,h2,h3 = st.columns(3)
        tg = summary.get("top_gainer",{})
        tl = summary.get("top_loser",{})
        ma = summary.get("most_active",{})
        with h1:
            st.markdown(f"""<div class='mcard' style='border-color:#00E676'>
              <div class='mlabel'>🏆 Top Gainer</div>
              <div class='mvalue' style='color:#00E676'>{tg.get("symbol","—")}</div>
              <div class='mdelta up'>+{tg.get("pct_change",0):.2f}%
                {"· Rs "+fmt_number(tg.get("ltp",0),2) if tg.get("ltp") else ""}</div>
            </div>""", unsafe_allow_html=True)
        with h2:
            st.markdown(f"""<div class='mcard' style='border-color:#FF5252'>
              <div class='mlabel'>📉 Top Loser</div>
              <div class='mvalue' style='color:#FF5252'>{tl.get("symbol","—")}</div>
              <div class='mdelta down'>{tl.get("pct_change",0):.2f}%
                {"· Rs "+fmt_number(tl.get("ltp",0),2) if tl.get("ltp") else ""}</div>
            </div>""", unsafe_allow_html=True)
        with h3:
            st.markdown(f"""<div class='mcard' style='border-color:#00D4FF'>
              <div class='mlabel'>🔥 Most Active</div>
              <div class='mvalue' style='color:#00D4FF'>{ma.get("symbol","—")}</div>
              <div class='mdelta flat'>Vol: {fmt_number(ma.get("volume",0),1)}</div>
            </div>""", unsafe_allow_html=True)

    # ── Tab 2: Smart Money ─────────────────────────────────────────────────────
    with tabs[1]:
        st.markdown("<div class='sec'>Smart Money & Institutional Signals ⭐</div>", unsafe_allow_html=True)
        if is_demo:
            st.info("📌 Showing synthetic demo data. Connect a live source for real signals.")
        ca,cb = st.columns(2)
        with ca:
            st.plotly_chart(smart_money_bar(filt,20), use_container_width=True, config={"displayModeBar":False})
        with cb:
            st.plotly_chart(smart_money_heatmap(filt,50), use_container_width=True, config={"displayModeBar":False})
        st.markdown("<div class='sec'>Top Smart Money Candidates</div>", unsafe_allow_html=True)
        sm_table(filt, 30)
        with st.expander("📐 Score Methodology"):
            st.markdown("""
| Component | Weight | Signal |
|---|---|---|
| Price Momentum | 30% | % change percentile rank |
| Volume Spike | 30% | Volume vs market median |
| Persistence | 25% | abs(% change) × volume rank |
| Liquidity Impact | 15% | Inverted price impact proxy |

**Score 0–100:** 🔥 75–100 Strong · 📈 55–74 Moderate · ➡️ 35–54 Neutral · 📉 0–34 Weak
            """)

    # ── Tab 3: Top Movers ──────────────────────────────────────────────────────
    with tabs[2]:
        st.markdown("<div class='sec'>Top Gainers & Losers</div>", unsafe_allow_html=True)
        st.plotly_chart(top_movers_chart(filt,15), use_container_width=True, config={"displayModeBar":False})
        ca,cb = st.columns(2)
        with ca:
            st.markdown("<div class='sec'>🟢 Top 15 Gainers</div>", unsafe_allow_html=True)
            if "pct_change" in filt.columns:
                cols = [c for c in ["symbol","pct_change","ltp","volume"] if c in filt.columns]
                g = filt[cols].dropna(subset=["pct_change"]).nlargest(15,"pct_change").reset_index(drop=True)
                g.index += 1
                st.dataframe(g.style.format({c:"{:.2f}" for c in g.select_dtypes("number").columns}), use_container_width=True)
        with cb:
            st.markdown("<div class='sec'>🔴 Top 15 Losers</div>", unsafe_allow_html=True)
            if "pct_change" in filt.columns:
                cols = [c for c in ["symbol","pct_change","ltp","volume"] if c in filt.columns]
                lo = filt[cols].dropna(subset=["pct_change"]).nsmallest(15,"pct_change").reset_index(drop=True)
                lo.index += 1
                st.dataframe(lo.style.format({c:"{:.2f}" for c in lo.select_dtypes("number").columns}), use_container_width=True)

    # ── Tab 4: Most Active ─────────────────────────────────────────────────────
    with tabs[3]:
        st.markdown("<div class='sec'>Most Active Stocks</div>", unsafe_allow_html=True)
        st.plotly_chart(volume_bar_chart(filt,25), use_container_width=True, config={"displayModeBar":False})
        if "large_activity_flag" in filt.columns:
            flagged = filt[filt["large_activity_flag"]==True]
            if not flagged.empty:
                st.markdown(f"<div class='sec'>⚡ Unusual Volume ({len(flagged)} stocks)</div>", unsafe_allow_html=True)
                sc = [c for c in ["symbol","volume","volume_ratio","pct_change","turnover"] if c in flagged.columns]
                st.dataframe(flagged[sc].sort_values("volume",ascending=False).reset_index(drop=True)
                             .style.format({c:"{:.2f}" for c in flagged[sc].select_dtypes("number").columns}),
                             use_container_width=True)

    # ── Tab 5: Liquidity ───────────────────────────────────────────────────────
    with tabs[4]:
        st.markdown("<div class='sec'>Liquidity & Market Impact</div>", unsafe_allow_html=True)
        ca,cb = st.columns([1.2,.8])
        with ca:
            st.plotly_chart(price_volume_impact_chart(filt,30), use_container_width=True, config={"displayModeBar":False})
        with cb:
            if "volatility_rank" in filt.columns:
                st.markdown("<div class='sec'>⚡ Top Volatile</div>", unsafe_allow_html=True)
                vc = [c for c in ["symbol","volatility_rank","pct_change"] if c in filt.columns]
                vt = filt[vc].dropna().nlargest(15,"volatility_rank").reset_index(drop=True)
                vt.index += 1
                st.dataframe(vt.style.format({c:"{:.4f}" for c in vt.select_dtypes("number").columns}), use_container_width=True)
        if "turnover_spike_flag" in filt.columns:
            spikes = filt[filt["turnover_spike_flag"]==True]
            if not spikes.empty:
                st.markdown(f"<div class='sec'>💹 Turnover Spikes ({len(spikes)})</div>", unsafe_allow_html=True)
                sc = [c for c in ["symbol","turnover","pct_change","volume","price_impact"] if c in spikes.columns]
                st.dataframe(spikes[sc].sort_values("turnover",ascending=False).reset_index(drop=True)
                             .style.format({c:"{:.4f}" for c in spikes[sc].select_dtypes("number").columns}),
                             use_container_width=True)

    # ── Tab 6: Data Source ─────────────────────────────────────────────────────
    with tabs[5]:
        st.markdown("<div class='sec'>Data Source Status & Pipeline</div>", unsafe_allow_html=True)
        ca,cb = st.columns(2)
        with ca:
            st.markdown("#### 🌐 Source Registry")
            for sname, url, priority, active in [
                ("NEPSE Official API","https://nepalstock.com.np/api","Primary",   source=="NEPSE Official API"),
                ("ShareSansar",       "https://www.sharesansar.com/today-share-price","Secondary",source=="ShareSansar"),
                ("NepseAlpha",        "https://nepsealpha.com/nepse-data","Tertiary",source=="NepseAlpha"),
            ]:
                dc = "dg" if active else "dr"
                st.markdown(f"""
                <div style='background:#161B22;border:1px solid #21262D;border-radius:8px;
                            padding:.75rem 1rem;margin-bottom:.4rem'>
                  <div style='display:flex;align-items:center;gap:.5rem;margin-bottom:.25rem'>
                    <div class='dot {dc}'></div>
                    <span style='font-weight:700;color:#E6EDF3'>{sname}</span>
                    <span style='margin-left:auto;color:#8B949E;font-size:.65rem'>{priority}</span>
                  </div>
                  <div style='color:#8B949E;font-size:.7rem;font-family:monospace'>{url}</div>
                  {"<div style='color:#00E676;font-size:.7rem;margin-top:.25rem'>✓ Active</div>" if active else ""}
                </div>""", unsafe_allow_html=True)
        with cb:
            st.markdown("#### 📋 Data Sample")
            sc = [c for c in ["symbol","ltp","pct_change","volume","turnover","smart_money_score"] if c in df.columns]
            if sc:
                st.dataframe(df[sc].head(20).style.format({c:"{:.3f}" for c in df[sc].select_dtypes("number").columns}),
                             use_container_width=True)

        st.markdown("#### 🗂️ Schema")
        schema = pd.DataFrame([
            {"Column": c, "Present": "✅" if c in df.columns else "❌",
             "Sample": str(df[c].iloc[0]) if c in df.columns and not df.empty else "—"}
            for c in ["symbol","ltp","pct_change","volume","turnover","transactions",
                      "open","high","low","prev_close","buy_pressure","sell_pressure",
                      "volume_ratio","smart_money_score","price_impact","volatility_rank"]
        ])
        st.dataframe(schema, use_container_width=True)

        with st.expander("🔁 Pipeline"):
            st.code("""
get_market_data()
├── fetch_from_api()           # nepalstock.com.np (Primary)
├── fetch_from_sharesansar()   # sharesansar.com   (Fallback 1)
├── fetch_from_nepsealpha()    # nepsealpha.com    (Fallback 2)
└── load_cache()               # disk cache        (Fallback 3)

enrich_dataframe()
├── order_flow_signal()        → buy_pressure, sell_pressure, persistence_score
├── detect_large_activity()    → volume_ratio, large_activity_flag
├── liquidity_metrics()        → volatility_rank, price_impact, turnover_spike_flag
└── smart_money_score()        → smart_money_score (0–100), smart_money_label
            """, language="text")


if __name__ == "__main__":
    main()
