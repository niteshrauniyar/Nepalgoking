"""All Plotly charts for the NEPSE dashboard. No matplotlib used."""

import logging
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

logger = logging.getLogger(__name__)

# ── Theme ──────────────────────────────────────────────────────────────────────
BG    = "#0D1117"
CARD  = "#161B22"
GRID  = "#21262D"
ACNT  = "#00D4FF"
GRN   = "#00E676"
RED   = "#FF5252"
AMBR  = "#FFB300"
PURP  = "#B388FF"
TEXT  = "#E6EDF3"
DIM   = "#8B949E"
FONT  = dict(family="'JetBrains Mono','Courier New',monospace", color=TEXT, size=11)
AX    = dict(showgrid=True, gridcolor=GRID, zeroline=False, color=DIM, tickfont=dict(size=10))
LAY   = dict(paper_bgcolor=BG, plot_bgcolor=CARD, font=FONT,
             margin=dict(l=45, r=20, t=50, b=40),
             legend=dict(bgcolor="rgba(0,0,0,0)", borderwidth=0))


def _empty(msg="No data available") -> go.Figure:
    fig = go.Figure()
    fig.add_annotation(text=msg, x=0.5, y=0.5, xref="paper", yref="paper",
                       showarrow=False, font=dict(size=15, color=DIM))
    fig.update_layout(height=300, **LAY)
    return fig


def _has(df, *cols):
    return not df.empty and all(c in df.columns for c in cols)


# ── Volume Bar ─────────────────────────────────────────────────────────────────

def volume_bar_chart(df: pd.DataFrame, top_n: int = 25) -> go.Figure:
    try:
        if not _has(df, "symbol", "volume"):
            return _empty("Volume data unavailable")
        plot = df.dropna(subset=["volume"]).nlargest(top_n, "volume").sort_values("volume")
        colors = ([GRN if x >= 0 else RED for x in plot["pct_change"]]
                  if "pct_change" in plot.columns else [ACNT] * len(plot))
        fig = go.Figure(go.Bar(
            x=plot["volume"], y=plot["symbol"], orientation="h",
            marker=dict(color=colors, opacity=0.85),
            hovertemplate="<b>%{y}</b><br>Volume: %{x:,.0f}<extra></extra>",
        ))
        fig.update_layout(title=dict(text=f"📊 Top {top_n} by Volume", font=dict(size=14, color=ACNT)),
                          xaxis=dict(**AX, title="Volume"), yaxis=dict(**AX),
                          height=max(350, top_n * 20), **LAY)
        return fig
    except Exception as e:
        logger.error(f"volume_bar_chart: {e}")
        return _empty("Chart error")


# ── % Change Distribution ──────────────────────────────────────────────────────

def pct_change_distribution(df: pd.DataFrame) -> go.Figure:
    try:
        if not _has(df, "pct_change"):
            return _empty("No % change data")
        pct = df["pct_change"].dropna()
        fig = go.Figure(go.Histogram(
            x=pct, nbinsx=40,
            marker=dict(color=[GRN if x >= 0 else RED for x in pct], line=dict(width=0)),
            opacity=0.75,
            hovertemplate="Change: %{x:.2f}%<br>Count: %{y}<extra></extra>",
        ))
        fig.add_vline(x=0, line=dict(color=AMBR, dash="dash", width=1.5))
        fig.add_vline(x=pct.mean(), line=dict(color=ACNT, dash="dot", width=1.5),
                      annotation_text=f"μ={pct.mean():.2f}%", annotation_font_color=ACNT)
        fig.update_layout(title=dict(text="📉 % Change Distribution", font=dict(size=14, color=ACNT)),
                          xaxis=dict(**AX, title="% Change"), yaxis=dict(**AX, title="Count"),
                          showlegend=False, height=350, bargap=0.05, **LAY)
        return fig
    except Exception as e:
        logger.error(f"pct_change_distribution: {e}")
        return _empty("Chart error")


# ── Smart Money Heatmap ────────────────────────────────────────────────────────

def smart_money_heatmap(df: pd.DataFrame, top_n: int = 50) -> go.Figure:
    try:
        if not _has(df, "symbol", "pct_change", "volume"):
            return _empty("Insufficient data for heatmap")
        plot = df.dropna(subset=["pct_change", "volume"]).head(top_n)
        vol   = plot["volume"].fillna(0)
        sizes = ((vol - vol.min()) / (vol.max() - vol.min() + 1) * 28 + 6).clip(6, 40)
        score = plot.get("smart_money_score", pd.Series(50.0, index=plot.index))
        fig = go.Figure(go.Scatter(
            x=plot["pct_change"], y=vol,
            mode="markers",
            marker=dict(size=sizes, color=score, colorscale="Plasma",
                        showscale=True, colorbar=dict(title="Score", tickfont=dict(color=DIM)),
                        line=dict(width=0.4, color=BG), opacity=0.87),
            customdata=plot[["symbol"]].values,
            hovertemplate="<b>%{customdata[0]}</b><br>% Chg: %{x:.2f}%<br>"
                          "Volume: %{y:,.0f}<br>Score: %{marker.color:.1f}<extra></extra>",
        ))
        fig.add_vline(x=0, line=dict(color=GRID, width=1))
        fig.add_hline(y=vol.median(), line=dict(color=GRID, dash="dot", width=1))
        fig.update_layout(title=dict(text="🔥 Smart Money Heatmap", font=dict(size=14, color=ACNT)),
                          xaxis=dict(**AX, title="% Change"),
                          yaxis=dict(**AX, title="Volume", type="log"),
                          height=440, **LAY)
        return fig
    except Exception as e:
        logger.error(f"smart_money_heatmap: {e}")
        return _empty("Chart error")


# ── Price vs Volume Impact ─────────────────────────────────────────────────────

def price_volume_impact_chart(df: pd.DataFrame, top_n: int = 30) -> go.Figure:
    try:
        if not _has(df, "symbol", "ltp", "volume"):
            return _empty("Insufficient data")
        plot  = df.dropna(subset=["ltp", "volume"]).nlargest(top_n, "volume")
        impact= plot.get("price_impact", pd.Series(0.0, index=plot.index)).fillna(0)
        sizes = (np.log1p(impact) * 10 + 5).clip(5, 50)
        colors= ([GRN if x >= 0 else RED for x in plot["pct_change"]]
                 if "pct_change" in plot.columns else [ACNT] * len(plot))
        fig = go.Figure(go.Scatter(
            x=plot["volume"], y=plot["ltp"],
            mode="markers+text", text=plot["symbol"],
            textposition="top center", textfont=dict(size=8, color=DIM),
            marker=dict(size=sizes, color=colors, opacity=0.82, line=dict(width=0.4, color=BG)),
            hovertemplate="<b>%{text}</b><br>LTP: Rs %{y:,.2f}<br>Volume: %{x:,.0f}<extra></extra>",
        ))
        fig.update_layout(title=dict(text="💡 Price vs Volume Impact", font=dict(size=14, color=ACNT)),
                          xaxis=dict(**AX, title="Volume", type="log"),
                          yaxis=dict(**AX, title="LTP (Rs)"),
                          height=420, **LAY)
        return fig
    except Exception as e:
        logger.error(f"price_volume_impact_chart: {e}")
        return _empty("Chart error")


# ── Market Breadth Donut ───────────────────────────────────────────────────────

def market_breadth_gauge(advances: int, declines: int, unchanged: int) -> go.Figure:
    try:
        total = advances + declines + unchanged
        if total == 0:
            return _empty("No breadth data")
        ratio = advances / (advances + declines) if (advances + declines) > 0 else 0.5
        sent  = "Bullish" if ratio > 0.55 else ("Bearish" if ratio < 0.45 else "Neutral")
        sc    = GRN if sent == "Bullish" else (RED if sent == "Bearish" else AMBR)
        fig = go.Figure(go.Pie(
            labels=["Advances", "Declines", "Unchanged"],
            values=[advances, declines, unchanged],
            hole=0.60,
            marker=dict(colors=[GRN, RED, AMBR]),
            textinfo="label+percent",
            hovertemplate="%{label}: %{value} stocks (%{percent})<extra></extra>",
        ))
        fig.update_layout(
            title=dict(text="🌡️ Market Breadth", font=dict(size=14, color=ACNT)),
            annotations=[dict(text=f"<b>{sent}</b>", x=0.5, y=0.5,
                              font=dict(size=17, color=sc), showarrow=False)],
            height=320, **LAY,
        )
        return fig
    except Exception as e:
        logger.error(f"market_breadth_gauge: {e}")
        return _empty("Chart error")


# ── Top Movers ─────────────────────────────────────────────────────────────────

def top_movers_chart(df: pd.DataFrame, top_n: int = 10) -> go.Figure:
    try:
        if not _has(df, "symbol", "pct_change"):
            return _empty("No data available")
        valid   = df.dropna(subset=["pct_change", "symbol"])
        gainers = valid.nlargest(top_n, "pct_change").sort_values("pct_change")
        losers  = valid.nsmallest(top_n, "pct_change").sort_values("pct_change", ascending=False)
        fig = make_subplots(rows=1, cols=2, subplot_titles=("🟢 Top Gainers", "🔴 Top Losers"))
        fig.add_trace(go.Bar(x=gainers["pct_change"], y=gainers["symbol"], orientation="h",
                             marker=dict(color=GRN, opacity=0.85),
                             hovertemplate="<b>%{y}</b>: %{x:.2f}%<extra></extra>"), row=1, col=1)
        fig.add_trace(go.Bar(x=losers["pct_change"].abs(), y=losers["symbol"], orientation="h",
                             marker=dict(color=RED, opacity=0.85),
                             hovertemplate="<b>%{y}</b>: -%{x:.2f}%<extra></extra>"), row=1, col=2)
        fig.update_layout(title=dict(text="📈 Top Movers", font=dict(size=14, color=ACNT)),
                          showlegend=False, height=380, **LAY)
        for ax in ("xaxis", "xaxis2", "yaxis", "yaxis2"):
            fig.update_layout(**{ax: AX})
        return fig
    except Exception as e:
        logger.error(f"top_movers_chart: {e}")
        return _empty("Chart error")


# ── Smart Money Bar ────────────────────────────────────────────────────────────

def smart_money_bar(df: pd.DataFrame, top_n: int = 20) -> go.Figure:
    try:
        if not _has(df, "symbol", "smart_money_score"):
            return _empty("Smart money scores unavailable")
        plot = (df[["symbol", "smart_money_score"]].dropna()
                .nlargest(top_n, "smart_money_score").sort_values("smart_money_score"))
        vals = plot["smart_money_score"] / 100
        colors = [f"rgba({int(255*(1-v))},{int(200*v)},{int(255*v)},0.88)" for v in vals]
        fig = go.Figure(go.Bar(
            x=plot["smart_money_score"], y=plot["symbol"], orientation="h",
            marker=dict(color=colors),
            hovertemplate="<b>%{y}</b><br>Score: %{x:.1f}/100<extra></extra>",
        ))
        fig.update_layout(title=dict(text="🧠 Smart Money Score", font=dict(size=14, color=ACNT)),
                          xaxis=dict(**AX, title="Score (0–100)", range=[0, 105]),
                          yaxis=dict(**AX),
                          height=max(300, top_n * 22), **LAY)
        return fig
    except Exception as e:
        logger.error(f"smart_money_bar: {e}")
        return _empty("Chart error")
