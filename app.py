"""
app.py
======
Synchronized Geopolitical-Equity Alpha Inference Architecture
--------------------------------------------------------------
3-column Streamlit Cloud dashboard:

  Col 1 – Candlestick + Alpha Prediction overlay
  Col 2 – Geopolitical Sentiment Pulse gauge
  Col 3 – Intelligence Feed (Reuters / FT / Al Jazeera) + Metrics

Run locally:
    streamlit run app.py

Deploy: push to GitHub → connect to share.streamlit.io
        add NEWSAPI_KEY to st.secrets in the Streamlit Cloud dashboard.
"""

from __future__ import annotations

import warnings
warnings.filterwarnings("ignore")

import logging
from datetime import datetime, timedelta
from typing import Optional

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit as st

from data_loader import (
    SynchronizedDataLoader,
    AlignedBundle,
    NewsArticle,
    APPROVED_DOMAINS,
    DOMAIN_DISPLAY,
    LOOKBACK_DAYS,
)
from model_utils import (
    AlphaInferenceModel,
    PriceOnlyBaseline,
    BacktestEngine,
    InferenceResult,
    AblationResult,
    build_model,
    run_inference,
    CRISIS_VOL_THRESHOLD,
)

logger = logging.getLogger(__name__)

# ─────────────────────────────────────────────────────────────────────────────
# Page configuration
# ─────────────────────────────────────────────────────────────────────────────

st.set_page_config(
    page_title  = "Geopolitical Alpha | SGEAIA",
    page_icon   = "🛰️",
    layout      = "wide",
    initial_sidebar_state = "expanded",
)

# ─────────────────────────────────────────────────────────────────────────────
# Custom CSS  — dark terminal / Bloomberg aesthetic
# ─────────────────────────────────────────────────────────────────────────────

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@400;600;700&family=IBM+Plex+Sans:wght@300;400;500;600&display=swap');

html, body, [class*="css"] {
    font-family: 'IBM Plex Sans', sans-serif;
    background-color: #080c14;
    color: #b0bec5;
}
h1,h2,h3,h4 { font-family: 'IBM Plex Mono', monospace; }

/* Header banner */
.sgeaia-header {
    background: linear-gradient(90deg, #0a1628 0%, #091220 60%, #0a1628 100%);
    border-bottom: 1px solid #1a3050;
    padding: 0.6rem 1.2rem;
    display: flex; align-items: center; gap: 1rem;
}
.sgeaia-title {
    font-family: 'IBM Plex Mono', monospace;
    font-size: 1.05rem; font-weight: 700;
    color: #00b4d8; letter-spacing: 0.06em; text-transform: uppercase;
}
.sgeaia-sub {
    font-size: 0.72rem; color: #546e7a;
    font-family: 'IBM Plex Mono', monospace;
    letter-spacing: 0.05em;
}
.ts-badge {
    margin-left: auto;
    font-family: 'IBM Plex Mono', monospace;
    font-size: 0.68rem; color: #37474f;
}

/* Section labels */
.sec-label {
    font-family: 'IBM Plex Mono', monospace;
    font-size: 0.62rem; color: #37474f;
    text-transform: uppercase; letter-spacing: 0.14em;
    margin-bottom: 0.3rem;
}

/* Panel card */
.panel {
    background: #0d1521;
    border: 1px solid #162030;
    border-radius: 6px; padding: 0.8rem; margin-bottom: 0.6rem;
}

/* Alpha value */
.alpha-big { font-family: 'IBM Plex Mono', monospace; font-size: 3rem; font-weight: 700; line-height: 1; }
.alpha-long    { color: #00e676; }
.alpha-short   { color: #ff1744; }
.alpha-flat    { color: #546e7a; }

/* Crisis Mode banner */
.crisis-banner {
    background: linear-gradient(90deg,#3d0000,#1a0000);
    border: 1px solid #ff1744; border-radius: 5px;
    padding: 0.4rem 0.8rem; text-align: center;
    font-family: 'IBM Plex Mono', monospace;
    font-size: 0.78rem; color: #ff6d6d;
    letter-spacing: 0.08em; animation: pulse 2s infinite;
}
@keyframes pulse {
    0%   { border-color: #ff1744; }
    50%  { border-color: #ff6d00; }
    100% { border-color: #ff1744; }
}

/* Intelligence feed */
.feed-item {
    border-left: 3px solid #00b4d8;
    background: #0a1220;
    border-radius: 0 4px 4px 0;
    padding: 0.45rem 0.7rem;
    margin-bottom: 0.35rem;
    font-size: 0.77rem; line-height: 1.45; color: #90a4ae;
}
.feed-item.reuters  { border-left-color: #ff8f00; }
.feed-item.ft       { border-left-color: #f9a825; }
.feed-item.aljazeera{ border-left-color: #00b4d8; }
.feed-source {
    font-family: 'IBM Plex Mono', monospace;
    font-size: 0.58rem; color: #546e7a;
    text-transform: uppercase; letter-spacing: 0.1em;
    margin-bottom: 2px;
}

/* Metric pill */
.metric-row { display: flex; gap: 0.5rem; margin-bottom: 0.5rem; }
.mpill {
    flex: 1; background: #0a1220; border: 1px solid #162030;
    border-radius: 5px; padding: 0.4rem 0.6rem;
}
.mpill-label {
    font-family: 'IBM Plex Mono', monospace;
    font-size: 0.58rem; color: #37474f;
    text-transform: uppercase; letter-spacing: 0.1em;
}
.mpill-value {
    font-family: 'IBM Plex Mono', monospace;
    font-size: 1rem; color: #eceff1; font-weight: 600;
}

/* Ablation table */
.ablation-table { width: 100%; border-collapse: collapse; font-family: 'IBM Plex Mono', monospace; font-size: 0.75rem; }
.ablation-table th { color: #37474f; border-bottom: 1px solid #162030; padding: 4px 8px; text-align: right; }
.ablation-table th:first-child { text-align: left; }
.ablation-table td { padding: 4px 8px; text-align: right; color: #90a4ae; }
.ablation-table td:first-child { text-align: left; color: #546e7a; }
.ablation-table tr:hover td { background: #0d1a28; }
.better { color: #00e676; }
.worse  { color: #ff5252; }

/* Streamlit overrides */
section[data-testid="stSidebar"] { background: #080c14; border-right: 1px solid #162030; }
[data-testid="metric-container"] { background: #0d1521; border: 1px solid #162030; border-radius: 5px; }
div.stButton > button { font-family: 'IBM Plex Mono', monospace; font-size: 0.8rem; border-radius: 4px; }
div[data-testid="stHorizontalBlock"] > div { padding: 0 6px; }
</style>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────────────────────
# Plotly theme
# ─────────────────────────────────────────────────────────────────────────────

PTHEME = dict(
    paper_bgcolor = "rgba(0,0,0,0)",
    plot_bgcolor  = "#0a1220",
    font          = dict(family="IBM Plex Mono, monospace", color="#546e7a", size=10),
    margin        = dict(l=8, r=8, t=28, b=8),
    xaxis         = dict(gridcolor="#111d2e", zerolinecolor="#111d2e", showgrid=True),
    yaxis         = dict(gridcolor="#111d2e", zerolinecolor="#111d2e", showgrid=True),
)


# ─────────────────────────────────────────────────────────────────────────────
# Session state initialisation
# ─────────────────────────────────────────────────────────────────────────────

for key, default in [
    ("bundle",   None), ("result",   None),
    ("ablation", None), ("loaded",   False),
    ("backtest", None),
]:
    if key not in st.session_state:
        st.session_state[key] = default


# ─────────────────────────────────────────────────────────────────────────────
# Sidebar
# ─────────────────────────────────────────────────────────────────────────────

with st.sidebar:
    st.markdown("### ⚙ Configuration")

    ticker = st.text_input("Ticker", value="SPY").upper().strip()

    lookback = st.slider(
        "Lookback window (calendar days)",
        min_value=90, max_value=365, value=120, step=15,
    )

    st.divider()
    st.caption("**Approved news sources (hard-coded)**")
    for d in APPROVED_DOMAINS:
        st.caption(f"• {DOMAIN_DISPLAY[d]}")

    st.divider()
    run_backtest_cb = st.checkbox("Run Backtesting & Ablation Study", value=False)
    st.caption("⚠ Backtesting iterates over all sequence windows — may take 1–3 min.")

    st.divider()
    run_btn = st.button("🛰 Run Inference", use_container_width=True, type="primary")
    st.caption(
        "API keys are read from `st.secrets` (Streamlit Cloud) "
        "or environment variables (local)."
    )


# ─────────────────────────────────────────────────────────────────────────────
# Header
# ─────────────────────────────────────────────────────────────────────────────

st.markdown(f"""
<div class="sgeaia-header">
  <div>
    <div class="sgeaia-title">🛰 Synchronized Geopolitical-Equity Alpha Inference Architecture</div>
    <div class="sgeaia-sub">Reuters · Financial Times · Al Jazeera  ·  FinBERT-Tone  ·  Bi-LSTM  ·  Crisis Mode Fusion</div>
  </div>
  <div class="ts-badge">{datetime.utcnow().strftime('%Y-%m-%d %H:%M UTC')}</div>
</div>
""", unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────────────────────
# Data + model loading
# ─────────────────────────────────────────────────────────────────────────────

@st.cache_data(ttl=300, show_spinner="Synchronizing market & geopolitical data…")
def load_bundle(ticker: str, lookback: int) -> AlignedBundle:
    loader = SynchronizedDataLoader(ticker=ticker, seq_len=LOOKBACK_DAYS)
    return loader.load(lookback_days=lookback)


# ─────────────────────────────────────────────────────────────────────────────
# Run button logic
# ─────────────────────────────────────────────────────────────────────────────

if run_btn:
    with st.spinner("Loading data…"):
        try:
            bundle = load_bundle(ticker, lookback)
            st.session_state.bundle = bundle
        except Exception as exc:
            st.error(f"Data loading failed: {exc}")
            st.stop()

    with st.spinner("Building models…"):
        model, baseline = build_model(num_features=bundle.ohlcv.num_features)

    with st.spinner("Running inference…"):
        try:
            loader     = SynchronizedDataLoader(ticker=ticker, seq_len=LOOKBACK_DAYS)
            latest_seq = loader.latest_window(bundle)

            latest_text = bundle.news_daily["headline_concat"].iloc[-1] or "No headlines."
            latest_text = latest_text[:512]

            vol_series  = bundle.ohlcv.features.get("Volatility_10d", pd.Series([0.0]))
            current_vol = float(vol_series.dropna().iloc[-1]) if len(vol_series.dropna()) else 0.0

            result = run_inference(model, latest_seq, latest_text, current_vol=current_vol)
            st.session_state.result = result
            st.session_state.loaded = True
        except Exception as exc:
            st.error(f"Inference failed: {exc}")
            st.stop()

    if run_backtest_cb:
        with st.spinner("Running backtesting & ablation study…"):
            try:
                X, y, dates = loader.build_sequences(bundle)
                texts = bundle.news_daily["headline_concat"].iloc[
                    len(bundle.news_daily) - len(y) :
                ].fillna("").tolist()
                vols_arr = bundle.ohlcv.features["Volatility_10d"].iloc[
                    len(bundle.ohlcv.features) - len(y) :
                ].fillna(0.0).values.astype(np.float32)

                engine   = BacktestEngine(model, baseline, num_features=bundle.ohlcv.num_features)
                ablation = engine.ablation_study(X, y, texts, vols_arr)
                st.session_state.ablation = ablation
                st.session_state.backtest = ablation.hybrid
            except Exception as exc:
                st.warning(f"Backtesting failed (non-fatal): {exc}")


# ─────────────────────────────────────────────────────────────────────────────
# Helper colour functions
# ─────────────────────────────────────────────────────────────────────────────

def _alpha_class(alpha: float) -> str:
    if alpha > 0.1:  return "alpha-long"
    if alpha < -0.1: return "alpha-short"
    return "alpha-flat"

def _feed_class(domain: str) -> str:
    return {"reuters.com": "reuters", "ft.com": "ft", "aljazeera.com": "aljazeera"}.get(domain, "")

def _pct(v: float, decimals: int = 1) -> str:
    return f"{v:.{decimals}%}"

def _signed(v: float, decimals: int = 3) -> str:
    return f"{v:+.{decimals}f}"


# ─────────────────────────────────────────────────────────────────────────────
# 3-COLUMN MAIN LAYOUT
# ─────────────────────────────────────────────────────────────────────────────

col1, col2, col3 = st.columns([1.7, 1.0, 1.3], gap="medium")


# ══════════════════════════════════════════════════════════════════════════════
# COLUMN 1 – Candlestick + Alpha Prediction Overlay
# ══════════════════════════════════════════════════════════════════════════════

with col1:
    st.markdown(f'<div class="sec-label">📈 {ticker} — Price & Alpha Overlay</div>', unsafe_allow_html=True)

    if not st.session_state.loaded:
        st.info("Configure the sidebar and press **Run Inference** to begin.")
    else:
        bundle = st.session_state.bundle
        result = st.session_state.result
        eq     = bundle.ohlcv.raw

        # ── Candlestick ──────────────────────────────────────────────────
        fig = make_subplots(
            rows=3, cols=1,
            shared_xaxes=True,
            row_heights=[0.55, 0.25, 0.20],
            vertical_spacing=0.04,
            subplot_titles=["", "Volume", "Volatility (10d ann.)"],
        )

        fig.add_trace(go.Candlestick(
            x=eq.index, open=eq["Open"], high=eq["High"],
            low=eq["Low"], close=eq["Close"],
            increasing_line_color="#00e676",
            decreasing_line_color="#ff1744",
            increasing_fillcolor="#00e676",
            decreasing_fillcolor="#ff1744",
            name=ticker,
        ), row=1, col=1)

        # Alpha prediction horizontal line overlay
        last_close = float(eq["Close"].dropna().iloc[-1])
        alpha_price_target = last_close * (1 + result.alpha * 0.03)   # illustrative projection
        fig.add_hline(
            y=alpha_price_target,
            line=dict(color="#00b4d8", width=1.5, dash="dot"),
            annotation_text=f"α={result.alpha:+.4f}  {result.signal_label}",
            annotation_font=dict(color="#00b4d8", family="IBM Plex Mono", size=10),
            row=1, col=1,
        )

        if "Volume" in eq.columns:
            colors = ["#00e676" if c >= o else "#ff1744"
                      for c, o in zip(eq["Close"], eq["Open"])]
            fig.add_trace(go.Bar(
                x=eq.index, y=eq["Volume"].squeeze(),
                marker_color=colors, opacity=0.55, name="Volume",
            ), row=2, col=1)

        if "Volatility_10d" in bundle.ohlcv.features.columns:
            vol_s = bundle.ohlcv.features["Volatility_10d"]
            fig.add_trace(go.Scatter(
                x=vol_s.index, y=vol_s,
                line=dict(color="#ffc400", width=1.3),
                fill="tozeroy", fillcolor="rgba(255,196,0,0.08)",
                name="Vol 10d",
            ), row=3, col=1)
            fig.add_hline(
                y=CRISIS_VOL_THRESHOLD,
                line=dict(color="#ff1744", width=1, dash="dot"),
                annotation_text="Crisis threshold",
                annotation_font=dict(color="#ff1744", size=9, family="IBM Plex Mono"),
                row=3, col=1,
            )

        fig.update_layout(
            **PTHEME,
            height=480,
            showlegend=False,
            xaxis_rangeslider_visible=False,
        )
        st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})

        # ── Crisis Mode banner ───────────────────────────────────────────
        if result.crisis_mode:
            st.markdown(
                f'<div class="crisis-banner">⚠ CRISIS MODE ACTIVE — '
                f'Realised Vol {result.current_vol:.1%} > {CRISIS_VOL_THRESHOLD:.0%} threshold — '
                f'Geopolitical text weight elevated to {result.crisis_weight:.0%}</div>',
                unsafe_allow_html=True,
            )

        # ── Performance metrics ──────────────────────────────────────────
        bt = st.session_state.backtest
        if bt:
            st.markdown('<div class="sec-label" style="margin-top:0.8rem;">📊 Backtest Metrics (Hybrid)</div>', unsafe_allow_html=True)
            m1, m2, m3 = st.columns(3)
            m1.metric("Sharpe Ratio",   f"{bt.sharpe_ratio:.3f}")
            m2.metric("Max Drawdown",   f"{bt.max_drawdown:.1%}")
            m3.metric("RMSE",           f"{bt.rmse:.5f}")
            m4, m5, m6 = st.columns(3)
            m4.metric("Annual Return",  f"{bt.annualised_return:.1%}")
            m5.metric("Total Return",   f"{bt.total_return:.1%}")
            m6.metric("Trades",         str(bt.n_trades))

            # Equity curve
            ec_fig = go.Figure(go.Scatter(
                x=list(range(len(bt.equity_curve))),
                y=bt.equity_curve.values,
                line=dict(color="#00b4d8", width=1.5),
                fill="tozeroy", fillcolor="rgba(0,180,216,0.07)",
            ))
            ec_fig.add_hline(y=1.0, line=dict(color="#546e7a", width=1, dash="dot"))
            ec_fig.update_layout(
                **PTHEME, height=180,
                title=dict(text="Equity Curve (Hybrid Strategy)", font=dict(size=11, color="#546e7a")),
            )
            st.plotly_chart(ec_fig, use_container_width=True, config={"displayModeBar": False})


# ══════════════════════════════════════════════════════════════════════════════
# COLUMN 2 – Geopolitical Sentiment Pulse Gauge
# ══════════════════════════════════════════════════════════════════════════════

with col2:
    st.markdown('<div class="sec-label">🌐 Geopolitical Sentiment Pulse</div>', unsafe_allow_html=True)

    if not st.session_state.loaded:
        st.info("Awaiting inference…")
    else:
        result = st.session_state.result
        bundle = st.session_state.bundle

        # ── Sentiment Pulse gauge ────────────────────────────────────────
        # Map Positive→+1, Neutral→0, Negative→-1 for gauge value
        sent_score = (
            result.sentiment_probs["Positive"] - result.sentiment_probs["Negative"]
        )   # in [-1, 1]
        gauge_val = (sent_score + 1) / 2 * 100   # map to [0, 100]

        SENT_COLOUR = {
            "Positive": "#00e676",
            "Neutral":  "#ffc400",
            "Negative": "#ff1744",
        }
        sent_col = SENT_COLOUR.get(result.sentiment, "#546e7a")

        gauge_fig = go.Figure(go.Indicator(
            mode="gauge+number+delta",
            value=gauge_val,
            number=dict(
                suffix=" / 100",
                font=dict(family="IBM Plex Mono", color=sent_col, size=26),
            ),
            title=dict(
                text=f"<b>{result.sentiment}</b>",
                font=dict(family="IBM Plex Mono", color=sent_col, size=14),
            ),
            delta=dict(reference=50, relative=False,
                       increasing=dict(color="#00e676"),
                       decreasing=dict(color="#ff1744")),
            gauge=dict(
                axis=dict(range=[0, 100], tickcolor="#162030",
                          tickfont=dict(family="IBM Plex Mono", size=9)),
                bar=dict(color=sent_col, thickness=0.22),
                bgcolor="#0a1220",
                borderwidth=0,
                steps=[
                    dict(range=[0,  30], color="#1a0a0a"),
                    dict(range=[30, 70], color="#0f1a0f"),
                    dict(range=[70,100], color="#0a1a0a"),
                ],
                threshold=dict(
                    line=dict(color="#eceff1", width=2),
                    thickness=0.85,
                    value=gauge_val,
                ),
            ),
        ))
        gauge_fig.update_layout(
            paper_bgcolor="rgba(0,0,0,0)",
            font=dict(family="IBM Plex Mono", color="#546e7a"),
            margin=dict(l=10, r=10, t=50, b=10),
            height=260,
        )
        st.plotly_chart(gauge_fig, use_container_width=True, config={"displayModeBar": False})

        # ── Sentiment breakdown bars ─────────────────────────────────────
        s_probs = result.sentiment_probs
        bar_fig = go.Figure(go.Bar(
            x=list(s_probs.keys()),
            y=list(s_probs.values()),
            marker_color=["#00e676", "#ff1744", "#ffc400"],
            text=[f"{v:.1%}" for v in s_probs.values()],
            textposition="outside",
            textfont=dict(family="IBM Plex Mono", size=10, color="#eceff1"),
        ))
        bar_fig.update_layout(
            **{**PTHEME, "yaxis": {**PTHEME["yaxis"], "range": [0, 1]}},
            height=160,
            showlegend=False,
            title=dict(text="FinBERT-Tone Probabilities", font=dict(size=10, color="#546e7a")),
        )
        st.plotly_chart(bar_fig, use_container_width=True, config={"displayModeBar": False})

        st.divider()

        # ── Alpha signal block ───────────────────────────────────────────
        cls = _alpha_class(result.alpha)
        st.markdown(
            f'<div style="text-align:center;padding:0.4rem 0;">'
            f'<div class="sec-label">ALPHA SIGNAL</div>'
            f'<div class="alpha-big {cls}">{result.alpha:+.4f}</div>'
            f'<div style="font-family:IBM Plex Mono;font-size:0.82rem;color:#546e7a;margin-top:4px;">'
            f'{result.signal_label}</div></div>',
            unsafe_allow_html=True,
        )

        # ── Regime probabilities ─────────────────────────────────────────
        st.markdown('<div class="sec-label" style="margin-top:0.6rem;">Market Regime</div>', unsafe_allow_html=True)
        reg_p = result.regime_probs
        reg_fig = go.Figure(go.Bar(
            x=list(reg_p.keys()), y=list(reg_p.values()),
            marker_color=["#00e676", "#ff1744", "#546e7a"],
            text=[f"{v:.0%}" for v in reg_p.values()],
            textposition="outside",
            textfont=dict(family="IBM Plex Mono", size=10, color="#eceff1"),
        ))
        reg_fig.update_layout(
            **{**PTHEME, "yaxis": {**PTHEME["yaxis"], "range": [0, 1.15]}},
            height=150,
            showlegend=False,
        )
        st.plotly_chart(reg_fig, use_container_width=True, config={"displayModeBar": False})

        # ── Headline volume by source ────────────────────────────────────
        st.markdown('<div class="sec-label">Daily Headline Count by Source</div>', unsafe_allow_html=True)
        nd = bundle.news_daily
        if len(nd):
            hvol_fig = go.Figure()
            src_cols = [("reuters_count", "#ff8f00", "Reuters"),
                        ("ft_count",      "#f9a825", "FT"),
                        ("aljazeera_count","#00b4d8","Al Jazeera")]
            for col, colour, label in src_cols:
                if col in nd.columns:
                    hvol_fig.add_trace(go.Bar(
                        x=nd.index, y=nd[col],
                        name=label, marker_color=colour, opacity=0.8,
                    ))
            hvol_fig.update_layout(
                **PTHEME, height=160, barmode="stack", showlegend=True,
                legend=dict(orientation="h", y=1.12,
                            font=dict(family="IBM Plex Mono", size=9, color="#546e7a")),
            )
            st.plotly_chart(hvol_fig, use_container_width=True, config={"displayModeBar": False})

        # ── Ablation Study ───────────────────────────────────────────────
        abl = st.session_state.ablation
        if abl:
            st.markdown('<div class="sec-label" style="margin-top:0.6rem;">🧪 Ablation Study</div>', unsafe_allow_html=True)
            h, p = abl.hybrid, abl.price_only

            def _better(h_val: float, p_val: float, higher_is_better: bool = True) -> tuple[str, str]:
                if higher_is_better:
                    hc = "better" if h_val >= p_val else "worse"
                    pc = "better" if p_val > h_val else "worse"
                else:
                    hc = "better" if h_val <= p_val else "worse"
                    pc = "better" if p_val < h_val else "worse"
                return hc, pc

            rows = [
                ("Sharpe Ratio",  f"{h.sharpe_ratio:.3f}",       f"{p.sharpe_ratio:.3f}",      *_better(h.sharpe_ratio, p.sharpe_ratio)),
                ("Max Drawdown",  f"{h.max_drawdown:.1%}",        f"{p.max_drawdown:.1%}",      *_better(h.max_drawdown, p.max_drawdown, higher_is_better=False)),
                ("RMSE",          f"{h.rmse:.5f}",                f"{p.rmse:.5f}",              *_better(h.rmse, p.rmse, higher_is_better=False)),
                ("Annual Return", f"{h.annualised_return:.1%}",   f"{p.annualised_return:.1%}", *_better(h.annualised_return, p.annualised_return)),
            ]

            table_html = """
<table class="ablation-table">
<tr>
  <th>Metric</th>
  <th>Hybrid</th>
  <th>Price-Only</th>
</tr>
"""
            for metric, hv, pv, hc, pc in rows:
                table_html += f"""<tr>
  <td>{metric}</td>
  <td class="{hc}">{hv}</td>
  <td class="{pc}">{pv}</td>
</tr>
"""
            table_html += "</table>"
            st.markdown(table_html, unsafe_allow_html=True)

            imp_s = abl.sharpe_improvement
            imp_r = abl.rmse_improvement
            s_col = "#00e676" if imp_s > 0 else "#ff1744"
            r_col = "#00e676" if imp_r > 0 else "#ff1744"
            st.markdown(
                f'<div style="font-family:IBM Plex Mono;font-size:0.68rem;color:#37474f;margin-top:6px;">'
                f'Sharpe Δ <span style="color:{s_col}">{imp_s:+.3f}</span> &nbsp;|&nbsp;'
                f'RMSE Δ <span style="color:{r_col}">{imp_r:+.5f}</span>'
                f'</div>',
                unsafe_allow_html=True,
            )


# ══════════════════════════════════════════════════════════════════════════════
# COLUMN 3 – Intelligence Feed + Macro
# ══════════════════════════════════════════════════════════════════════════════

with col3:
    st.markdown('<div class="sec-label">📡 Intelligence Feed — Reuters · FT · Al Jazeera</div>', unsafe_allow_html=True)

    if not st.session_state.loaded:
        st.info("Awaiting inference…")
    else:
        bundle  = st.session_state.bundle
        articles: list[NewsArticle] = bundle.articles

        if not articles:
            st.warning(
                "No articles loaded. Ensure `NEWSAPI_KEY` is set in "
                "`st.secrets` (Streamlit Cloud) or in your environment."
            )
        else:
            # ── Source filter ────────────────────────────────────────────
            selected_sources = st.multiselect(
                "Filter by source",
                options=list(DOMAIN_DISPLAY.values()),
                default=list(DOMAIN_DISPLAY.values()),
            )
            label_to_domain = {v: k for k, v in DOMAIN_DISPLAY.items()}
            selected_domains = [label_to_domain[s] for s in selected_sources]

            filtered = [a for a in articles if a.source_domain in selected_domains]
            filtered.sort(key=lambda a: a.published_at, reverse=True)

            st.caption(f"Showing {len(filtered)} of {len(articles)} articles")

            for art in filtered[:20]:
                fc = _feed_class(art.source_domain)
                ts = art.published_at.strftime("%b %d %H:%M UTC")
                title_snip = art.title[:140] if art.title else "(No title)"
                link_html = (
                    f'<a href="{art.url}" target="_blank" '
                    f'style="color:#90a4ae;text-decoration:none;">{title_snip}</a>'
                    if art.url else title_snip
                )
                st.markdown(
                    f'<div class="feed-item {fc}">'
                    f'<div class="feed-source">{art.source_label} &nbsp;·&nbsp; {ts}</div>'
                    f'{link_html}'
                    f'</div>',
                    unsafe_allow_html=True,
                )

        st.divider()

        # ── Macro panel ───────────────────────────────────────────────────
        st.markdown('<div class="sec-label">📉 Macro Indicators (FRED)</div>', unsafe_allow_html=True)
        macro = bundle.macro

        if not macro.empty and macro.values.any():
            macro_labels = {
                "FEDFUNDS":   "Fed Funds Rate",
                "T10Y2Y":     "10Y–2Y Spread",
                "DCOILWTICO": "WTI Crude $/bbl",
            }
            mac_fig = go.Figure()
            colours = ["#00b4d8", "#b39ddb", "#69f0ae"]
            for i, col in enumerate(macro.columns):
                if col in macro.columns and macro[col].any():
                    mac_fig.add_trace(go.Scatter(
                        x=macro.index, y=macro[col],
                        name=macro_labels.get(col, col),
                        line=dict(color=colours[i % len(colours)], width=1.3),
                    ))
            mac_fig.update_layout(
                **PTHEME, height=200,
                legend=dict(
                    orientation="h", y=1.1,
                    font=dict(family="IBM Plex Mono", size=9, color="#546e7a"),
                ),
            )
            st.plotly_chart(mac_fig, use_container_width=True, config={"displayModeBar": False})

            # Latest macro values
            mc1, mc2, mc3 = st.columns(3)
            for mcol, mcell in zip(macro.columns, [mc1, mc2, mc3]):
                latest = macro[mcol].dropna()
                val    = f"{float(latest.iloc[-1]):.2f}" if len(latest) else "N/A"
                mcell.metric(macro_labels.get(mcol, mcol), val)
        else:
            st.caption(
                "Macro data unavailable — set `FRED_API_KEY` in secrets "
                "or install `pandas_datareader`."
            )

        st.divider()

        # ── Model + data metadata ────────────────────────────────────────
        st.markdown('<div class="sec-label">ℹ System Info</div>', unsafe_allow_html=True)
        if st.session_state.result:
            r = st.session_state.result
            b = st.session_state.bundle
            st.markdown(f"""
<div style="font-family:'IBM Plex Mono',monospace;font-size:0.68rem;color:#37474f;line-height:1.9;">
TICKER &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;{b.ticker}<br>
PERIOD &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;{b.start} → {b.end}<br>
TRADING DAYS {len(b.aligned_index)}<br>
ARTICLES &nbsp;&nbsp;&nbsp;&nbsp;{len(b.articles)}<br>
SEQ LEN &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;{LOOKBACK_DAYS} days<br>
FINBERT &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;yiyanghkust/finbert-tone<br>
CRISIS VOL &nbsp;&nbsp;{CRISIS_VOL_THRESHOLD:.0%}<br>
DOMAINS &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;{" · ".join(APPROVED_DOMAINS)}<br>
</div>
""", unsafe_allow_html=True)
