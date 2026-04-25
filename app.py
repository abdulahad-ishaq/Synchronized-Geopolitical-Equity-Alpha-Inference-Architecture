"""
app.py
======
Synchronized Geopolitical-Equity Alpha Inference Architecture
-------------------------------------------------------------
Main Streamlit dashboard connecting all modules:

  config.py        - Central configuration
  data_loader.py   - GNews, RSS, PSX, DSA pipeline
  model_utils.py   - BiLSTM + Sentiment + Fusion + Training
  portfolio.py     - Multi-ticker, watchlist, paper trading
  alerts.py        - Email alerts, signal detection, crisis monitor
  reports.py       - PDF report generation
  auth.py          - User authentication, free/premium tiers
  backtest_viz.py  - Equity curves, drawdown, ablation charts

Layout: 3-column dashboard
  Col 1: Candlestick + Alpha overlay + Backtest
  Col 2: Sentiment Pulse + Regime + Signals
  Col 3: Intelligence Feed + Macro + Portfolio
"""

from __future__ import annotations

import warnings
warnings.filterwarnings("ignore")

import logging
from datetime import datetime
from typing import Optional

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit as st

# ── Local modules ──────────────────────────────────────────────────────────
from config import Config
from data_loader import (
    SynchronizedDataLoader,
    AlignedBundle,
    NewsArticle,
    MultiTickerBundle,
)
from model_utils import (
    AlphaInferenceModel,
    PriceOnlyBaseline,
    BacktestEngine,
    InferenceResult,
    AblationResult,
    build_model,
    run_inference,
    prepare_training_data,
)
from portfolio import (
    MultiTickerAnalyser,
    WatchlistManager,
    PaperTradingEngine,
    PortfolioAnalytics,
    PSXPortfolioHelper,
    TickerSnapshot,
)
from alerts import AlertManager
from reports import render_pdf_download_button, build_report_data
from auth import AuthManager
from backtest_viz import BacktestVisualiser

logger = logging.getLogger(__name__)

# =============================================================================
# Page configuration
# =============================================================================

st.set_page_config(
    page_title=Config.ui.APP_TITLE,
    page_icon=Config.ui.APP_ICON,
    layout="wide",
    initial_sidebar_state="expanded",
)

# =============================================================================
# CSS
# =============================================================================

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@400;600;700&family=IBM+Plex+Sans:wght@300;400;500;600&display=swap');

html, body, [class*="css"] {
    font-family: 'IBM Plex Sans', sans-serif;
    background-color: #080c14;
    color: #b0bec5;
}
h1,h2,h3,h4 { font-family: 'IBM Plex Mono', monospace; }

.sgeaia-header {
    background: linear-gradient(90deg,#0a1628 0%,#091220 60%,#0a1628 100%);
    border-bottom: 1px solid #1a3050;
    padding: 0.6rem 1.2rem;
    display: flex; align-items: center; gap: 1rem;
    margin-bottom: 0.8rem;
}
.sgeaia-title {
    font-family: 'IBM Plex Mono', monospace;
    font-size: 1.0rem; font-weight: 700;
    color: #00b4d8; letter-spacing: 0.06em; text-transform: uppercase;
}
.sgeaia-sub {
    font-size: 0.68rem; color: #546e7a;
    font-family: 'IBM Plex Mono', monospace; letter-spacing: 0.04em;
}
.ts-badge {
    margin-left: auto; font-family: 'IBM Plex Mono', monospace;
    font-size: 0.65rem; color: #37474f;
}
.sec-label {
    font-family: 'IBM Plex Mono', monospace;
    font-size: 0.60rem; color: #37474f;
    text-transform: uppercase; letter-spacing: 0.14em;
    margin-bottom: 0.3rem;
}
.alpha-big {
    font-family: 'IBM Plex Mono', monospace;
    font-size: 2.8rem; font-weight: 700; line-height: 1;
}
.crisis-banner {
    background: linear-gradient(90deg,#3d0000,#1a0000);
    border: 1px solid #ff1744; border-radius: 5px;
    padding: 0.4rem 0.8rem; text-align: center;
    font-family: 'IBM Plex Mono', monospace;
    font-size: 0.76rem; color: #ff6d6d;
    letter-spacing: 0.08em; margin-bottom: 0.5rem;
}
.feed-item {
    border-left: 3px solid #00b4d8;
    background: #0a1220; border-radius: 0 4px 4px 0;
    padding: 0.4rem 0.65rem; margin-bottom: 0.3rem;
    font-size: 0.75rem; line-height: 1.4; color: #90a4ae;
}
.feed-item.reuters      { border-left-color: #ff8f00; }
.feed-item.ft           { border-left-color: #f9a825; }
.feed-item.aljazeera    { border-left-color: #00b4d8; }
.feed-item.nikkei       { border-left-color: #e91e63; }
.feed-item.wsj          { border-left-color: #7c4dff; }
.feed-item.bloomberg    { border-left-color: #00e676; }
.feed-item.dawn         { border-left-color: #009688; }
.feed-item.brecorder    { border-left-color: #4caf50; }
.feed-item.thenews      { border-left-color: #ff5722; }
.feed-item.tribune      { border-left-color: #2196f3; }
.feed-item.geo          { border-left-color: #9c27b0; }
.feed-source {
    font-family: 'IBM Plex Mono', monospace;
    font-size: 0.57rem; color: #546e7a;
    text-transform: uppercase; letter-spacing: 0.1em; margin-bottom: 2px;
}
.impact-badge {
    display: inline-block; font-size: 0.55rem;
    font-family: 'IBM Plex Mono', monospace;
    padding: 1px 5px; border-radius: 3px;
    background: #162030; color: #00b4d8;
    margin-left: 4px;
}
.psx-badge {
    display: inline-block; font-size: 0.58rem;
    font-family: 'IBM Plex Mono', monospace;
    padding: 2px 6px; border-radius: 3px;
    background: #0a1a0a; color: #4caf50;
    border: 1px solid #1a3a1a;
}
section[data-testid="stSidebar"] {
    background: #080c14; border-right: 1px solid #162030;
}
[data-testid="metric-container"] {
    background: #0d1521; border: 1px solid #162030; border-radius: 5px;
}
div[data-testid="stHorizontalBlock"] > div { padding: 0 5px; }
</style>
""", unsafe_allow_html=True)

# =============================================================================
# Plotly theme
# =============================================================================

PTHEME = dict(
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="#0a1220",
    font=dict(family="IBM Plex Mono, monospace", color="#546e7a", size=10),
    margin=dict(l=8, r=8, t=28, b=8),
    xaxis=dict(gridcolor="#111d2e", zerolinecolor="#111d2e"),
    yaxis=dict(gridcolor="#111d2e", zerolinecolor="#111d2e"),
)

# =============================================================================
# Session state initialisation
# =============================================================================

for key, default in [
    ("bundle",       None),
    ("multi_bundle", None),
    ("result",       None),
    ("backtest",     None),
    ("ablation",     None),
    ("loaded",       False),
    ("active_tab",   "Dashboard"),
]:
    if key not in st.session_state:
        st.session_state[key] = default

# =============================================================================
# Auth
# =============================================================================

auth    = AuthManager()
if not auth.is_logged_in:
    auth.render_login_page()
    st.stop()

user = auth.current_user
tier = auth.tier

# =============================================================================
# Initialise managers
# =============================================================================

alert_mgr   = AlertManager()
watchlist   = WatchlistManager()
paper_eng   = PaperTradingEngine(
    currency="PKR" if (user and user.country == "PK") else "USD"
)
viz         = BacktestVisualiser()
psx_helper  = PSXPortfolioHelper()

# =============================================================================
# Sidebar
# =============================================================================

with st.sidebar:
    auth.render_user_menu()
    st.divider()

    st.markdown("### ⚙ Configuration")

    # Market selector
    market_options = list(Config.global_markets.EXCHANGES.keys())
    selected_market = st.selectbox(
        "Market",
        options=market_options,
        index=0,
        format_func=lambda m: (
            f"{Config.global_markets.EXCHANGES[m]['flag']} "
            f"{Config.global_markets.EXCHANGES[m]['name']}"
        ),
    )

    # Ticker input
    if selected_market == "PSX":
        ticker_groups = {
            k: v for k, v in Config.market.TICKER_GROUPS.items()
            if "🇵🇰" in k
        }
    else:
        ticker_groups = {
            k: v for k, v in Config.market.TICKER_GROUPS.items()
            if "🇵🇰" not in k
        }

    selected_group = st.selectbox(
        "Ticker Group",
        options=["Custom"] + list(ticker_groups.keys()),
    )

    if selected_group == "Custom":
        ticker_input = st.text_input(
            "Ticker Symbol",
            value="ENGRO.KA" if selected_market == "PSX" else "SPY",
        ).upper().strip()
    else:
        default_tickers = ticker_groups.get(selected_group, ["SPY"])
        ticker_input = st.selectbox(
            "Select Ticker",
            options=default_tickers,
        )

    # Multi-ticker (premium)
    if tier.can_access_multi_ticker():
        compare_tickers_str = st.text_input(
            "Compare Tickers (comma-separated, max 5)",
            value="",
            placeholder="e.g. ENGRO.KA, HBL.KA, SYS.KA",
        )
        compare_tickers = [
            t.strip().upper()
            for t in compare_tickers_str.split(",")
            if t.strip()
        ][:Config.market.MAX_COMPARISON_TICKERS]
    else:
        compare_tickers = []
        tier.render_upgrade_prompt("Multi-ticker comparison")

    lookback = st.slider(
        "Lookback (days)", 90, 365,
        value=Config.ui.DEFAULT_LOOKBACK_DAYS, step=15,
    )

    st.divider()

    # Watchlist
    with st.expander("📌 Watchlist", expanded=False):
        wl_tickers = watchlist.tickers
        if wl_tickers:
            for wl_ticker in wl_tickers:
                c1, c2 = st.columns([3, 1])
                c1.caption(wl_ticker)
                if c2.button("×", key=f"rm_{wl_ticker}"):
                    watchlist.remove(wl_ticker)
                    st.rerun()
        add_col1, add_col2 = st.columns([3, 1])
        new_ticker = add_col1.text_input(
            "Add ticker", placeholder="e.g. OGDC.KA", label_visibility="collapsed"
        )
        if add_col2.button("Add"):
            if new_ticker:
                if watchlist.add(new_ticker.upper()):
                    st.success(f"Added {new_ticker.upper()}")
                    st.rerun()
                else:
                    st.warning("Already in watchlist")

    st.divider()

    # Options
    run_backtest_cb = st.checkbox(
        "Run Backtesting & Ablation",
        value=False,
        help="Iterates all sequence windows. Takes 1-3 min.",
    )

    show_portfolio_cb = st.checkbox("Show Portfolio View", value=False)
    show_paper_cb     = st.checkbox(
        "Show Paper Trading",
        value=False,
        disabled=not tier.can_paper_trade(),
    )

    st.divider()
    run_btn = st.button(
        "🛰 Run Inference", use_container_width=True, type="primary"
    )

    st.caption(
        "API keys are read from `st.secrets` (Streamlit Cloud) "
        "or environment variables (local)."
    )

# =============================================================================
# Header
# =============================================================================

psx_open_str = "🟢 PSX OPEN" if psx_helper.is_psx_market_open() else "🔴 PSX CLOSED"
st.markdown(f"""
<div class="sgeaia-header">
  <div>
    <div class="sgeaia-title">
      {Config.ui.APP_ICON} {Config.ui.APP_TITLE}
    </div>
    <div class="sgeaia-sub">{Config.ui.APP_SUBTITLE}</div>
  </div>
  <div class="ts-badge">
    {psx_open_str} &nbsp;|&nbsp;
    {datetime.utcnow().strftime('%Y-%m-%d %H:%M UTC')}
  </div>
</div>
""", unsafe_allow_html=True)

# Alert market status bar
alert_mgr.render_market_status_bar()

# =============================================================================
# Data & model loading helpers
# =============================================================================

@st.cache_data(ttl=300, show_spinner="Synchronizing market & geopolitical data...")
def load_bundle(ticker: str, lookback: int) -> AlignedBundle:
    loader = SynchronizedDataLoader(ticker=ticker, seq_len=Config.model.SEQ_LEN)
    return loader.load(lookback_days=lookback)


@st.cache_data(ttl=300, show_spinner="Loading multi-ticker data...")
def load_multi_bundle(tickers: list[str], lookback: int) -> MultiTickerBundle:
    loader = SynchronizedDataLoader(
        ticker=tickers[0],
        tickers=tickers,
        seq_len=Config.model.SEQ_LEN,
    )
    return loader.load_multi(lookback_days=lookback)

# =============================================================================
# Run inference
# =============================================================================

if run_btn:
    with st.spinner("Loading data..."):
        try:
            bundle = load_bundle(ticker_input, lookback)
            st.session_state.bundle = bundle
        except Exception as exc:
            st.error(f"Data loading failed: {exc}")
            st.stop()

    with st.spinner("Building models..."):
        model, baseline = build_model(
            num_features=bundle.ohlcv.num_features
        )

    with st.spinner("Running inference..."):
        try:
            loader     = SynchronizedDataLoader(
                ticker=ticker_input, seq_len=Config.model.SEQ_LEN
            )
            latest_seq = loader.latest_window(bundle)
            latest_txt = (
                bundle.news_daily["headline_concat"].iloc[-1]
                if "headline_concat" in bundle.news_daily.columns
                else ""
            ) or ""
            latest_txt = latest_txt[:512]

            vol_series  = bundle.ohlcv.features.get(
                "Volatility_10d", pd.Series([0.0])
            )
            current_vol = float(
                vol_series.dropna().iloc[-1]
            ) if len(vol_series.dropna()) else 0.0

            result = run_inference(
                model, latest_seq, latest_txt, current_vol=current_vol
            )
            st.session_state.result  = result
            st.session_state.loaded  = True

            # Process alerts
            top_headline = ""
            if bundle.ranked_articles:
                top_headline = bundle.ranked_articles[0].title or ""
            alert_mgr.process_new_result(
                ticker_input, result, top_headline,
                market=bundle.market,
            )

        except Exception as exc:
            st.error(f"Inference failed: {exc}")
            st.stop()

    # Multi-ticker comparison
    if compare_tickers and tier.can_access_multi_ticker():
        with st.spinner(f"Loading {len(compare_tickers)} tickers..."):
            try:
                all_tickers = [ticker_input] + [
                    t for t in compare_tickers if t != ticker_input
                ]
                multi = load_multi_bundle(all_tickers, lookback)
                st.session_state.multi_bundle = multi
            except Exception as exc:
                st.warning(f"Multi-ticker load failed (non-fatal): {exc}")

    # Backtest
    if run_backtest_cb:
        with st.spinner("Running backtesting & ablation study..."):
            try:
                X, y, dates = loader.build_sequences(bundle)
                n = len(y)
                texts = (
                    bundle.news_daily["headline_concat"]
                    .iloc[len(bundle.news_daily) - n:]
                    .fillna("")
                    .tolist()
                )
                vols = (
                    bundle.ohlcv.features["Volatility_10d"]
                    .iloc[len(bundle.ohlcv.features) - n:]
                    .fillna(0.0)
                    .values.astype(np.float32)
                )
                engine   = BacktestEngine(model, baseline)
                ablation = engine.ablation_study(
                    X, y, texts, vols, market=bundle.market
                )
                st.session_state.backtest = ablation.hybrid
                st.session_state.ablation = ablation
            except Exception as exc:
                st.warning(f"Backtesting failed (non-fatal): {exc}")

# =============================================================================
# 3-column main layout
# =============================================================================

col1, col2, col3 = st.columns([1.7, 1.0, 1.3], gap="medium")

# =============================================================================
# COLUMN 1 — Price Chart + Alpha Overlay + Backtest
# =============================================================================

with col1:
    st.markdown(
        f'<div class="sec-label">📈 {ticker_input} — Price & Alpha Overlay</div>',
        unsafe_allow_html=True,
    )

    if not st.session_state.loaded:
        st.info("Configure the sidebar and press **Run Inference** to begin.")
    else:
        bundle = st.session_state.bundle
        result = st.session_state.result
        eq     = bundle.ohlcv.raw
        sym    = Config.global_markets.EXCHANGES.get(
            bundle.market, {}
        ).get("symbol", "$")

        # ── Candlestick + Volume + Volatility ─────────────────────────
        fig = make_subplots(
            rows=3, cols=1,
            shared_xaxes=True,
            row_heights=[0.55, 0.25, 0.20],
            vertical_spacing=0.04,
        )

        fig.add_trace(go.Candlestick(
            x=eq.index,
            open=eq["Open"].squeeze(),
            high=eq["High"].squeeze(),
            low=eq["Low"].squeeze(),
            close=eq["Close"].squeeze(),
            increasing_line_color=Config.ui.BULL_COLOUR,
            decreasing_line_color=Config.ui.BEAR_COLOUR,
            increasing_fillcolor=Config.ui.BULL_COLOUR,
            decreasing_fillcolor=Config.ui.BEAR_COLOUR,
            name=ticker_input,
        ), row=1, col=1)

        # Alpha projection line
        last_close = bundle.ohlcv.latest_price
        alpha_target = last_close * (1 + result.alpha * 0.03)
        fig.add_hline(
            y=alpha_target,
            line=dict(color=Config.ui.PRIMARY_COLOUR, width=1.5, dash="dot"),
            annotation_text=(
                f"α={result.alpha:+.4f}  "
                f"{result.signal_arrow} {result.signal_label}"
            ),
            annotation_font=dict(
                color=Config.ui.PRIMARY_COLOUR,
                family="IBM Plex Mono", size=10,
            ),
            row=1, col=1,
        )

        # Volume
        if "Volume" in eq.columns:
            close_s = eq["Close"].squeeze()
            open_s  = eq["Open"].squeeze()
            vol_colours = [
                Config.ui.BULL_COLOUR if c >= o else Config.ui.BEAR_COLOUR
                for c, o in zip(close_s, open_s)
            ]
            fig.add_trace(go.Bar(
                x=eq.index,
                y=eq["Volume"].squeeze(),
                marker_color=vol_colours,
                opacity=0.55,
                name="Volume",
            ), row=2, col=1)

        # Volatility
        if "Volatility_10d" in bundle.ohlcv.features.columns:
            vol_s = bundle.ohlcv.features["Volatility_10d"]
            fig.add_trace(go.Scatter(
                x=vol_s.index, y=vol_s.values,
                line=dict(color=Config.ui.WARNING_COLOUR, width=1.3),
                fill="tozeroy",
                fillcolor="rgba(255,196,0,0.06)",
                name="Vol 10d",
            ), row=3, col=1)
            fig.add_hline(
                y=Config.model.CRISIS_VOL_THRESHOLD,
                line=dict(
                    color=Config.ui.CRISIS_COLOUR,
                    width=1, dash="dot",
                ),
                annotation_text="Crisis threshold",
                annotation_font=dict(
                    color=Config.ui.CRISIS_COLOUR,
                    size=8, family="IBM Plex Mono",
                ),
                row=3, col=1,
            )

        fig.update_layout(
            **PTHEME,
            height=460,
            showlegend=False,
            xaxis_rangeslider_visible=False,
        )
        st.plotly_chart(
            fig, use_container_width=True,
            config={"displayModeBar": False},
        )

        # Crisis Mode banner
        if result.crisis_mode:
            st.markdown(
                f'<div class="crisis-banner">'
                f'⚠ CRISIS MODE ACTIVE — '
                f'Vol {result.current_vol:.1%} > '
                f'{Config.model.CRISIS_VOL_THRESHOLD:.0%} threshold — '
                f'Geo text weight: {result.crisis_weight:.0%}'
                f'</div>',
                unsafe_allow_html=True,
            )

        # PSX market status
        if bundle.market == "PSX":
            is_open  = psx_helper.is_psx_market_open()
            next_ses = psx_helper.next_trading_session()
            colour   = Config.ui.BULL_COLOUR if is_open else Config.ui.BEAR_COLOUR
            st.markdown(
                f'<div style="font-family:IBM Plex Mono;font-size:0.68rem;'
                f'color:{colour};">● PSX '
                f'{"OPEN" if is_open else "CLOSED"}'
                f'{" · " + next_ses if not is_open else ""}'
                f'</div>',
                unsafe_allow_html=True,
            )

        # ── Backtest section ─────────────────────────────────────────
        if st.session_state.backtest and st.session_state.ablation:
            with st.expander("📊 Backtesting & Ablation Study", expanded=False):
                viz.render(
                    backtest=st.session_state.backtest,
                    ablation=st.session_state.ablation,
                    ticker=ticker_input,
                    market=bundle.market if bundle else "US",
                )

        # ── Multi-ticker comparison ───────────────────────────────────
        if st.session_state.multi_bundle:
            with st.expander(
                f"📊 Multi-Ticker Comparison "
                f"({len(st.session_state.multi_bundle.tickers)} tickers)",
                expanded=False,
            ):
                analyser = MultiTickerAnalyser(
                    st.session_state.multi_bundle.tickers
                )
                tab_norm, tab_corr, tab_vol = st.tabs([
                    "Normalised Returns",
                    "Correlation Matrix",
                    "Volatility",
                ])
                with tab_norm:
                    st.plotly_chart(
                        analyser.build_comparison_chart(
                            st.session_state.multi_bundle, normalise=True
                        ),
                        use_container_width=True,
                        config={"displayModeBar": False},
                    )
                with tab_corr:
                    st.plotly_chart(
                        analyser.build_correlation_heatmap(
                            st.session_state.multi_bundle
                        ),
                        use_container_width=True,
                        config={"displayModeBar": False},
                    )
                with tab_vol:
                    st.plotly_chart(
                        analyser.build_volatility_comparison(
                            st.session_state.multi_bundle
                        ),
                        use_container_width=True,
                        config={"displayModeBar": False},
                    )

        # ── PDF Report ────────────────────────────────────────────────
        if tier.can_generate_pdf() and st.session_state.loaded:
            st.divider()
            if st.button(
                "📄 Generate PDF Report",
                use_container_width=True,
            ):
                report_data = build_report_data(
                    bundle=st.session_state.bundle,
                    result=st.session_state.result,
                    backtest=st.session_state.backtest,
                    ablation=st.session_state.ablation,
                )
                render_pdf_download_button(report_data)
        elif not tier.can_generate_pdf():
            tier.render_upgrade_prompt("PDF Report Generation")

# =============================================================================
# COLUMN 2 — Sentiment Pulse + Regime + Alpha Signal
# =============================================================================

with col2:
    st.markdown(
        '<div class="sec-label">🌐 Geopolitical Sentiment Pulse</div>',
        unsafe_allow_html=True,
    )

    if not st.session_state.loaded:
        st.info("Awaiting inference...")
    else:
        result = st.session_state.result

        # ── Sentiment gauge ───────────────────────────────────────────
        sent_score = (
            result.sentiment_probs["Positive"] -
            result.sentiment_probs["Negative"]
        )
        gauge_val  = (sent_score + 1) / 2 * 100
        sent_col_map = {
            "Positive": Config.ui.BULL_COLOUR,
            "Negative": Config.ui.BEAR_COLOUR,
            "Neutral":  Config.ui.WARNING_COLOUR,
        }
        sent_col = sent_col_map.get(result.sentiment, Config.ui.NEUTRAL_COLOUR)

        gauge_fig = go.Figure(go.Indicator(
            mode="gauge+number+delta",
            value=gauge_val,
            number=dict(
                suffix=" / 100",
                font=dict(
                    family="IBM Plex Mono",
                    color=sent_col, size=24,
                ),
            ),
            title=dict(
                text=f"<b>{result.sentiment}</b>",
                font=dict(
                    family="IBM Plex Mono",
                    color=sent_col, size=13,
                ),
            ),
            delta=dict(
                reference=50, relative=False,
                increasing=dict(color=Config.ui.BULL_COLOUR),
                decreasing=dict(color=Config.ui.BEAR_COLOUR),
            ),
            gauge=dict(
                axis=dict(
                    range=[0, 100], tickcolor="#162030",
                    tickfont=dict(family="IBM Plex Mono", size=8),
                ),
                bar=dict(color=sent_col, thickness=0.22),
                bgcolor="#0a1220",
                borderwidth=0,
                steps=[
                    dict(range=[0,  30], color="#1a0a0a"),
                    dict(range=[30, 70], color="#0f180f"),
                    dict(range=[70,100], color="#0a1a0a"),
                ],
            ),
        ))
        gauge_fig.update_layout(
            paper_bgcolor="rgba(0,0,0,0)",
            font=dict(family="IBM Plex Mono", color="#546e7a"),
            margin=dict(l=10, r=10, t=50, b=10),
            height=240,
        )
        st.plotly_chart(
            gauge_fig, use_container_width=True,
            config={"displayModeBar": False},
        )

        # ── FinBERT-Tone probabilities ────────────────────────────────
        s_probs  = result.sentiment_probs
        bar_fig  = go.Figure(go.Bar(
            x=list(s_probs.keys()),
            y=list(s_probs.values()),
            marker_color=[
                Config.ui.BULL_COLOUR,
                Config.ui.BEAR_COLOUR,
                Config.ui.WARNING_COLOUR,
            ],
            text=[f"{v:.1%}" for v in s_probs.values()],
            textposition="outside",
            textfont=dict(
                family="IBM Plex Mono", size=10, color="#eceff1"
            ),
        ))
        bar_fig.update_layout(
            **{**PTHEME, "yaxis": {**PTHEME["yaxis"], "range": [0, 1]}},
            height=155,
            showlegend=False,
            title=dict(
                text="FinBERT-Tone Probabilities",
                font=dict(size=10, color="#546e7a"),
            ),
        )
        st.plotly_chart(
            bar_fig, use_container_width=True,
            config={"displayModeBar": False},
        )

        st.divider()

        # ── Alpha signal ──────────────────────────────────────────────
        alpha_col_map = {
            "LONG":  "alpha-long  style='color:" + Config.ui.BULL_COLOUR + ";'",
            "SHORT": "alpha-short style='color:" + Config.ui.BEAR_COLOUR + ";'",
            "FLAT":  "alpha-flat  style='color:" + Config.ui.NEUTRAL_COLOUR + ";'",
        }
        alpha_css_colour = {
            "LONG":  Config.ui.BULL_COLOUR,
            "SHORT": Config.ui.BEAR_COLOUR,
            "FLAT":  Config.ui.NEUTRAL_COLOUR,
        }.get(result.signal_label, Config.ui.NEUTRAL_COLOUR)

        st.markdown(
            f'<div style="text-align:center;padding:0.3rem 0;">'
            f'<div class="sec-label">ALPHA SIGNAL</div>'
            f'<div class="alpha-big" '
            f'style="color:{alpha_css_colour};">'
            f'{result.alpha:+.4f}</div>'
            f'<div style="font-family:IBM Plex Mono;font-size:0.8rem;'
            f'color:#546e7a;margin-top:3px;">'
            f'{result.signal_arrow} {result.signal_label}</div>'
            f'</div>',
            unsafe_allow_html=True,
        )

        # ── Regime probabilities ──────────────────────────────────────
        st.markdown(
            '<div class="sec-label" style="margin-top:0.5rem;">'
            'Market Regime</div>',
            unsafe_allow_html=True,
        )
        reg_p   = result.regime_probs
        reg_fig = go.Figure(go.Bar(
            x=list(reg_p.keys()),
            y=list(reg_p.values()),
            marker_color=[
                Config.ui.BULL_COLOUR,
                Config.ui.BEAR_COLOUR,
                Config.ui.NEUTRAL_COLOUR,
            ],
            text=[f"{v:.0%}" for v in reg_p.values()],
            textposition="outside",
            textfont=dict(
                family="IBM Plex Mono", size=10, color="#eceff1"
            ),
        ))
        reg_fig.update_layout(
            **{**PTHEME, "yaxis": {**PTHEME["yaxis"], "range": [0, 1.15]}},
            height=155,
            showlegend=False,
        )
        st.plotly_chart(
            reg_fig, use_container_width=True,
            config={"displayModeBar": False},
        )

        # ── Key metrics ───────────────────────────────────────────────
        m1, m2 = st.columns(2)
        m1.metric("Alpha",      f"{result.alpha:+.4f}")
        m2.metric("Confidence", f"{result.confidence:.1%}")
        m3, m4 = st.columns(2)
        m3.metric("Regime",     result.regime)
        m4.metric("Sentiment",  result.sentiment)

        # ── Alert panel ───────────────────────────────────────────────
        st.divider()
        if tier.can_set_email_alerts():
            alert_mgr.render_alert_panel()
        else:
            tier.render_upgrade_prompt("Email Alerts")

# =============================================================================
# COLUMN 3 — Intelligence Feed + Macro + Portfolio + Paper Trading
# =============================================================================

with col3:
    st.markdown(
        '<div class="sec-label">'
        '📡 Intelligence Feed — '
        + " · ".join(Config.news.DOMAIN_DISPLAY.values()) +
        '</div>',
        unsafe_allow_html=True,
    )

    if not st.session_state.loaded:
        st.info("Awaiting inference...")
    else:
        bundle  = st.session_state.bundle
        result  = st.session_state.result

        # Use ranked articles if available, else all articles
        articles: list[NewsArticle] = (
            bundle.ranked_articles
            if bundle.ranked_articles
            else bundle.articles
        )

        if not articles:
            st.warning(
                "No articles loaded. Configure `GNEWS_API_KEY` in "
                "Streamlit secrets to activate the feed."
            )
        else:
            # Source filter
            all_source_labels = list(
                set(a.source_label for a in articles)
            )
            selected_sources = st.multiselect(
                "Filter by source",
                options=all_source_labels,
                default=all_source_labels,
            )
            filtered = [
                a for a in articles
                if a.source_label in selected_sources
            ]

            # Pakistan / Global tabs
            pak_arts = [a for a in filtered if a.is_pakistan_source]
            gbl_arts = [a for a in filtered if not a.is_pakistan_source]

            tab_all, tab_pak, tab_gbl = st.tabs([
                f"All ({len(filtered)})",
                f"🇵🇰 PSX ({len(pak_arts)})",
                f"🌍 Global ({len(gbl_arts)})",
            ])

            def _render_feed(arts: list[NewsArticle], limit: int = 15) -> None:
                for art in arts[:limit]:
                    css  = art.source_css
                    ts   = art.published_at.strftime("%b %d %H:%M UTC")
                    title_snip = art.title[:140] if art.title else "(No title)"
                    impact_str = (
                        f'<span class="impact-badge">'
                        f'⚡{art.impact_score:.2f}</span>'
                        if art.impact_score > 0 else ""
                    )
                    pak_str = (
                        '<span class="psx-badge">PSX</span>'
                        if art.is_pakistan_source else ""
                    )
                    link_html = (
                        f'<a href="{art.url}" target="_blank" '
                        f'style="color:#90a4ae;text-decoration:none;">'
                        f'{title_snip}</a>'
                        if art.url else title_snip
                    )
                    st.markdown(
                        f'<div class="feed-item {css}">'
                        f'<div class="feed-source">'
                        f'{art.source_label} · {ts}'
                        f'{impact_str}{pak_str}'
                        f'</div>'
                        f'{link_html}'
                        f'</div>',
                        unsafe_allow_html=True,
                    )

            with tab_all:
                _render_feed(filtered)
            with tab_pak:
                if pak_arts:
                    _render_feed(pak_arts)
                else:
                    st.caption(
                        "No Pakistani news. Dawn and Business Recorder "
                        "RSS feeds will appear here."
                    )
            with tab_gbl:
                _render_feed(gbl_arts)

        st.divider()

        # ── Macro indicators ──────────────────────────────────────────
        st.markdown(
            '<div class="sec-label">📉 Macro Indicators (FRED)</div>',
            unsafe_allow_html=True,
        )
        macro = bundle.macro
        macro_labels = {
            "FEDFUNDS":   "Fed Funds Rate",
            "T10Y2Y":     "10Y-2Y Spread",
            "DCOILWTICO": "WTI Crude $/bbl",
        }
        if not macro.empty and macro.values.any():
            mac_fig = go.Figure()
            colours = [
                Config.ui.PRIMARY_COLOUR, "#b39ddb", "#69f0ae"
            ]
            for i, col in enumerate(macro.columns):
                if macro[col].any():
                    mac_fig.add_trace(go.Scatter(
                        x=macro.index, y=macro[col],
                        name=macro_labels.get(col, col),
                        line=dict(
                            color=colours[i % len(colours)], width=1.3
                        ),
                    ))
            mac_fig.update_layout(
                **PTHEME, height=190,
                legend=dict(
                    orientation="h", y=1.1,
                    font=dict(
                        family="IBM Plex Mono",
                        size=8, color="#546e7a",
                    ),
                ),
            )
            st.plotly_chart(
                mac_fig, use_container_width=True,
                config={"displayModeBar": False},
            )

            mc1, mc2, mc3 = st.columns(3)
            for mcol, mcell in zip(macro.columns, [mc1, mc2, mc3]):
                latest = macro[mcol].dropna()
                val    = (
                    f"{float(latest.iloc[-1]):.2f}"
                    if len(latest) else "N/A"
                )
                mcell.metric(macro_labels.get(mcol, mcol), val)
        else:
            st.caption(
                "Macro data unavailable. Set `FRED_API_KEY` in secrets."
            )

        # ── Portfolio view ────────────────────────────────────────────
        if show_portfolio_cb and st.session_state.multi_bundle:
            st.divider()
            st.markdown(
                '<div class="sec-label">💼 Portfolio View</div>',
                unsafe_allow_html=True,
            )
            multi = st.session_state.multi_bundle
            breakdown_fig = PortfolioAnalytics.build_market_breakdown_chart(
                multi.tickers
            )
            div_score = PortfolioAnalytics.diversification_score(
                multi.correlation_matrix
            )
            c_left, c_right = st.columns([1, 2])
            with c_left:
                st.plotly_chart(
                    breakdown_fig,
                    use_container_width=True,
                    config={"displayModeBar": False},
                )
                st.metric("Diversification Score", f"{div_score}/100")
            with c_right:
                st.caption("Correlation Matrix")
                st.plotly_chart(
                    MultiTickerAnalyser(
                        multi.tickers
                    ).build_correlation_heatmap(multi),
                    use_container_width=True,
                    config={"displayModeBar": False},
                )

        # ── Paper trading ─────────────────────────────────────────────
        if show_paper_cb and tier.can_paper_trade():
            st.divider()
            st.markdown(
                '<div class="sec-label">🎮 Paper Trading Simulator</div>',
                unsafe_allow_html=True,
            )
            port = paper_eng.portfolio

            p1, p2, p3 = st.columns(3)
            p1.metric(
                "Portfolio Value",
                f"{port.currency_symbol}{port.portfolio_value:,.0f}",
            )
            p2.metric("Total P&L",  f"{port.total_pnl:+,.2f}")
            p3.metric("Win Rate",   f"{port.win_rate:.0f}%")

            # Quick trade entry
            with st.expander("Open New Trade", expanded=False):
                t1, t2, t3 = st.columns(3)
                trade_ticker = t1.text_input(
                    "Ticker", value=ticker_input
                )
                direction = t2.selectbox(
                    "Direction", ["LONG", "SHORT"]
                )
                entry_price = t3.number_input(
                    "Entry Price",
                    value=bundle.ohlcv.latest_price,
                    min_value=0.01,
                )
                if st.button("Open Trade", use_container_width=True):
                    trade = paper_eng.open_trade(
                        ticker=trade_ticker,
                        direction=direction,
                        entry_price=entry_price,
                        currency=port.currency,
                    )
                    if trade:
                        st.success(
                            f"Opened {direction} #{trade.trade_id} "
                            f"@ {port.currency_symbol}{entry_price:,.2f}"
                        )
                        st.rerun()
                    else:
                        st.error(
                            "Trade failed. Check capital and position limits."
                        )

            # Equity curve
            if port.closed_trades:
                st.plotly_chart(
                    paper_eng.build_equity_curve(),
                    use_container_width=True,
                    config={"displayModeBar": False},
                )

            # Trade log
            if port.trades:
                st.plotly_chart(
                    paper_eng.build_trade_log_table(),
                    use_container_width=True,
                    config={"displayModeBar": False},
                )

            if st.button("Reset Portfolio", type="secondary"):
                paper_eng.reset_portfolio(currency=port.currency)
                st.rerun()

        # ── System info ───────────────────────────────────────────────
        st.divider()
        st.markdown(
            '<div class="sec-label">ℹ System Info</div>',
            unsafe_allow_html=True,
        )
        if st.session_state.loaded:
            b = st.session_state.bundle
            r = st.session_state.result
            st.markdown(
                f'<div style="font-family:IBM Plex Mono;'
                f'font-size:0.65rem;color:#37474f;line-height:1.9;">'
                f'TICKER &nbsp;&nbsp;&nbsp;{b.ticker}<br>'
                f'MARKET &nbsp;&nbsp;&nbsp;{b.market} ({b.currency})<br>'
                f'PERIOD &nbsp;&nbsp;&nbsp;{b.start} → {b.end}<br>'
                f'T-DAYS &nbsp;&nbsp;&nbsp;{len(b.aligned_index)}<br>'
                f'ARTICLES &nbsp;{len(b.articles)}<br>'
                f'RANKED &nbsp;&nbsp;{len(b.ranked_articles)}<br>'
                f'SEQ LEN &nbsp;&nbsp;{Config.model.SEQ_LEN} days<br>'
                f'CRISIS THR {Config.model.CRISIS_VOL_THRESHOLD:.0%}<br>'
                f'TIER &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;{user.tier_label if user else "N/A"}<br>'
                f'</div>',
                unsafe_allow_html=True,
            )
