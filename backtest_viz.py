"""
backtest_viz.py
===============
Synchronized Geopolitical-Equity Alpha Inference Architecture
-------------------------------------------------------------
Historical backtesting visualisation components:

  BacktestVisualiser    - Full backtesting dashboard charts
  EquityCurveChart      - Portfolio equity curve with benchmark
  DrawdownChart         - Underwater (drawdown) chart
  RollingMetricsChart   - Rolling Sharpe and volatility
  TradeDistribution     - Win/loss distribution and stats
  AblationComparer      - Side-by-side Hybrid vs Price-Only
  WalkForwardChart      - Walk-forward validation results
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Optional

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import streamlit as st

from config import Config
from model_utils import BacktestResult, AblationResult

logger = logging.getLogger(__name__)

# ── Shared Plotly theme ────────────────────────────────────────────────────

PTHEME = dict(
    paper_bgcolor = "rgba(0,0,0,0)",
    plot_bgcolor  = "#0a1220",
    font          = dict(
        family="IBM Plex Mono, monospace",
        color="#546e7a",
        size=10,
    ),
    margin = dict(l=8, r=8, t=36, b=8),
    xaxis  = dict(gridcolor="#111d2e", zerolinecolor="#111d2e", showgrid=True),
    yaxis  = dict(gridcolor="#111d2e", zerolinecolor="#111d2e", showgrid=True),
)


# =============================================================================
# 1. Equity Curve Chart
# =============================================================================

class EquityCurveChart:
    """
    Portfolio equity curve with buy-and-hold benchmark comparison.
    Shows how the strategy performed vs simply holding the asset.
    """

    @staticmethod
    def build(
        backtest:       BacktestResult,
        benchmark_returns: Optional[np.ndarray] = None,
        title:          str = "Strategy Equity Curve",
    ) -> go.Figure:
        """
        Build equity curve chart.

        Args:
            backtest:          BacktestResult from BacktestEngine
            benchmark_returns: Optional array of buy-and-hold returns
            title:             Chart title
        """
        fig = go.Figure()

        curve   = backtest.equity_curve
        x_range = list(range(len(curve)))

        # Strategy equity curve
        fig.add_trace(go.Scatter(
            x=x_range,
            y=curve.values,
            name="SGEAIA Strategy",
            line=dict(color=Config.ui.PRIMARY_COLOUR, width=2),
            fill="tozeroy",
            fillcolor="rgba(0,180,216,0.07)",
            hovertemplate=(
                "Step: %{x}<br>"
                "Value: %{y:.4f}<br>"
                "<extra>SGEAIA Strategy</extra>"
            ),
        ))

        # Benchmark (buy and hold)
        if benchmark_returns is not None and len(benchmark_returns) > 0:
            bh_curve = pd.Series(
                (1 + benchmark_returns).cumprod()
            )
            fig.add_trace(go.Scatter(
                x=x_range[:len(bh_curve)],
                y=bh_curve.values,
                name="Buy & Hold",
                line=dict(
                    color=Config.ui.NEUTRAL_COLOUR,
                    width=1.5,
                    dash="dot",
                ),
                hovertemplate=(
                    "Step: %{x}<br>"
                    "Value: %{y:.4f}<br>"
                    "<extra>Buy & Hold</extra>"
                ),
            ))

        # Initial capital line
        fig.add_hline(
            y=1.0,
            line=dict(color="#546e7a", width=1, dash="dot"),
            annotation_text="Initial Capital",
            annotation_font=dict(color="#546e7a", size=9),
        )

        # Annotate final return
        final_val = float(curve.iloc[-1])
        colour    = (
            Config.ui.BULL_COLOUR
            if final_val >= 1.0
            else Config.ui.BEAR_COLOUR
        )
        fig.add_annotation(
            x=len(curve) - 1,
            y=final_val,
            text=f"{final_val - 1:+.1%}",
            font=dict(
                family="IBM Plex Mono",
                size=11,
                color=colour,
            ),
            showarrow=True,
            arrowhead=2,
            arrowcolor=colour,
            bgcolor="#0a1220",
            bordercolor=colour,
        )

        fig.update_layout(
            **{**PTHEME,
               "yaxis": {**PTHEME["yaxis"],
                         "title": "Portfolio Value (Base=1.0)",
                         "tickformat": ".2f"},
               "xaxis": {**PTHEME["xaxis"],
                         "title": "Trading Steps"}},
            height=300,
            title=dict(text=title, font=dict(size=12, color="#546e7a")),
            legend=dict(
                orientation="h", y=1.08,
                font=dict(family="IBM Plex Mono", size=9, color="#90a4ae"),
            ),
            hovermode="x unified",
        )
        return fig


# =============================================================================
# 2. Drawdown Chart
# =============================================================================

class DrawdownChart:
    """
    Underwater (drawdown) chart showing peak-to-trough losses over time.
    The most important risk visualisation for any trading strategy.
    """

    @staticmethod
    def build(
        backtest: BacktestResult,
        title:    str = "Drawdown (Underwater) Chart",
    ) -> go.Figure:
        curve    = backtest.equity_curve
        roll_max = curve.cummax()
        drawdown = (curve - roll_max) / (roll_max + 1e-9) * 100
        x_range  = list(range(len(drawdown)))

        fig = go.Figure()

        # Drawdown area
        fig.add_trace(go.Scatter(
            x=x_range,
            y=drawdown.values,
            name="Drawdown",
            fill="tozeroy",
            line=dict(color=Config.ui.BEAR_COLOUR, width=1),
            fillcolor="rgba(255,23,68,0.15)",
            hovertemplate=(
                "Step: %{x}<br>"
                "Drawdown: %{y:.2f}%<br>"
                "<extra></extra>"
            ),
        ))

        # Max drawdown line
        max_dd = float(drawdown.min())
        fig.add_hline(
            y=max_dd,
            line=dict(
                color=Config.ui.BEAR_COLOUR,
                width=1.5,
                dash="dash",
            ),
            annotation_text=f"Max DD: {max_dd:.1f}%",
            annotation_font=dict(
                color=Config.ui.BEAR_COLOUR,
                size=9,
                family="IBM Plex Mono",
            ),
        )

        # Zero line
        fig.add_hline(
            y=0,
            line=dict(color="#546e7a", width=0.8),
        )

        fig.update_layout(
            **{**PTHEME,
               "yaxis": {**PTHEME["yaxis"],
                         "title": "Drawdown (%)",
                         "ticksuffix": "%"},
               "xaxis": {**PTHEME["xaxis"],
                         "title": "Trading Steps"}},
            height=220,
            title=dict(text=title, font=dict(size=12, color="#546e7a")),
            showlegend=False,
        )
        return fig


# =============================================================================
# 3. Rolling Metrics Chart
# =============================================================================

class RollingMetricsChart:
    """
    Rolling Sharpe Ratio and Rolling Volatility charts.
    Shows how strategy performance evolves over time —
    identifies periods of strength and weakness.
    """

    @staticmethod
    def build(
        backtest:     BacktestResult,
        window:       int = 20,
        title:        str = "Rolling Performance Metrics",
    ) -> go.Figure:
        curve   = backtest.equity_curve
        returns = curve.pct_change().dropna()

        fig = make_subplots(
            rows=2, cols=1,
            shared_xaxes=True,
            row_heights=[0.55, 0.45],
            vertical_spacing=0.06,
            subplot_titles=["Rolling Sharpe Ratio (20-step)", "Rolling Volatility"],
        )

        x_range = list(range(len(returns)))

        # Rolling Sharpe
        daily_rf      = Config.backtest.RISK_FREE_RATE / 252
        excess        = returns - daily_rf
        roll_sharpe   = (
            excess.rolling(window).mean() /
            (excess.rolling(window).std() + 1e-9) *
            np.sqrt(252)
        ).fillna(0)

        fig.add_trace(go.Scatter(
            x=x_range,
            y=roll_sharpe.values,
            name=f"Sharpe ({window}-step)",
            line=dict(color=Config.ui.PRIMARY_COLOUR, width=1.5),
            hovertemplate="Step: %{x}<br>Sharpe: %{y:.3f}<extra></extra>",
        ), row=1, col=1)

        fig.add_hline(
            y=1.0,
            line=dict(color=Config.ui.BULL_COLOUR, width=1, dash="dot"),
            annotation_text="Sharpe=1.0",
            annotation_font=dict(
                color=Config.ui.BULL_COLOUR, size=8, family="IBM Plex Mono"
            ),
            row=1, col=1,
        )
        fig.add_hline(
            y=0.0,
            line=dict(color="#546e7a", width=0.8),
            row=1, col=1,
        )

        # Rolling Volatility
        roll_vol = (
            returns.rolling(window).std() * np.sqrt(252) * 100
        ).fillna(0)

        fig.add_trace(go.Scatter(
            x=x_range,
            y=roll_vol.values,
            name=f"Volatility ({window}-step)",
            line=dict(color=Config.ui.WARNING_COLOUR, width=1.5),
            fill="tozeroy",
            fillcolor="rgba(255,196,0,0.07)",
            hovertemplate="Step: %{x}<br>Vol: %{y:.1f}%<extra></extra>",
        ), row=2, col=1)

        # Crisis threshold
        fig.add_hline(
            y=Config.model.CRISIS_VOL_THRESHOLD * 100,
            line=dict(
                color=Config.ui.CRISIS_COLOUR,
                width=1,
                dash="dot",
            ),
            annotation_text="Crisis threshold",
            annotation_font=dict(
                color=Config.ui.CRISIS_COLOUR,
                size=8,
                family="IBM Plex Mono",
            ),
            row=2, col=1,
        )

        fig.update_layout(
            **PTHEME,
            height=340,
            title=dict(
                text=title,
                font=dict(size=12, color="#546e7a"),
            ),
            showlegend=False,
        )
        fig.update_yaxes(
            gridcolor="#111d2e",
            zerolinecolor="#111d2e",
        )
        return fig


# =============================================================================
# 4. Trade Distribution
# =============================================================================

class TradeDistribution:
    """
    Win/loss trade distribution and statistics.
    Shows the distribution of trade P&L values — the shape
    reveals whether the strategy is consistent or relies on
    a few large wins to compensate many small losses.
    """

    @staticmethod
    def build_histogram(
        backtest: BacktestResult,
        title:    str = "Trade P&L Distribution",
    ) -> go.Figure:
        """Build P&L distribution histogram."""
        log = backtest.trade_log
        if log.empty or "pnl" not in log.columns:
            fig = go.Figure()
            fig.update_layout(**PTHEME, height=220)
            fig.add_annotation(
                text="No trades to display",
                xref="paper", yref="paper",
                x=0.5, y=0.5, showarrow=False,
                font=dict(color="#546e7a", size=12),
            )
            return fig

        pnl    = log["pnl"].values
        wins   = pnl[pnl > 0]
        losses = pnl[pnl < 0]

        fig = go.Figure()

        if len(wins) > 0:
            fig.add_trace(go.Histogram(
                x=wins,
                name=f"Wins ({len(wins)})",
                marker_color=Config.ui.BULL_COLOUR,
                opacity=0.75,
                nbinsx=min(20, len(wins)),
                hovertemplate="P&L: %{x:.4f}<br>Count: %{y}<extra>Win</extra>",
            ))

        if len(losses) > 0:
            fig.add_trace(go.Histogram(
                x=losses,
                name=f"Losses ({len(losses)})",
                marker_color=Config.ui.BEAR_COLOUR,
                opacity=0.75,
                nbinsx=min(20, len(losses)),
                hovertemplate="P&L: %{x:.4f}<br>Count: %{y}<extra>Loss</extra>",
            ))

        fig.add_vline(
            x=0,
            line=dict(color="#546e7a", width=1.5),
        )

        fig.update_layout(
            **{**PTHEME,
               "yaxis": {**PTHEME["yaxis"], "title": "Trade Count"},
               "xaxis": {**PTHEME["xaxis"], "title": "P&L per Trade"}},
            height=240,
            title=dict(text=title, font=dict(size=12, color="#546e7a")),
            barmode="overlay",
            legend=dict(
                orientation="h", y=1.08,
                font=dict(family="IBM Plex Mono", size=9, color="#90a4ae"),
            ),
        )
        return fig

    @staticmethod
    def build_stats_table(backtest: BacktestResult) -> go.Figure:
        """Build compact stats summary table."""
        log = backtest.trade_log

        if log.empty:
            return go.Figure()

        pnl = log["pnl"] if "pnl" in log.columns else pd.Series([0])

        rows = [
            ["Total Trades",     str(backtest.n_trades)],
            ["Win Rate",         f"{backtest.win_rate:.1%}"],
            ["Avg Win",          f"{pnl[pnl>0].mean():.5f}" if len(pnl[pnl>0]) else "N/A"],
            ["Avg Loss",         f"{pnl[pnl<0].mean():.5f}" if len(pnl[pnl<0]) else "N/A"],
            ["Best Trade",       f"{pnl.max():.5f}"],
            ["Worst Trade",      f"{pnl.min():.5f}"],
            ["Sharpe Ratio",     f"{backtest.sharpe_ratio:.3f}"],
            ["Max Drawdown",     f"{backtest.max_drawdown:.1%}"],
            ["Annual Return",    f"{backtest.annualised_return:.1%}"],
            ["RMSE",             f"{backtest.rmse:.5f}"],
        ]

        headers = [r[0] for r in rows]
        values  = [r[1] for r in rows]

        fig = go.Figure(go.Table(
            header=dict(
                values=["<b>Metric</b>", "<b>Value</b>"],
                fill_color="#0b0f1a",
                align="left",
                font=dict(family="IBM Plex Mono", size=10, color="#00b4d8"),
                height=28,
            ),
            cells=dict(
                values=[headers, values],
                fill_color=[
                    ["#0d1521", "#0a1220"] * (len(rows) // 2 + 1),
                    ["#0d1521", "#0a1220"] * (len(rows) // 2 + 1),
                ],
                align="left",
                font=dict(family="IBM Plex Mono", size=10, color="#90a4ae"),
                height=24,
            ),
        ))
        fig.update_layout(
            **PTHEME,
            height=80 + len(rows) * 25,
            margin=dict(l=0, r=0, t=8, b=0),
        )
        return fig


# =============================================================================
# 5. Ablation Comparer
# =============================================================================

class AblationComparer:
    """
    Side-by-side visualisation comparing:
      Hybrid Architecture (Bi-LSTM + NLP + Fusion)
      Price-Only Baseline (Bi-LSTM alone)

    This is the key academic contribution visualisation —
    it quantifies the value added by geopolitical NLP.
    """

    @staticmethod
    def build_equity_comparison(
        ablation: AblationResult,
        title:    str = "Ablation Study — Equity Curve Comparison",
    ) -> go.Figure:
        """Compare equity curves of both models."""
        hybrid_curve   = ablation.hybrid.equity_curve
        baseline_curve = ablation.price_only.equity_curve
        x_range        = list(range(max(len(hybrid_curve), len(baseline_curve))))

        fig = go.Figure()

        # Hybrid model
        fig.add_trace(go.Scatter(
            x=list(range(len(hybrid_curve))),
            y=hybrid_curve.values,
            name="Hybrid (BiLSTM + NLP)",
            line=dict(color=Config.ui.PRIMARY_COLOUR, width=2),
            fill="tozeroy",
            fillcolor="rgba(0,180,216,0.06)",
            hovertemplate="Step: %{x}<br>Hybrid: %{y:.4f}<extra></extra>",
        ))

        # Price-only baseline
        fig.add_trace(go.Scatter(
            x=list(range(len(baseline_curve))),
            y=baseline_curve.values,
            name="Price-Only (BiLSTM)",
            line=dict(
                color=Config.ui.NEUTRAL_COLOUR,
                width=1.5,
                dash="dash",
            ),
            hovertemplate="Step: %{x}<br>Baseline: %{y:.4f}<extra></extra>",
        ))

        # Initial capital
        fig.add_hline(
            y=1.0,
            line=dict(color="#37474f", width=0.8, dash="dot"),
        )

        fig.update_layout(
            **{**PTHEME,
               "yaxis": {**PTHEME["yaxis"],
                         "title": "Portfolio Value",
                         "tickformat": ".3f"}},
            height=300,
            title=dict(text=title, font=dict(size=12, color="#546e7a")),
            legend=dict(
                orientation="h", y=1.08,
                font=dict(family="IBM Plex Mono", size=9, color="#90a4ae"),
            ),
            hovermode="x unified",
        )
        return fig

    @staticmethod
    def build_metrics_comparison(
        ablation: AblationResult,
        title:    str = "Ablation Study — Metrics Comparison",
    ) -> go.Figure:
        """Bar chart comparing key metrics."""
        h = ablation.hybrid
        p = ablation.price_only

        metrics = ["Sharpe Ratio", "Annual Return", "Win Rate"]
        hybrid_vals  = [
            h.sharpe_ratio,
            h.annualised_return * 100,
            h.win_rate * 100,
        ]
        baseline_vals = [
            p.sharpe_ratio,
            p.annualised_return * 100,
            p.win_rate * 100,
        ]

        fig = go.Figure()

        fig.add_trace(go.Bar(
            name="Hybrid (BiLSTM + NLP)",
            x=metrics,
            y=hybrid_vals,
            marker_color=Config.ui.PRIMARY_COLOUR,
            opacity=0.85,
            text=[f"{v:.2f}" for v in hybrid_vals],
            textposition="outside",
            textfont=dict(family="IBM Plex Mono", size=10, color="#eceff1"),
        ))

        fig.add_trace(go.Bar(
            name="Price-Only (BiLSTM)",
            x=metrics,
            y=baseline_vals,
            marker_color=Config.ui.NEUTRAL_COLOUR,
            opacity=0.85,
            text=[f"{v:.2f}" for v in baseline_vals],
            textposition="outside",
            textfont=dict(family="IBM Plex Mono", size=10, color="#eceff1"),
        ))

        fig.update_layout(
            **{**PTHEME,
               "yaxis": {**PTHEME["yaxis"], "title": "Value"}},
            height=300,
            title=dict(text=title, font=dict(size=12, color="#546e7a")),
            barmode="group",
            legend=dict(
                orientation="h", y=1.08,
                font=dict(family="IBM Plex Mono", size=9, color="#90a4ae"),
            ),
        )
        return fig

    @staticmethod
    def build_improvement_summary(ablation: AblationResult) -> None:
        """Render improvement summary metrics in Streamlit."""
        imp_sharpe  = ablation.sharpe_improvement
        imp_rmse    = ablation.rmse_improvement
        imp_dd      = ablation.drawdown_improvement

        s_col = Config.ui.BULL_COLOUR if imp_sharpe > 0 else Config.ui.BEAR_COLOUR
        r_col = Config.ui.BULL_COLOUR if imp_rmse   > 0 else Config.ui.BEAR_COLOUR
        d_col = Config.ui.BULL_COLOUR if imp_dd     > 0 else Config.ui.BEAR_COLOUR

        c1, c2, c3 = st.columns(3)
        c1.metric(
            "Sharpe Improvement",
            f"{imp_sharpe:+.3f}",
            delta=f"{'Better' if imp_sharpe > 0 else 'Worse'}",
            delta_color="normal",
        )
        c2.metric(
            "RMSE Improvement",
            f"{imp_rmse:+.5f}",
            delta=f"{'Better' if imp_rmse > 0 else 'Worse'}",
            delta_color="normal",
        )
        c3.metric(
            "Max DD Improvement",
            f"{imp_dd:+.3f}",
            delta=f"{'Better' if imp_dd > 0 else 'Worse'}",
            delta_color="normal",
        )

        st.markdown(
            f'<div style="font-family:IBM Plex Mono;font-size:0.72rem;'
            f'color:#546e7a;margin-top:4px;">'
            f'Text signal value: '
            f'<span style="color:{s_col};">'
            f'{ablation.text_signal_value}</span>'
            f'</div>',
            unsafe_allow_html=True,
        )


# =============================================================================
# 6. Walk-Forward Chart
# =============================================================================

class WalkForwardChart:
    """
    Visualises walk-forward validation results.
    Each fold shows RMSE and direction accuracy on out-of-sample data.
    """

    @staticmethod
    def build(
        fold_details: pd.DataFrame,
        title:        str = "Walk-Forward Validation Results",
    ) -> go.Figure:
        """Build walk-forward validation chart."""
        if fold_details.empty:
            fig = go.Figure()
            fig.update_layout(**PTHEME, height=260)
            return fig

        fig = make_subplots(
            rows=2, cols=1,
            shared_xaxes=True,
            row_heights=[0.5, 0.5],
            vertical_spacing=0.08,
            subplot_titles=["RMSE per Fold", "Direction Accuracy per Fold"],
        )

        folds = fold_details["fold"].tolist()

        # RMSE per fold
        fig.add_trace(go.Bar(
            x=folds,
            y=fold_details["rmse"].tolist(),
            name="RMSE",
            marker_color=Config.ui.WARNING_COLOUR,
            opacity=0.8,
            hovertemplate="Fold %{x}<br>RMSE: %{y:.5f}<extra></extra>",
        ), row=1, col=1)

        # Mean RMSE line
        mean_rmse = fold_details["rmse"].mean()
        fig.add_hline(
            y=mean_rmse,
            line=dict(color=Config.ui.PRIMARY_COLOUR, width=1.5, dash="dot"),
            annotation_text=f"Mean: {mean_rmse:.5f}",
            annotation_font=dict(
                color=Config.ui.PRIMARY_COLOUR,
                size=8,
                family="IBM Plex Mono",
            ),
            row=1, col=1,
        )

        # Direction accuracy per fold
        acc_col = fold_details["direction_accuracy"].tolist()
        colours = [
            Config.ui.BULL_COLOUR if v >= 0.5 else Config.ui.BEAR_COLOUR
            for v in acc_col
        ]
        fig.add_trace(go.Bar(
            x=folds,
            y=acc_col,
            name="Direction Accuracy",
            marker_color=colours,
            opacity=0.8,
            hovertemplate="Fold %{x}<br>Accuracy: %{y:.1%}<extra></extra>",
        ), row=2, col=1)

        # 50% baseline
        fig.add_hline(
            y=0.5,
            line=dict(color="#546e7a", width=1, dash="dot"),
            annotation_text="Random (50%)",
            annotation_font=dict(color="#546e7a", size=8),
            row=2, col=1,
        )

        fig.update_layout(
            **PTHEME,
            height=340,
            title=dict(
                text=title,
                font=dict(size=12, color="#546e7a"),
            ),
            showlegend=False,
        )
        fig.update_yaxes(gridcolor="#111d2e", zerolinecolor="#111d2e")
        return fig


# =============================================================================
# 7. BacktestVisualiser — Orchestrates all charts
# =============================================================================

class BacktestVisualiser:
    """
    Orchestrates all backtesting charts into a Streamlit section.

    Usage in app.py:
        viz = BacktestVisualiser()
        viz.render(backtest_result, ablation_result)
    """

    def render(
        self,
        backtest:  BacktestResult,
        ablation:  Optional[AblationResult] = None,
        benchmark: Optional[np.ndarray] = None,
        walk_forward_folds: Optional[pd.DataFrame] = None,
        ticker:    str = "SPY",
        market:    str = "US",
    ) -> None:
        """Render full backtesting dashboard in Streamlit."""

        st.markdown(
            '<div style="font-family:IBM Plex Mono;font-size:0.65rem;'
            'color:#546e7a;text-transform:uppercase;letter-spacing:0.12em;'
            'margin-bottom:0.5rem;">📊 Backtesting Results</div>',
            unsafe_allow_html=True,
        )

        # ── Top metrics row ──────────────────────────────────────────
        m1, m2, m3, m4, m5 = st.columns(5)
        m1.metric("Sharpe Ratio",   f"{backtest.sharpe_ratio:.3f}")
        m2.metric("Max Drawdown",   f"{backtest.max_drawdown:.1%}")
        m3.metric("RMSE",           f"{backtest.rmse:.5f}")
        m4.metric("Annual Return",  f"{backtest.annualised_return:.1%}")
        m5.metric("Win Rate",       f"{backtest.win_rate:.1%}")

        st.divider()

        # ── Equity curve + drawdown ───────────────────────────────────
        col1, col2 = st.columns([1.4, 1])
        with col1:
            st.plotly_chart(
                EquityCurveChart.build(backtest, benchmark),
                use_container_width=True,
                config={"displayModeBar": False},
            )
        with col2:
            st.plotly_chart(
                DrawdownChart.build(backtest),
                use_container_width=True,
                config={"displayModeBar": False},
            )

        # ── Rolling metrics ───────────────────────────────────────────
        st.plotly_chart(
            RollingMetricsChart.build(backtest),
            use_container_width=True,
            config={"displayModeBar": False},
        )

        # ── Trade distribution ────────────────────────────────────────
        col3, col4 = st.columns([1.2, 1])
        with col3:
            st.plotly_chart(
                TradeDistribution.build_histogram(backtest),
                use_container_width=True,
                config={"displayModeBar": False},
            )
        with col4:
            st.plotly_chart(
                TradeDistribution.build_stats_table(backtest),
                use_container_width=True,
                config={"displayModeBar": False},
            )

        # ── Ablation study ────────────────────────────────────────────
        if ablation is not None:
            st.divider()
            st.markdown(
                '<div style="font-family:IBM Plex Mono;font-size:0.65rem;'
                'color:#546e7a;text-transform:uppercase;letter-spacing:0.12em;'
                'margin-bottom:0.5rem;">🧪 Ablation Study — '
                'Hybrid vs Price-Only Baseline</div>',
                unsafe_allow_html=True,
            )

            AblationComparer.build_improvement_summary(ablation)

            col5, col6 = st.columns(2)
            with col5:
                st.plotly_chart(
                    AblationComparer.build_equity_comparison(ablation),
                    use_container_width=True,
                    config={"displayModeBar": False},
                )
            with col6:
                st.plotly_chart(
                    AblationComparer.build_metrics_comparison(ablation),
                    use_container_width=True,
                    config={"displayModeBar": False},
                )

        # ── Walk-forward validation ───────────────────────────────────
        if walk_forward_folds is not None and not walk_forward_folds.empty:
            st.divider()
            st.markdown(
                '<div style="font-family:IBM Plex Mono;font-size:0.65rem;'
                'color:#546e7a;text-transform:uppercase;letter-spacing:0.12em;'
                'margin-bottom:0.5rem;">🔄 Walk-Forward Validation</div>',
                unsafe_allow_html=True,
            )
            st.plotly_chart(
                WalkForwardChart.build(walk_forward_folds),
                use_container_width=True,
                config={"displayModeBar": False},
            )
            mean_acc = walk_forward_folds["direction_accuracy"].mean()
            st.caption(
                f"Mean direction accuracy across {len(walk_forward_folds)} folds: "
                f"{mean_acc:.1%}  "
                f"({'above' if mean_acc > 0.5 else 'below'} random baseline of 50%)"
            )


# =============================================================================
# CLI smoke test
# =============================================================================

if __name__ == "__main__":
    print("=" * 60)
    print("SGEAIA Backtest Visualiser - Smoke Test")
    print("=" * 60)

    from model_utils import BacktestResult, AblationResult
    import pandas as pd

    # Mock data
    N = 100
    returns = np.random.randn(N) * 0.01
    equity  = pd.Series((1 + returns).cumprod(), name="equity_curve")

    trades = pd.DataFrame({
        "step":   range(N // 2),
        "signal": ["LONG"] * (N // 4) + ["SHORT"] * (N // 4),
        "alpha":  np.random.randn(N // 2) * 0.1,
        "actual": np.random.randn(N // 2) * 0.01,
        "pnl":    np.random.randn(N // 2) * 0.005,
        "win":    np.random.rand(N // 2) > 0.45,
    })

    mock_bt = BacktestResult(
        sharpe_ratio=1.42,
        max_drawdown=-0.087,
        rmse=0.00312,
        annualised_return=0.187,
        total_return=0.045,
        n_trades=len(trades),
        win_rate=float(trades["win"].mean()),
        equity_curve=equity,
        trade_log=trades,
    )

    mock_baseline = BacktestResult(
        sharpe_ratio=0.98,
        max_drawdown=-0.112,
        rmse=0.00445,
        annualised_return=0.124,
        total_return=0.030,
        n_trades=len(trades),
        win_rate=0.48,
        equity_curve=pd.Series((1 + np.random.randn(N) * 0.008).cumprod()),
        trade_log=trades,
    )

    mock_ablation = AblationResult(
        hybrid=mock_bt,
        price_only=mock_baseline,
    )

    print(f"\n  Sharpe improvement: {mock_ablation.sharpe_improvement:+.3f}")
    print(f"  RMSE improvement:   {mock_ablation.rmse_improvement:+.5f}")
    print(f"  Text signal value:  {mock_ablation.text_signal_value}")

    # Test chart builds
    fig_ec  = EquityCurveChart.build(mock_bt)
    fig_dd  = DrawdownChart.build(mock_bt)
    fig_rm  = RollingMetricsChart.build(mock_bt)
    fig_hist = TradeDistribution.build_histogram(mock_bt)
    fig_abl  = AblationComparer.build_equity_comparison(mock_ablation)
    fig_bar  = AblationComparer.build_metrics_comparison(mock_ablation)

    print(f"\n  EquityCurveChart:    {len(fig_ec.data)} trace(s)")
    print(f"  DrawdownChart:       {len(fig_dd.data)} trace(s)")
    print(f"  RollingMetrics:      {len(fig_rm.data)} trace(s)")
    print(f"  TradeHistogram:      {len(fig_hist.data)} trace(s)")
    print(f"  AblationEquity:      {len(fig_abl.data)} trace(s)")
    print(f"  AblationMetrics:     {len(fig_bar.data)} trace(s)")

    # Walk-forward
    fold_data = pd.DataFrame({
        "fold":               range(1, 6),
        "train_size":         [60, 80, 100, 120, 140],
        "rmse":               np.random.uniform(0.002, 0.006, 5),
        "direction_accuracy": np.random.uniform(0.45, 0.65, 5),
    })
    fig_wf = WalkForwardChart.build(fold_data)
    print(f"  WalkForwardChart:    {len(fig_wf.data)} trace(s)")

    print("\nAll visualisation tests passed.")