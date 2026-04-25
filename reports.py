"""
reports.py
==========
Synchronized Geopolitical-Equity Alpha Inference Architecture
-------------------------------------------------------------
PDF report generation using ReportLab.

  ReportGenerator    - Full intelligence report builder
  ReportSection      - Individual report section builder
  ChartExporter      - Exports Plotly charts as images for PDF embedding
  ReportScheduler    - Manages report generation timing
"""

from __future__ import annotations

import io
import logging
import os
from dataclasses import dataclass
from datetime import datetime
from typing import Optional

import numpy as np
import pandas as pd
import streamlit as st

from config import Config

logger = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s - %(message)s",
)


# =============================================================================
# Data containers
# =============================================================================

@dataclass
class ReportData:
    """All data needed to generate a report."""
    ticker:          str
    market:          str
    currency:        str
    report_date:     str
    alpha_signal:    float
    confidence:      float
    regime:          str
    sentiment:       str
    crisis_mode:     bool
    current_vol:     float
    latest_price:    float
    daily_change_pct: float
    top_headlines:   list[str]
    sharpe_ratio:    Optional[float] = None
    max_drawdown:    Optional[float] = None
    rmse:            Optional[float] = None
    annualised_return: Optional[float] = None
    win_rate:        Optional[float] = None
    hybrid_sharpe:   Optional[float] = None
    baseline_sharpe: Optional[float] = None
    hybrid_rmse:     Optional[float] = None
    baseline_rmse:   Optional[float] = None

    @property
    def signal_label(self) -> str:
        if self.alpha_signal > Config.backtest.LONG_THRESHOLD:
            return "LONG"
        if self.alpha_signal < Config.backtest.SHORT_THRESHOLD:
            return "SHORT"
        return "FLAT"

    @property
    def currency_symbol(self) -> str:
        symbols = {
            "PKR": "PKR ", "USD": "USD ",
            "GBP": "GBP ", "EUR": "EUR ",
        }
        return symbols.get(self.currency, self.currency + " ")

    @property
    def has_backtest(self) -> bool:
        return self.sharpe_ratio is not None


# =============================================================================
# Colour palette (ReportLab HEX colours)
# =============================================================================

class Colours:
    DARK_BG     = (11/255,  15/255,  26/255)   # #0b0f1a
    PANEL_BG    = (13/255,  21/255,  33/255)   # #0d1521
    BORDER      = (22/255,  32/255,  48/255)   # #162030
    PRIMARY     = (0/255,  180/255, 216/255)   # #00b4d8 cyan
    BULL        = (0/255,  230/255, 118/255)   # #00e676 green
    BEAR        = (255/255, 23/255,  68/255)   # #ff1744 red
    NEUTRAL     = (84/255,  110/255, 122/255)  # #546e7a grey
    WARNING     = (255/255, 196/255,  0/255)   # #ffc400 yellow
    TEXT_MAIN   = (176/255, 190/255, 197/255)  # #b0bec5
    TEXT_DIM    = (84/255,  110/255, 122/255)  # #546e7a
    WHITE       = (1.0, 1.0, 1.0)
    PSX_GREEN   = (76/255,  175/255,  80/255)  # #4caf50


# =============================================================================
# Report Generator
# =============================================================================

class ReportGenerator:
    """
    Generates professional PDF intelligence reports using ReportLab.

    Report structure:
      Page 1: Cover page with key metrics
      Page 2: Alpha signal analysis + sentiment breakdown
      Page 3: Backtesting results (if available)
      Page 4: Ablation study (if available)
      Page 5: Intelligence feed (top headlines)
      Page 6: Disclaimer + methodology notes
    """

    def __init__(self):
        self._check_reportlab()

    def _check_reportlab(self) -> None:
        try:
            import reportlab
            self.available = True
        except ImportError:
            self.available = False
            logger.warning(
                "reportlab not installed. "
                "PDF generation unavailable. "
                "Run: pip install reportlab"
            )

    def generate(self, data: ReportData) -> Optional[bytes]:
        """
        Generate complete PDF report.
        Returns PDF bytes or None if reportlab unavailable.
        """
        if not self.available:
            return None

        try:
            return self._build_pdf(data)
        except Exception as exc:
            logger.error("PDF generation failed: %s", exc)
            return None

    def _build_pdf(self, data: ReportData) -> bytes:
        from reportlab.lib.pagesizes import A4
        from reportlab.lib.units import cm
        from reportlab.platypus import (
            SimpleDocTemplate, Paragraph, Spacer,
            Table, TableStyle, PageBreak, HRFlowable,
        )
        from reportlab.lib.styles import ParagraphStyle
        from reportlab.lib.enums import TA_LEFT, TA_CENTER, TA_RIGHT
        from reportlab.lib import colors

        buf = io.BytesIO()
        doc = SimpleDocTemplate(
            buf,
            pagesize=A4,
            leftMargin=1.8*cm, rightMargin=1.8*cm,
            topMargin=1.8*cm,  bottomMargin=1.8*cm,
            title=f"SGEAIA Report — {data.ticker}",
            author="Synchronized Geopolitical-Equity Alpha Inference Architecture",
            subject=f"Intelligence Report {data.report_date}",
        )

        # ── Paragraph styles ──────────────────────────────────────────
        styles = self._build_styles()
        story  = []

        # ── Cover Page ────────────────────────────────────────────────
        story += self._cover_page(data, styles, colors)
        story.append(PageBreak())

        # ── Signal Analysis Page ──────────────────────────────────────
        story += self._signal_page(data, styles, colors)
        story.append(PageBreak())

        # ── Backtesting Page ──────────────────────────────────────────
        if data.has_backtest:
            story += self._backtest_page(data, styles, colors)
            story.append(PageBreak())

        # ── Intelligence Feed Page ────────────────────────────────────
        story += self._headlines_page(data, styles, colors)
        story.append(PageBreak())

        # ── Disclaimer Page ───────────────────────────────────────────
        story += self._disclaimer_page(data, styles, colors)

        doc.build(story)
        return buf.getvalue()

    # ── Styles ────────────────────────────────────────────────────────

    def _build_styles(self) -> dict:
        from reportlab.lib.styles import ParagraphStyle
        from reportlab.lib.enums import TA_LEFT, TA_CENTER, TA_RIGHT
        from reportlab.lib import colors

        c = Colours

        def rgb(r, g, b):
            return colors.Color(r, g, b)

        return {
            "title": ParagraphStyle(
                "title",
                fontName="Courier-Bold",
                fontSize=18,
                textColor=rgb(*c.PRIMARY),
                spaceAfter=8,
                alignment=TA_LEFT,
                leading=22,
            ),
            "subtitle": ParagraphStyle(
                "subtitle",
                fontName="Courier",
                fontSize=9,
                textColor=rgb(*c.TEXT_DIM),
                spaceAfter=4,
                alignment=TA_LEFT,
            ),
            "section": ParagraphStyle(
                "section",
                fontName="Courier-Bold",
                fontSize=11,
                textColor=rgb(*c.PRIMARY),
                spaceBefore=12,
                spaceAfter=6,
                leading=14,
            ),
            "body": ParagraphStyle(
                "body",
                fontName="Courier",
                fontSize=8.5,
                textColor=rgb(*c.TEXT_MAIN),
                spaceAfter=4,
                leading=13,
            ),
            "small": ParagraphStyle(
                "small",
                fontName="Courier",
                fontSize=7.5,
                textColor=rgb(*c.TEXT_DIM),
                spaceAfter=3,
                leading=11,
            ),
            "metric_label": ParagraphStyle(
                "metric_label",
                fontName="Courier",
                fontSize=7,
                textColor=rgb(*c.TEXT_DIM),
                alignment=TA_CENTER,
            ),
            "metric_value": ParagraphStyle(
                "metric_value",
                fontName="Courier-Bold",
                fontSize=14,
                textColor=rgb(*c.WHITE),
                alignment=TA_CENTER,
            ),
            "bull": ParagraphStyle(
                "bull",
                fontName="Courier-Bold",
                fontSize=12,
                textColor=rgb(*c.BULL),
                alignment=TA_CENTER,
            ),
            "bear": ParagraphStyle(
                "bear",
                fontName="Courier-Bold",
                fontSize=12,
                textColor=rgb(*c.BEAR),
                alignment=TA_CENTER,
            ),
            "disclaimer": ParagraphStyle(
                "disclaimer",
                fontName="Courier",
                fontSize=7,
                textColor=rgb(*c.TEXT_DIM),
                spaceAfter=4,
                leading=11,
            ),
            "center": ParagraphStyle(
                "center",
                fontName="Courier",
                fontSize=8.5,
                textColor=rgb(*c.TEXT_MAIN),
                alignment=TA_CENTER,
            ),
        }

    # ── Cover Page ────────────────────────────────────────────────────

    def _cover_page(self, data: ReportData, styles: dict, colors) -> list:
        from reportlab.lib.units import cm
        from reportlab.platypus import Spacer, Table, TableStyle, HRFlowable
        from reportlab.lib import colors as rl_colors

        def rgb(r, g, b):
            return rl_colors.Color(r, g, b)

        story = []
        c = Colours

        # Header
        story.append(Paragraph(
            "SGEAIA INTELLIGENCE REPORT",
            styles["title"],
        ))
        story.append(Paragraph(
            "Synchronized Geopolitical-Equity Alpha Inference Architecture",
            styles["subtitle"],
        ))
        story.append(Paragraph(
            f"Reuters  |  Financial Times  |  Al Jazeera  |  "
            f"Nikkei Asia  |  WSJ  |  Bloomberg  |  Dawn",
            styles["small"],
        ))
        story.append(HRFlowable(
            width="100%", thickness=1,
            color=rgb(*c.PRIMARY),
            spaceAfter=16,
        ))

        # Report metadata
        meta_data = [
            ["Ticker",       data.ticker],
            ["Market",       data.market],
            ["Currency",     data.currency],
            ["Report Date",  data.report_date],
            ["Generated",    datetime.utcnow().strftime("%Y-%m-%d %H:%M UTC")],
        ]
        meta_table = Table(meta_data, colWidths=[4*cm, 10*cm])
        meta_table.setStyle(TableStyle([
            ("FONTNAME",    (0,0), (-1,-1), "Courier"),
            ("FONTSIZE",    (0,0), (-1,-1), 8.5),
            ("TEXTCOLOR",   (0,0), (0,-1), rgb(*c.TEXT_DIM)),
            ("TEXTCOLOR",   (1,0), (1,-1), rgb(*c.TEXT_MAIN)),
            ("FONTNAME",    (0,0), (0,-1), "Courier-Bold"),
            ("ROWBACKGROUNDS", (0,0), (-1,-1),
             [rgb(*c.PANEL_BG), rgb(*c.DARK_BG)]),
            ("TOPPADDING",  (0,0), (-1,-1), 4),
            ("BOTTOMPADDING", (0,0), (-1,-1), 4),
            ("LEFTPADDING", (0,0), (-1,-1), 8),
        ]))
        story.append(meta_table)
        story.append(Spacer(1, 16))

        # Key metrics grid
        story.append(Paragraph("KEY METRICS", styles["section"]))

        signal_colour = (
            c.BULL if data.signal_label == "LONG" else
            c.BEAR if data.signal_label == "SHORT" else
            c.NEUTRAL
        )
        regime_colour = (
            c.BULL  if data.regime == "Bull" else
            c.BEAR  if data.regime == "Bear" else
            c.NEUTRAL
        )

        metrics = [
            ("ALPHA SIGNAL",   f"{data.alpha_signal:+.4f}", signal_colour),
            ("DIRECTION",      data.signal_label,            signal_colour),
            ("CONFIDENCE",     f"{data.confidence:.1%}",     c.PRIMARY),
            ("REGIME",         data.regime,                  regime_colour),
            ("SENTIMENT",      data.sentiment,               c.PRIMARY),
            ("VOLATILITY",     f"{data.current_vol:.1%}",    c.WARNING),
        ]

        metric_rows = []
        row = []
        for i, (label, value, colour) in enumerate(metrics):
            cell_data = [
                Paragraph(label, styles["metric_label"]),
                Paragraph(value, ParagraphStyleHelper.coloured(
                    styles["metric_value"], rgb(*colour), colors
                )),
            ]
            row.append(cell_data)
            if len(row) == 3:
                metric_rows.append(row)
                row = []
        if row:
            while len(row) < 3:
                row.append(["", ""])
            metric_rows.append(row)

        from reportlab.lib.styles import ParagraphStyle as PS

        for mrow in metric_rows:
            flat = []
            for cell in mrow:
                flat.append(cell)
            t = Table([flat], colWidths=[5.5*cm, 5.5*cm, 5.5*cm])
            t.setStyle(TableStyle([
                ("BACKGROUND",    (0,0), (-1,-1), rgb(*c.PANEL_BG)),
                ("GRID",          (0,0), (-1,-1), 0.5, rgb(*c.BORDER)),
                ("TOPPADDING",    (0,0), (-1,-1), 8),
                ("BOTTOMPADDING", (0,0), (-1,-1), 8),
                ("ALIGN",         (0,0), (-1,-1), "CENTER"),
                ("VALIGN",        (0,0), (-1,-1), "MIDDLE"),
            ]))
            story.append(t)
            story.append(Spacer(1, 2))

        # Price info
        story.append(Spacer(1, 10))
        chg_col = c.BULL if data.daily_change_pct >= 0 else c.BEAR
        story.append(Paragraph(
            f"Latest Price: {data.currency_symbol}{data.latest_price:,.4f}  "
            f"| Daily Change: {data.daily_change_pct:+.2f}%"
            f"{'  | CRISIS MODE ACTIVE' if data.crisis_mode else ''}",
            styles["body"],
        ))

        return story

    # ── Signal Analysis Page ──────────────────────────────────────────

    def _signal_page(self, data: ReportData, styles: dict, colors) -> list:
        from reportlab.lib.units import cm
        from reportlab.platypus import Spacer, Table, TableStyle, HRFlowable
        from reportlab.lib import colors as rl_colors

        def rgb(r, g, b):
            return rl_colors.Color(r, g, b)

        c = Colours
        story = []

        story.append(Paragraph("SIGNAL ANALYSIS", styles["section"]))
        story.append(HRFlowable(
            width="100%", thickness=0.5,
            color=rgb(*c.BORDER), spaceAfter=8,
        ))

        # Alpha explanation
        story.append(Paragraph("Alpha Signal Interpretation", styles["section"]))
        story.append(Paragraph(
            f"The alpha signal of {data.alpha_signal:+.4f} indicates a "
            f"{data.signal_label} recommendation for {data.ticker}. "
            f"This signal is generated by fusing 60-day price momentum "
            f"(Bi-LSTM encoder) with real-time geopolitical sentiment "
            f"(VADER + TextBlob NLP pipeline) through a cross-modal "
            f"attention fusion layer.",
            styles["body"],
        ))
        story.append(Spacer(1, 6))

        # Regime analysis
        story.append(Paragraph("Market Regime Classification", styles["section"]))
        regime_text = {
            "Bull":    (
                "The model classifies current market conditions as BULL REGIME. "
                "Characterised by rising prices, positive momentum, and "
                "favourable geopolitical environment. Historically, bull "
                "regimes favour long positions and momentum strategies."
            ),
            "Bear":    (
                "The model classifies current market conditions as BEAR REGIME. "
                "Characterised by falling prices, negative momentum, and "
                "adverse geopolitical signals. Bear regimes typically favour "
                "defensive positioning, short exposure, or cash preservation."
            ),
            "Neutral": (
                "The model classifies current market conditions as NEUTRAL REGIME. "
                "Mixed signals from price and geopolitical streams. "
                "Neutral regimes typically favour range-trading strategies "
                "or reduced position sizes until a clearer trend emerges."
            ),
        }
        story.append(Paragraph(
            regime_text.get(data.regime, "Regime classification unavailable."),
            styles["body"],
        ))
        story.append(Spacer(1, 6))

        # Sentiment table
        story.append(Paragraph("Geopolitical Sentiment", styles["section"]))
        story.append(Paragraph(
            f"Current sentiment: {data.sentiment}. "
            f"Geopolitical headlines from Reuters, Financial Times, "
            f"Al Jazeera, Nikkei Asia, WSJ, Bloomberg, Dawn, and "
            f"Business Recorder are processed by the VADER + TextBlob "
            f"pipeline. Articles are ranked by impact score using a "
            f"DSA Priority Queue (heapq-based max-heap).",
            styles["body"],
        ))
        story.append(Spacer(1, 6))

        # Crisis mode section
        if data.crisis_mode:
            story.append(Paragraph("CRISIS MODE STATUS", styles["section"]))
            story.append(Paragraph(
                f"CRISIS MODE IS ACTIVE. Realised volatility "
                f"({data.current_vol:.1%}) has exceeded the "
                f"{Config.model.CRISIS_VOL_THRESHOLD:.0%} threshold. "
                f"The fusion layer has elevated the geopolitical text "
                f"stream weighting. During crisis periods, qualitative "
                f"news signals historically outperform quantitative "
                f"price signals as the primary return predictor.",
                styles["body"],
            ))

        return story

    # ── Backtesting Page ──────────────────────────────────────────────

    def _backtest_page(self, data: ReportData, styles: dict, colors) -> list:
        from reportlab.lib.units import cm
        from reportlab.platypus import Spacer, Table, TableStyle, HRFlowable
        from reportlab.lib import colors as rl_colors

        def rgb(r, g, b):
            return rl_colors.Color(r, g, b)

        c = Colours
        story = []

        story.append(Paragraph("BACKTESTING & PERFORMANCE METRICS", styles["section"]))
        story.append(HRFlowable(
            width="100%", thickness=0.5,
            color=rgb(*c.BORDER), spaceAfter=8,
        ))

        # Metrics table
        rows = [
            ["METRIC", "HYBRID MODEL", "INTERPRETATION"],
            ["Sharpe Ratio",
             f"{data.sharpe_ratio:.3f}" if data.sharpe_ratio else "N/A",
             "Risk-adjusted return (>1.0 = good, >2.0 = excellent)"],
            ["Max Drawdown",
             f"{data.max_drawdown:.1%}" if data.max_drawdown else "N/A",
             "Worst peak-to-trough loss"],
            ["RMSE",
             f"{data.rmse:.5f}" if data.rmse else "N/A",
             "Prediction accuracy vs actual returns"],
            ["Annual Return",
             f"{data.annualised_return:.1%}" if data.annualised_return else "N/A",
             "Annualised strategy return"],
            ["Win Rate",
             f"{data.win_rate:.1%}" if data.win_rate else "N/A",
             "% of trades with positive P&L"],
        ]

        t = Table(rows, colWidths=[4.5*cm, 4*cm, 8*cm])
        t.setStyle(TableStyle([
            ("BACKGROUND",    (0,0), (-1,0),  rgb(*c.PRIMARY)),
            ("TEXTCOLOR",     (0,0), (-1,0),  rgb(*c.DARK_BG)),
            ("FONTNAME",      (0,0), (-1,0),  "Courier-Bold"),
            ("FONTSIZE",      (0,0), (-1,-1), 8),
            ("FONTNAME",      (0,1), (-1,-1), "Courier"),
            ("TEXTCOLOR",     (0,1), (-1,-1), rgb(*c.TEXT_MAIN)),
            ("TEXTCOLOR",     (0,1), (0,-1),  rgb(*c.TEXT_DIM)),
            ("ROWBACKGROUNDS",(0,1), (-1,-1),
             [rgb(*c.PANEL_BG), rgb(*c.DARK_BG)]),
            ("GRID",          (0,0), (-1,-1), 0.3, rgb(*c.BORDER)),
            ("TOPPADDING",    (0,0), (-1,-1), 5),
            ("BOTTOMPADDING", (0,0), (-1,-1), 5),
            ("LEFTPADDING",   (0,0), (-1,-1), 6),
        ]))
        story.append(t)
        story.append(Spacer(1, 12))

        # Ablation study
        if data.hybrid_sharpe is not None and data.baseline_sharpe is not None:
            story.append(Paragraph("ABLATION STUDY", styles["section"]))
            story.append(Paragraph(
                "Comparison of Hybrid Architecture (Bi-LSTM + NLP + Fusion) "
                "against Price-Only Baseline (Bi-LSTM alone):",
                styles["body"],
            ))
            story.append(Spacer(1, 6))

            abl_rows = [
                ["METRIC",        "HYBRID",                       "PRICE-ONLY",                    "IMPROVEMENT"],
                ["Sharpe Ratio",
                 f"{data.hybrid_sharpe:.3f}",
                 f"{data.baseline_sharpe:.3f}",
                 f"{data.hybrid_sharpe - data.baseline_sharpe:+.3f}"],
                ["RMSE",
                 f"{data.hybrid_rmse:.5f}" if data.hybrid_rmse else "N/A",
                 f"{data.baseline_rmse:.5f}" if data.baseline_rmse else "N/A",
                 f"{(data.baseline_rmse or 0) - (data.hybrid_rmse or 0):+.5f}"],
            ]
            at = Table(abl_rows, colWidths=[4*cm, 3.5*cm, 3.5*cm, 4*cm])
            at.setStyle(TableStyle([
                ("BACKGROUND",    (0,0), (-1,0),  rgb(*c.PRIMARY)),
                ("TEXTCOLOR",     (0,0), (-1,0),  rgb(*c.DARK_BG)),
                ("FONTNAME",      (0,0), (-1,0),  "Courier-Bold"),
                ("FONTSIZE",      (0,0), (-1,-1), 8),
                ("FONTNAME",      (0,1), (-1,-1), "Courier"),
                ("TEXTCOLOR",     (0,1), (-1,-1), rgb(*c.TEXT_MAIN)),
                ("ROWBACKGROUNDS",(0,1), (-1,-1),
                 [rgb(*c.PANEL_BG), rgb(*c.DARK_BG)]),
                ("GRID",          (0,0), (-1,-1), 0.3, rgb(*c.BORDER)),
                ("TOPPADDING",    (0,0), (-1,-1), 5),
                ("BOTTOMPADDING", (0,0), (-1,-1), 5),
                ("LEFTPADDING",   (0,0), (-1,-1), 6),
            ]))
            story.append(at)

        return story

    # ── Headlines Page ────────────────────────────────────────────────

    def _headlines_page(self, data: ReportData, styles: dict, colors) -> list:
        from reportlab.lib.units import cm
        from reportlab.platypus import Spacer, Table, TableStyle, HRFlowable
        from reportlab.lib import colors as rl_colors

        def rgb(r, g, b):
            return rl_colors.Color(r, g, b)

        c = Colours
        story = []

        story.append(Paragraph("INTELLIGENCE FEED", styles["section"]))
        story.append(HRFlowable(
            width="100%", thickness=0.5,
            color=rgb(*c.BORDER), spaceAfter=8,
        ))
        story.append(Paragraph(
            "Top geopolitical headlines ranked by impact score "
            "(Priority Queue — recency x credibility x keyword density):",
            styles["body"],
        ))
        story.append(Spacer(1, 6))

        headlines = data.top_headlines[:15]
        if headlines:
            for i, headline in enumerate(headlines, 1):
                clean = headline[:180] if headline else ""
                story.append(Paragraph(
                    f"{i:02d}.  {clean}",
                    styles["small"],
                ))
                story.append(Spacer(1, 2))
        else:
            story.append(Paragraph(
                "No headlines available. Configure GNews API key in "
                "Streamlit secrets to enable live intelligence feed.",
                styles["body"],
            ))

        return story

    # ── Disclaimer Page ───────────────────────────────────────────────

    def _disclaimer_page(self, data: ReportData, styles: dict, colors) -> list:
        from reportlab.lib.units import cm
        from reportlab.platypus import Spacer, HRFlowable
        from reportlab.lib import colors as rl_colors

        def rgb(r, g, b):
            return rl_colors.Color(r, g, b)

        c = Colours
        story = []

        story.append(Paragraph("METHODOLOGY", styles["section"]))
        story.append(Paragraph(
            "This report is generated by the Synchronized Geopolitical-Equity "
            "Alpha Inference Architecture (SGEAIA). The system uses a hybrid "
            "deep learning architecture combining:",
            styles["body"],
        ))
        story.append(Spacer(1, 4))

        components = [
            "Bi-LSTM Encoder: Processes 60-day OHLCV price sequences with "
            "9 engineered features (returns, volatility, RSI, Bollinger Bands)",
            "Sentiment Encoder: VADER + TextBlob pure-Python NLP pipeline "
            "for financial text sentiment analysis",
            "Weighted Fusion Layer: Cross-modal attention mechanism that "
            "dynamically weights price vs text signals based on market regime",
            "Crisis Mode: Automatically elevates geopolitical text signal "
            "weight when realised volatility exceeds 30% annualised threshold",
            "DSA Optimisations: Trie (keyword detection), LRU Cache "
            "(API response caching), Priority Queue (article ranking), "
            "Binary Search (date alignment)",
        ]
        for comp in components:
            story.append(Paragraph(f"  - {comp}", styles["small"]))
            story.append(Spacer(1, 2))

        story.append(Spacer(1, 12))
        story.append(HRFlowable(
            width="100%", thickness=0.5,
            color=rgb(*c.BORDER), spaceAfter=8,
        ))
        story.append(Paragraph("IMPORTANT DISCLAIMER", styles["section"]))

        disclaimer_text = [
            Config.pdf.FOOTER_TEXT,
            "",
            "The model weights in this version are randomly initialised. "
            "To generate real alpha signals, the model must be trained on "
            "labeled historical return data. This report represents the "
            "system's architectural output, not a trained prediction.",
            "",
            "Past performance of any strategy shown in backtesting results "
            "does not guarantee future performance. Markets are inherently "
            "unpredictable. Always conduct independent research.",
            "",
            "This software is provided for educational purposes only. "
            "The developers accept no liability for financial decisions "
            "made based on this system's output.",
        ]
        for line in disclaimer_text:
            if line:
                story.append(Paragraph(line, styles["disclaimer"]))
            else:
                story.append(Spacer(1, 4))

        story.append(Spacer(1, 12))
        story.append(Paragraph(
            f"SGEAIA  |  Generated {datetime.utcnow().strftime('%Y-%m-%d %H:%M UTC')}  "
            f"|  {data.ticker} {data.market}",
            styles["small"],
        ))

        return story


# =============================================================================
# Helper: coloured paragraph style
# =============================================================================

class ParagraphStyleHelper:
    @staticmethod
    def coloured(base_style, colour, colors_module):
        """Return a copy of a ParagraphStyle with a different text colour."""
        from reportlab.lib.styles import ParagraphStyle
        new_style = ParagraphStyle(
            base_style.name + "_coloured",
            parent=base_style,
            textColor=colour,
        )
        return new_style


# =============================================================================
# Streamlit download button helper
# =============================================================================

def render_pdf_download_button(
    data:        ReportData,
    button_label: str = "Download PDF Report",
) -> None:
    """
    Render a Streamlit download button for the PDF report.
    Shows a spinner while generating, then presents download.
    """
    generator = ReportGenerator()

    if not generator.available:
        st.warning(
            "PDF generation requires `reportlab`. "
            "Add it to requirements.txt and redeploy."
        )
        return

    with st.spinner("Generating PDF report..."):
        pdf_bytes = generator.generate(data)

    if pdf_bytes:
        filename = (
            f"SGEAIA_{data.ticker}_{data.report_date.replace('-', '')}.pdf"
        )
        st.download_button(
            label=button_label,
            data=pdf_bytes,
            file_name=filename,
            mime="application/pdf",
            use_container_width=True,
        )
        st.caption(
            f"Report: {data.ticker} | {data.report_date} | "
            f"{len(pdf_bytes) // 1024} KB"
        )
    else:
        st.error("PDF generation failed. Check logs for details.")


def build_report_data(
    bundle,
    result,
    backtest=None,
    ablation=None,
) -> ReportData:
    """
    Build ReportData from app session state objects.

    Args:
        bundle:   AlignedBundle from data_loader
        result:   InferenceResult from model_utils
        backtest: BacktestResult (optional)
        ablation: AblationResult (optional)
    """
    headlines = []
    if hasattr(bundle, "ranked_articles") and bundle.ranked_articles:
        headlines = [
            f"[{a.source_label}] {a.title}"
            for a in bundle.ranked_articles[:15]
            if a.title
        ]
    elif hasattr(bundle, "articles") and bundle.articles:
        headlines = [
            f"[{a.source_label}] {a.title}"
            for a in bundle.articles[:15]
            if a.title
        ]

    return ReportData(
        ticker=bundle.ticker,
        market=bundle.market if hasattr(bundle, "market") else "US",
        currency=bundle.currency if hasattr(bundle, "currency") else "USD",
        report_date=datetime.utcnow().strftime("%Y-%m-%d"),
        alpha_signal=result.alpha,
        confidence=result.confidence,
        regime=result.regime,
        sentiment=result.sentiment,
        crisis_mode=result.crisis_mode,
        current_vol=result.current_vol,
        latest_price=bundle.ohlcv.latest_price,
        daily_change_pct=bundle.ohlcv.daily_change_pct,
        top_headlines=headlines,
        sharpe_ratio=backtest.sharpe_ratio if backtest else None,
        max_drawdown=backtest.max_drawdown if backtest else None,
        rmse=backtest.rmse if backtest else None,
        annualised_return=backtest.annualised_return if backtest else None,
        win_rate=backtest.win_rate if backtest else None,
        hybrid_sharpe=ablation.hybrid.sharpe_ratio if ablation else None,
        baseline_sharpe=ablation.price_only.sharpe_ratio if ablation else None,
        hybrid_rmse=ablation.hybrid.rmse if ablation else None,
        baseline_rmse=ablation.price_only.rmse if ablation else None,
    )


# =============================================================================
# CLI smoke test
# =============================================================================

if __name__ == "__main__":
    print("=" * 60)
    print("SGEAIA Reports Module - Smoke Test")
    print("=" * 60)

    generator = ReportGenerator()
    print(f"\n  ReportLab available: {generator.available}")

    if generator.available:
        mock_data = ReportData(
            ticker="ENGRO.KA",
            market="PSX",
            currency="PKR",
            report_date="2026-04-22",
            alpha_signal=0.18,
            confidence=0.72,
            regime="Bull",
            sentiment="Positive",
            crisis_mode=False,
            current_vol=0.22,
            latest_price=285.50,
            daily_change_pct=1.34,
            top_headlines=[
                "[Business Recorder] KSE-100 hits record high on IMF deal",
                "[Dawn] Pakistan economy shows signs of stabilisation",
                "[Reuters] CPEC Phase-2 infrastructure projects accelerate",
                "[Financial Times] Emerging markets rally on rate cut hopes",
                "[Al Jazeera] Regional stability boosts investor confidence",
            ],
            sharpe_ratio=1.42,
            max_drawdown=-0.08,
            rmse=0.00312,
            annualised_return=0.187,
            win_rate=0.61,
            hybrid_sharpe=1.42,
            baseline_sharpe=0.98,
            hybrid_rmse=0.00312,
            baseline_rmse=0.00445,
        )

        pdf_bytes = generator.generate(mock_data)
        if pdf_bytes:
            out_path = "test_report.pdf"
            with open(out_path, "wb") as f:
                f.write(pdf_bytes)
            print(f"\n  PDF generated: {out_path} ({len(pdf_bytes)//1024} KB)")
            print("  Open test_report.pdf to verify output.")
        else:
            print("\n  PDF generation returned None.")
    else:
        print("\n  Install reportlab to test: pip install reportlab")

    print("\nSmoke test complete.")
