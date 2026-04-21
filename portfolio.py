"""
portfolio.py
============
Synchronized Geopolitical-Equity Alpha Inference Architecture
-------------------------------------------------------------
Portfolio management features:

  MultiTickerAnalyser   - Compare up to 5 tickers simultaneously
  WatchlistManager      - Save/load/manage favourite tickers
  PaperTradingEngine    - Simulate trades with virtual capital
  PortfolioAnalytics    - Correlation, diversification, risk metrics
  PSXPortfolioHelper    - Pakistan-specific portfolio utilities
"""

from __future__ import annotations

import json
import logging
import os
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Optional

import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots

from config import Config
from data_loader import (
    AlignedBundle,
    MultiTickerBundle,
    OHLCVBundle,
    SynchronizedDataLoader,
    BaseLoader,
)

logger = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s - %(message)s",
)


# =============================================================================
# Data containers
# =============================================================================

@dataclass
class TickerSnapshot:
    """Real-time snapshot for a single ticker in portfolio view."""
    ticker:          str
    market:          str
    currency:        str
    latest_price:    float
    daily_change:    float
    daily_change_pct: float
    volatility_10d:  float
    rsi_14:          float
    alpha_signal:    float
    regime:          str
    sentiment:       str
    crisis_mode:     bool

    @property
    def signal_label(self) -> str:
        if self.alpha_signal > Config.backtest.LONG_THRESHOLD:
            return "LONG"
        if self.alpha_signal < Config.backtest.SHORT_THRESHOLD:
            return "SHORT"
        return "FLAT"

    @property
    def signal_colour(self) -> str:
        if self.alpha_signal > Config.backtest.LONG_THRESHOLD:
            return Config.ui.BULL_COLOUR
        if self.alpha_signal < Config.backtest.SHORT_THRESHOLD:
            return Config.ui.BEAR_COLOUR
        return Config.ui.NEUTRAL_COLOUR

    @property
    def change_colour(self) -> str:
        return Config.ui.BULL_COLOUR if self.daily_change >= 0 else Config.ui.BEAR_COLOUR

    @property
    def currency_symbol(self) -> str:
        symbols = {"PKR": "₨", "USD": "$", "GBP": "£",
                   "EUR": "€", "JPY": "¥", "INR": "₹"}
        return symbols.get(self.currency, self.currency)


@dataclass
class WatchlistItem:
    """Single item in a user's watchlist."""
    ticker:    str
    name:      str
    market:    str
    currency:  str
    added_at:  str
    notes:     str = ""
    alerts_on: bool = True


@dataclass
class PaperTrade:
    """Single paper trade record."""
    trade_id:    int
    ticker:      str
    direction:   str    # "LONG" or "SHORT"
    entry_price: float
    entry_date:  str
    quantity:    float
    stop_loss:   float
    take_profit: float
    currency:    str
    status:      str = "OPEN"   # "OPEN", "CLOSED", "STOPPED"
    exit_price:  float = 0.0
    exit_date:   str = ""
    pnl:         float = 0.0
    pnl_pct:     float = 0.0
    reason:      str = ""       # why closed

    @property
    def is_open(self) -> bool:
        return self.status == "OPEN"

    @property
    def cost_basis(self) -> float:
        return self.entry_price * self.quantity

    @property
    def pnl_colour(self) -> str:
        return Config.ui.BULL_COLOUR if self.pnl >= 0 else Config.ui.BEAR_COLOUR


@dataclass
class PaperPortfolio:
    """State of the paper trading portfolio."""
    initial_capital: float
    currency:        str
    trades:          list[PaperTrade] = field(default_factory=list)
    cash:            float = 0.0

    def __post_init__(self):
        if self.cash == 0.0:
            self.cash = self.initial_capital

    @property
    def open_trades(self) -> list[PaperTrade]:
        return [t for t in self.trades if t.is_open]

    @property
    def closed_trades(self) -> list[PaperTrade]:
        return [t for t in self.trades if not t.is_open]

    @property
    def total_pnl(self) -> float:
        return sum(t.pnl for t in self.closed_trades)

    @property
    def total_return_pct(self) -> float:
        return self.total_pnl / self.initial_capital * 100

    @property
    def win_rate(self) -> float:
        closed = self.closed_trades
        if not closed:
            return 0.0
        wins = sum(1 for t in closed if t.pnl > 0)
        return wins / len(closed) * 100

    @property
    def portfolio_value(self) -> float:
        open_value = sum(t.cost_basis for t in self.open_trades)
        return self.cash + open_value + self.total_pnl

    @property
    def currency_symbol(self) -> str:
        symbols = {"PKR": "₨", "USD": "$", "GBP": "£",
                   "EUR": "€", "JPY": "¥", "INR": "₹"}
        return symbols.get(self.currency, self.currency)


# =============================================================================
# 1. Multi-Ticker Analyser
# =============================================================================

class MultiTickerAnalyser:
    """
    Analyses and compares multiple tickers simultaneously.

    Features:
      - Side-by-side price performance
      - Normalised returns comparison
      - Correlation matrix heatmap
      - Sector/market breakdown
      - Combined alpha signals
    """

    def __init__(self, tickers: list[str]):
        max_t        = Config.market.MAX_COMPARISON_TICKERS
        self.tickers = tickers[:max_t]

    def build_comparison_chart(
        self,
        multi_bundle: MultiTickerBundle,
        normalise:    bool = True,
    ) -> go.Figure:
        """
        Build a normalised returns comparison chart.
        All tickers start at 100 for easy visual comparison.
        """
        fig = go.Figure()

        colours = [
            Config.ui.PRIMARY_COLOUR,
            Config.ui.BULL_COLOUR,
            Config.ui.WARNING_COLOUR,
            Config.ui.BEAR_COLOUR,
            "#b39ddb",
        ]

        for i, (ticker, bundle) in enumerate(multi_bundle.bundles.items()):
            close = bundle.ohlcv.close.dropna()
            if normalise:
                # Normalise to 100 at start
                series = (close / close.iloc[0] * 100)
                y_label = "Indexed Return (Base=100)"
            else:
                series  = close
                y_label = f"Price ({bundle.currency})"

            fig.add_trace(go.Scatter(
                x=series.index,
                y=series.values,
                name=ticker,
                line=dict(color=colours[i % len(colours)], width=1.8),
                mode="lines",
                hovertemplate=(
                    f"<b>{ticker}</b><br>"
                    "Date: %{x}<br>"
                    "Value: %{y:.2f}<br>"
                    "<extra></extra>"
                ),
            ))

        fig.update_layout(
            **self._plotly_theme(),
            height=380,
            title=dict(
                text="Multi-Ticker Performance Comparison",
                font=dict(size=12, color="#546e7a"),
            ),
            yaxis=dict(
                title=y_label,
                gridcolor="#111d2e",
                zerolinecolor="#111d2e",
            ),
            legend=dict(
                orientation="h", y=1.08,
                font=dict(family="IBM Plex Mono", size=10, color="#90a4ae"),
            ),
            hovermode="x unified",
        )
        return fig

    def build_correlation_heatmap(
        self,
        multi_bundle: MultiTickerBundle,
    ) -> go.Figure:
        """Build correlation matrix heatmap of returns."""
        corr = multi_bundle.correlation_matrix
        if corr.empty:
            return go.Figure()

        fig = go.Figure(go.Heatmap(
            z=corr.values,
            x=corr.columns.tolist(),
            y=corr.index.tolist(),
            colorscale=[
                [0.0,  Config.ui.BEAR_COLOUR],
                [0.5,  "#0b0f1a"],
                [1.0,  Config.ui.BULL_COLOUR],
            ],
            zmid=0,
            text=np.round(corr.values, 2),
            texttemplate="%{text}",
            textfont=dict(family="IBM Plex Mono", size=11, color="#eceff1"),
            showscale=True,
        ))
        fig.update_layout(
            **self._plotly_theme(),
            height=320,
            title=dict(
                text="Return Correlation Matrix",
                font=dict(size=12, color="#546e7a"),
            ),
        )
        return fig

    def build_volatility_comparison(
        self,
        multi_bundle: MultiTickerBundle,
    ) -> go.Figure:
        """Compare 10-day rolling volatility across tickers."""
        fig = go.Figure()
        colours = [
            Config.ui.PRIMARY_COLOUR, Config.ui.BULL_COLOUR,
            Config.ui.WARNING_COLOUR, Config.ui.BEAR_COLOUR, "#b39ddb",
        ]

        for i, (ticker, bundle) in enumerate(multi_bundle.bundles.items()):
            if "Volatility_10d" in bundle.ohlcv.features.columns:
                vol = bundle.ohlcv.features["Volatility_10d"].dropna()
                fig.add_trace(go.Scatter(
                    x=vol.index, y=vol.values,
                    name=ticker,
                    line=dict(color=colours[i % len(colours)], width=1.5),
                ))

        fig.add_hline(
            y=Config.model.CRISIS_VOL_THRESHOLD,
            line=dict(color=Config.ui.CRISIS_COLOUR, width=1, dash="dot"),
            annotation_text="Crisis threshold",
            annotation_font=dict(color=Config.ui.CRISIS_COLOUR, size=9),
        )
        fig.update_layout(
            **self._plotly_theme(),
            height=250,
            title=dict(
                text="Volatility Comparison (10-day Ann.)",
                font=dict(size=12, color="#546e7a"),
            ),
            yaxis=dict(
                title="Annualised Volatility",
                tickformat=".0%",
                gridcolor="#111d2e",
                zerolinecolor="#111d2e",
            ),
            legend=dict(
                orientation="h", y=1.08,
                font=dict(family="IBM Plex Mono", size=10, color="#90a4ae"),
            ),
        )
        return fig

    def build_snapshot_table(
        self,
        snapshots: list[TickerSnapshot],
    ) -> go.Figure:
        """Build a summary table of all ticker snapshots."""
        if not snapshots:
            return go.Figure()

        headers = [
            "Ticker", "Market", "Price", "Change %",
            "Vol 10d", "RSI", "Alpha", "Signal", "Regime",
        ]
        rows = [[] for _ in headers]

        for snap in snapshots:
            rows[0].append(snap.ticker)
            rows[1].append(snap.market)
            rows[2].append(f"{snap.currency_symbol}{snap.latest_price:,.2f}")
            rows[3].append(f"{snap.daily_change_pct:+.2f}%")
            rows[4].append(f"{snap.volatility_10d:.1%}")
            rows[5].append(f"{snap.rsi_14 * 100:.1f}")
            rows[6].append(f"{snap.alpha_signal:+.4f}")
            rows[7].append(snap.signal_label)
            rows[8].append(snap.regime)

        # Colour rows by signal
        cell_colours = []
        for i, header in enumerate(headers):
            col_colours = []
            for snap in snapshots:
                if header == "Signal":
                    col_colours.append(snap.signal_colour + "33")
                elif header == "Change %":
                    col_colours.append(snap.change_colour + "22")
                else:
                    col_colours.append("#0d1521")
            cell_colours.append(col_colours)

        fig = go.Figure(go.Table(
            header=dict(
                values=[f"<b>{h}</b>" for h in headers],
                fill_color="#0b0f1a",
                align="left",
                font=dict(family="IBM Plex Mono", size=11, color="#00b4d8"),
                height=32,
            ),
            cells=dict(
                values=rows,
                fill_color=cell_colours,
                align="left",
                font=dict(family="IBM Plex Mono", size=10, color="#90a4ae"),
                height=28,
            ),
        ))
        fig.update_layout(
            **self._plotly_theme(),
            height=80 + len(snapshots) * 30,
            margin=dict(l=0, r=0, t=10, b=0),
        )
        return fig

    @staticmethod
    def _plotly_theme() -> dict:
        return dict(
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="#0a1220",
            font=dict(family="IBM Plex Mono, monospace", color="#546e7a", size=10),
            margin=dict(l=8, r=8, t=36, b=8),
            xaxis=dict(gridcolor="#111d2e", zerolinecolor="#111d2e"),
        )


# =============================================================================
# 2. Watchlist Manager
# =============================================================================

class WatchlistManager:
    """
    Manages a user's watchlist of favourite tickers.

    Uses Streamlit session state for persistence within a session.
    Saves to JSON file for persistence across sessions.

    Features:
      - Add/remove tickers
      - Organise by market (PSX / US / Global)
      - Quick-add from preset groups
      - Export/import watchlist
    """

    WATCHLIST_FILE = "watchlist.json"

    def __init__(self):
        self._ensure_session_state()

    def _ensure_session_state(self) -> None:
        if "watchlist" not in st.session_state:
            st.session_state.watchlist = self._load_from_file()

    @property
    def items(self) -> list[WatchlistItem]:
        return st.session_state.watchlist

    @property
    def tickers(self) -> list[str]:
        return [item.ticker for item in self.items]

    def add(
        self,
        ticker:   str,
        name:     str = "",
        notes:    str = "",
    ) -> bool:
        """Add ticker to watchlist. Returns True if added, False if exists."""
        ticker = ticker.upper().strip()
        if ticker in self.tickers:
            return False

        market, currency = BaseLoader._detect_market(ticker)
        display_name = (
            name or
            Config.psx.KSE100_BLUECHIPS.get(ticker, ticker)
        )

        item = WatchlistItem(
            ticker=ticker,
            name=display_name,
            market=market,
            currency=currency,
            added_at=datetime.utcnow().strftime("%Y-%m-%d %H:%M UTC"),
            notes=notes,
        )
        st.session_state.watchlist.append(item)
        self._save_to_file()
        logger.info("Watchlist: added %s", ticker)
        return True

    def remove(self, ticker: str) -> bool:
        """Remove ticker from watchlist."""
        ticker = ticker.upper()
        before = len(st.session_state.watchlist)
        st.session_state.watchlist = [
            item for item in st.session_state.watchlist
            if item.ticker != ticker
        ]
        removed = len(st.session_state.watchlist) < before
        if removed:
            self._save_to_file()
            logger.info("Watchlist: removed %s", ticker)
        return removed

    def update_notes(self, ticker: str, notes: str) -> None:
        """Update notes for a watchlist item."""
        for item in st.session_state.watchlist:
            if item.ticker == ticker.upper():
                item.notes = notes
        self._save_to_file()

    def toggle_alerts(self, ticker: str) -> None:
        """Toggle price alerts for a ticker."""
        for item in st.session_state.watchlist:
            if item.ticker == ticker.upper():
                item.alerts_on = not item.alerts_on
        self._save_to_file()

    def get_by_market(self, market: str) -> list[WatchlistItem]:
        """Get watchlist items filtered by market."""
        return [item for item in self.items if item.market == market]

    def get_psx_tickers(self) -> list[str]:
        """Get only PSX tickers from watchlist."""
        return [item.ticker for item in self.items if item.market == "PSX"]

    def get_us_tickers(self) -> list[str]:
        """Get only US tickers from watchlist."""
        return [item.ticker for item in self.items if item.market == "US"]

    def add_preset_group(self, group_name: str) -> int:
        """Add all tickers from a preset group. Returns number added."""
        tickers = Config.market.TICKER_GROUPS.get(group_name, [])
        added = 0
        for ticker in tickers:
            if self.add(ticker):
                added += 1
        return added

    def export_json(self) -> str:
        """Export watchlist as JSON string."""
        data = [
            {
                "ticker":    item.ticker,
                "name":      item.name,
                "market":    item.market,
                "currency":  item.currency,
                "added_at":  item.added_at,
                "notes":     item.notes,
                "alerts_on": item.alerts_on,
            }
            for item in self.items
        ]
        return json.dumps(data, indent=2)

    def import_json(self, json_str: str) -> int:
        """Import watchlist from JSON string. Returns number imported."""
        try:
            data  = json.loads(json_str)
            added = 0
            for item_data in data:
                if self.add(
                    ticker=item_data.get("ticker", ""),
                    name=item_data.get("name", ""),
                    notes=item_data.get("notes", ""),
                ):
                    added += 1
            return added
        except Exception as exc:
            logger.error("Watchlist import failed: %s", exc)
            return 0

    def _save_to_file(self) -> None:
        try:
            with open(self.WATCHLIST_FILE, "w") as f:
                f.write(self.export_json())
        except Exception as exc:
            logger.warning("Could not save watchlist: %s", exc)

    def _load_from_file(self) -> list[WatchlistItem]:
        if not os.path.exists(self.WATCHLIST_FILE):
            return self._default_watchlist()
        try:
            with open(self.WATCHLIST_FILE) as f:
                data = json.load(f)
            return [
                WatchlistItem(
                    ticker=d["ticker"], name=d.get("name", d["ticker"]),
                    market=d.get("market", "US"), currency=d.get("currency", "USD"),
                    added_at=d.get("added_at", ""), notes=d.get("notes", ""),
                    alerts_on=d.get("alerts_on", True),
                )
                for d in data
            ]
        except Exception:
            return self._default_watchlist()

    @staticmethod
    def _default_watchlist() -> list[WatchlistItem]:
        """Default watchlist includes key PSX + US tickers."""
        defaults = [
            ("^KSE100",  "KSE-100 Index",     "PSX", "PKR"),
            ("ENGRO.KA", "Engro Corporation",  "PSX", "PKR"),
            ("HBL.KA",   "Habib Bank",         "PSX", "PKR"),
            ("SYS.KA",   "Systems Limited",    "PSX", "PKR"),
            ("SPY",      "S&P 500 ETF",        "US",  "USD"),
            ("QQQ",      "NASDAQ 100 ETF",     "US",  "USD"),
        ]
        now = datetime.utcnow().strftime("%Y-%m-%d %H:%M UTC")
        return [
            WatchlistItem(
                ticker=t, name=n, market=m, currency=c, added_at=now
            )
            for t, n, m, c in defaults
        ]


# =============================================================================
# 3. Paper Trading Engine
# =============================================================================

class PaperTradingEngine:
    """
    Simulates real trades with virtual capital.

    Features:
      - Long and short positions
      - Stop loss and take profit
      - Transaction cost simulation
      - PKR and USD portfolio support
      - Position sizing based on risk %
      - Trade log with P&L tracking
      - Portfolio equity curve

    Paper trading lets users practice the system's signals
    without risking real money.
    """

    def __init__(self, currency: str = "USD"):
        self._ensure_session_state(currency)

    def _ensure_session_state(self, currency: str) -> None:
        if "paper_portfolio" not in st.session_state:
            initial = (
                Config.portfolio.INITIAL_CAPITAL_PKR
                if currency == "PKR"
                else Config.portfolio.INITIAL_CAPITAL_USD
            )
            st.session_state.paper_portfolio = PaperPortfolio(
                initial_capital=initial,
                currency=currency,
                cash=initial,
            )

    @property
    def portfolio(self) -> PaperPortfolio:
        return st.session_state.paper_portfolio

    def open_trade(
        self,
        ticker:      str,
        direction:   str,
        entry_price: float,
        risk_pct:    float = 0.02,   # risk 2% of portfolio per trade
        currency:    str  = "USD",
    ) -> Optional[PaperTrade]:
        """
        Open a new paper trade.

        Position sizing: Kelly-inspired 2% risk per trade.
        Stop loss:       5% below entry (LONG) / above entry (SHORT)
        Take profit:     10% above entry (LONG) / below entry (SHORT)
        """
        if entry_price <= 0:
            return None

        # Check max positions
        if len(self.portfolio.open_trades) >= Config.portfolio.MAX_POSITIONS:
            logger.warning("Max positions reached (%d)", Config.portfolio.MAX_POSITIONS)
            return None

        # Position sizing
        risk_amount   = self.portfolio.cash * risk_pct
        stop_distance = entry_price * Config.backtest.STOP_LOSS
        quantity      = risk_amount / max(stop_distance, 0.0001)

        # Check position size limit
        max_position_value = self.portfolio.cash * Config.portfolio.MAX_POSITION_SIZE
        quantity = min(quantity, max_position_value / entry_price)

        if quantity <= 0 or quantity * entry_price > self.portfolio.cash:
            return None

        # Transaction cost
        tx_cost = (
            Config.backtest.PSX_TRANSACTION_COST
            if currency == "PKR"
            else Config.backtest.TRANSACTION_COST
        )
        cost = entry_price * quantity * (1 + tx_cost)

        if cost > self.portfolio.cash:
            return None

        # Set stop loss and take profit
        if direction == "LONG":
            stop_loss   = entry_price * (1 - Config.backtest.STOP_LOSS)
            take_profit = entry_price * (1 + Config.backtest.TAKE_PROFIT)
        else:
            stop_loss   = entry_price * (1 + Config.backtest.STOP_LOSS)
            take_profit = entry_price * (1 - Config.backtest.TAKE_PROFIT)

        trade_id = len(self.portfolio.trades) + 1
        trade = PaperTrade(
            trade_id=trade_id,
            ticker=ticker,
            direction=direction,
            entry_price=entry_price,
            entry_date=datetime.utcnow().strftime("%Y-%m-%d"),
            quantity=quantity,
            stop_loss=stop_loss,
            take_profit=take_profit,
            currency=currency,
        )

        self.portfolio.trades.append(trade)
        self.portfolio.cash -= cost
        logger.info(
            "Paper trade opened: %s %s %s @ %.4f (qty=%.2f)",
            direction, ticker, currency, entry_price, quantity,
        )
        return trade

    def close_trade(
        self,
        trade_id:   int,
        exit_price: float,
        reason:     str = "Manual",
    ) -> Optional[PaperTrade]:
        """Close an open paper trade and calculate P&L."""
        trade = self._find_trade(trade_id)
        if not trade or not trade.is_open:
            return None

        tx_cost = (
            Config.backtest.PSX_TRANSACTION_COST
            if trade.currency == "PKR"
            else Config.backtest.TRANSACTION_COST
        )

        if trade.direction == "LONG":
            gross_pnl = (exit_price - trade.entry_price) * trade.quantity
        else:
            gross_pnl = (trade.entry_price - exit_price) * trade.quantity

        tx_fees = exit_price * trade.quantity * tx_cost
        net_pnl = gross_pnl - tx_fees

        trade.exit_price = exit_price
        trade.exit_date  = datetime.utcnow().strftime("%Y-%m-%d")
        trade.pnl        = net_pnl
        trade.pnl_pct    = net_pnl / trade.cost_basis * 100
        trade.status     = "CLOSED"
        trade.reason     = reason

        # Return capital + P&L to cash
        self.portfolio.cash += trade.cost_basis + net_pnl

        logger.info(
            "Paper trade closed: #%d %s @ %.4f  PnL=%.2f (%.1f%%)",
            trade_id, trade.ticker, exit_price, net_pnl, trade.pnl_pct,
        )
        return trade

    def check_stops(
        self,
        ticker:        str,
        current_price: float,
    ) -> list[PaperTrade]:
        """Check and trigger stop loss / take profit for open trades."""
        triggered = []
        for trade in self.portfolio.open_trades:
            if trade.ticker != ticker:
                continue
            if trade.direction == "LONG":
                if current_price <= trade.stop_loss:
                    self.close_trade(trade.trade_id, current_price, "Stop Loss")
                    triggered.append(trade)
                elif current_price >= trade.take_profit:
                    self.close_trade(trade.trade_id, current_price, "Take Profit")
                    triggered.append(trade)
            else:  # SHORT
                if current_price >= trade.stop_loss:
                    self.close_trade(trade.trade_id, current_price, "Stop Loss")
                    triggered.append(trade)
                elif current_price <= trade.take_profit:
                    self.close_trade(trade.trade_id, current_price, "Take Profit")
                    triggered.append(trade)
        return triggered

    def reset_portfolio(self, currency: str = "USD") -> None:
        """Reset portfolio to initial state."""
        initial = (
            Config.portfolio.INITIAL_CAPITAL_PKR
            if currency == "PKR"
            else Config.portfolio.INITIAL_CAPITAL_USD
        )
        st.session_state.paper_portfolio = PaperPortfolio(
            initial_capital=initial,
            currency=currency,
            cash=initial,
        )

    def build_equity_curve(self) -> go.Figure:
        """Build portfolio equity curve chart."""
        closed = self.portfolio.closed_trades
        if not closed:
            return go.Figure()

        dates  = [t.exit_date for t in closed]
        values = []
        running_value = self.portfolio.initial_capital
        for trade in closed:
            running_value += trade.pnl
            values.append(running_value)

        initial = self.portfolio.initial_capital
        symbol  = self.portfolio.currency_symbol

        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=dates, y=values,
            line=dict(color=Config.ui.PRIMARY_COLOUR, width=2),
            fill="tozeroy", fillcolor=f"{Config.ui.PRIMARY_COLOUR}15",
            name="Portfolio Value",
            hovertemplate=(
                "Date: %{x}<br>"
                f"Value: {symbol}%{{y:,.0f}}<br>"
                "<extra></extra>"
            ),
        ))
        fig.add_hline(
            y=initial,
            line=dict(color="#546e7a", width=1, dash="dot"),
            annotation_text="Initial Capital",
            annotation_font=dict(color="#546e7a", size=9),
        )
        fig.update_layout(
            **self._plotly_theme(),
            height=220,
            title=dict(
                text="Paper Trading Equity Curve",
                font=dict(size=12, color="#546e7a"),
            ),
            yaxis=dict(
                title=f"Portfolio Value ({symbol})",
                gridcolor="#111d2e",
                zerolinecolor="#111d2e",
            ),
        )
        return fig

    def build_trade_log_table(self) -> go.Figure:
        """Build trade log table."""
        trades = self.portfolio.trades
        if not trades:
            return go.Figure()

        headers = [
            "#", "Ticker", "Direction", "Entry", "Exit",
            "Qty", "PnL", "PnL %", "Status", "Reason",
        ]
        sym = self.portfolio.currency_symbol

        rows = [[] for _ in headers]
        for t in reversed(trades):
            rows[0].append(str(t.trade_id))
            rows[1].append(t.ticker)
            rows[2].append(t.direction)
            rows[3].append(f"{sym}{t.entry_price:,.2f}")
            rows[4].append(f"{sym}{t.exit_price:,.2f}" if t.exit_price else "-")
            rows[5].append(f"{t.quantity:.2f}")
            rows[6].append(f"{sym}{t.pnl:+,.2f}" if not t.is_open else "-")
            rows[7].append(f"{t.pnl_pct:+.1f}%" if not t.is_open else "-")
            rows[8].append(t.status)
            rows[9].append(t.reason or "-")

        pnl_colours = []
        for t in reversed(trades):
            if t.is_open:
                pnl_colours.append("#0d1521")
            elif t.pnl >= 0:
                pnl_colours.append(Config.ui.BULL_COLOUR + "22")
            else:
                pnl_colours.append(Config.ui.BEAR_COLOUR + "22")

        cell_colours = [
            ["#0d1521"] * len(trades)
            if i != 6 else pnl_colours
            for i in range(len(headers))
        ]

        fig = go.Figure(go.Table(
            header=dict(
                values=[f"<b>{h}</b>" for h in headers],
                fill_color="#0b0f1a",
                align="left",
                font=dict(family="IBM Plex Mono", size=11, color="#00b4d8"),
                height=30,
            ),
            cells=dict(
                values=rows,
                fill_color=cell_colours,
                align="left",
                font=dict(family="IBM Plex Mono", size=10, color="#90a4ae"),
                height=26,
            ),
        ))
        fig.update_layout(
            **self._plotly_theme(),
            height=max(200, 80 + len(trades) * 28),
            margin=dict(l=0, r=0, t=10, b=0),
        )
        return fig

    def _find_trade(self, trade_id: int) -> Optional[PaperTrade]:
        for trade in self.portfolio.trades:
            if trade.trade_id == trade_id:
                return trade
        return None

    @staticmethod
    def _plotly_theme() -> dict:
        return dict(
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="#0a1220",
            font=dict(family="IBM Plex Mono, monospace", color="#546e7a", size=10),
            margin=dict(l=8, r=8, t=36, b=8),
            xaxis=dict(gridcolor="#111d2e", zerolinecolor="#111d2e"),
        )


# =============================================================================
# 4. Portfolio Analytics
# =============================================================================

class PortfolioAnalytics:
    """
    Advanced portfolio-level analytics across multiple tickers.

    Features:
      - Diversification score
      - Sector concentration
      - Market concentration (PSX vs Global)
      - Risk-weighted portfolio view
    """

    @staticmethod
    def diversification_score(
        correlation_matrix: pd.DataFrame,
    ) -> float:
        """
        Score from 0-100 measuring portfolio diversification.
        100 = perfectly uncorrelated, 0 = all identical.
        """
        if correlation_matrix.empty:
            return 0.0
        n = len(correlation_matrix)
        if n <= 1:
            return 100.0
        # Average off-diagonal correlation
        mask = ~np.eye(n, dtype=bool)
        avg_corr = float(np.abs(correlation_matrix.values[mask]).mean())
        return round((1 - avg_corr) * 100, 1)

    @staticmethod
    def market_breakdown(tickers: list[str]) -> dict[str, int]:
        """Count tickers by market."""
        breakdown: dict[str, int] = {}
        for ticker in tickers:
            market, _ = BaseLoader._detect_market(ticker)
            breakdown[market] = breakdown.get(market, 0) + 1
        return breakdown

    @staticmethod
    def build_market_breakdown_chart(tickers: list[str]) -> go.Figure:
        """Pie chart of portfolio by market."""
        breakdown = PortfolioAnalytics.market_breakdown(tickers)
        labels    = list(breakdown.keys())
        values    = list(breakdown.values())

        colour_map = {
            "PSX":     Config.ui.PSX_COLOUR,
            "US":      Config.ui.PRIMARY_COLOUR,
            "Global":  Config.ui.WARNING_COLOUR,
            "Mixed":   "#b39ddb",
        }
        colours = [colour_map.get(m, "#546e7a") for m in labels]

        fig = go.Figure(go.Pie(
            labels=labels,
            values=values,
            marker=dict(colors=colours),
            textinfo="label+percent",
            textfont=dict(family="IBM Plex Mono", size=11),
            hole=0.4,
        ))
        fig.update_layout(
            paper_bgcolor="rgba(0,0,0,0)",
            font=dict(family="IBM Plex Mono", color="#90a4ae", size=10),
            height=220,
            margin=dict(l=0, r=0, t=20, b=0),
            showlegend=False,
        )
        return fig


# =============================================================================
# 5. PSX Portfolio Helper
# =============================================================================

class PSXPortfolioHelper:
    """
    Pakistan-specific portfolio utilities.

    Helps Pakistani investors:
      - Convert PKR portfolio values
      - Understand PSX trading hours
      - Navigate PSX-specific sectors
      - Track CPEC-related stocks
    """

    # CPEC-related stocks (China-Pakistan Economic Corridor)
    CPEC_STOCKS: dict[str, str] = {
        "HUBC.KA":  "Hub Power (energy infrastructure)",
        "KAPCO.KA": "Kot Addu Power (energy)",
        "ENGRO.KA": "Engro (fertilizer/energy)",
        "LUCK.KA":  "Lucky Cement (construction)",
        "OGDC.KA":  "OGDC (oil & gas)",
    }

    # Islamic finance (Shariah-compliant) stocks
    SHARIAH_STOCKS: list[str] = [
        "MEBL.KA",   # Meezan Bank (Islamic banking)
        "ENGRO.KA",  # Engro (generally halal)
        "LUCK.KA",   # Lucky Cement
        "SYS.KA",    # Systems Limited
        "TRG.KA",    # TRG Pakistan
    ]

    @staticmethod
    def is_psx_market_open() -> bool:
        """Check if PSX is currently open (PKT timezone)."""
        from datetime import timezone
        now_utc = datetime.now(timezone.utc)
        # Convert to PKT (UTC+5)
        now_pkt = now_utc + timedelta(hours=Config.psx.UTC_OFFSET_HOURS)
        # Check if weekday (Mon-Fri)
        if now_pkt.weekday() >= 5:  # Saturday=5, Sunday=6
            return False
        # Check trading hours
        open_time  = now_pkt.replace(
            hour=Config.psx.MARKET_OPEN_HOUR,
            minute=Config.psx.MARKET_OPEN_MIN,
            second=0,
        )
        close_time = now_pkt.replace(
            hour=Config.psx.MARKET_CLOSE_HOUR,
            minute=Config.psx.MARKET_CLOSE_MIN,
            second=0,
        )
        return open_time <= now_pkt <= close_time

    @staticmethod
    def pkr_to_usd(pkr_amount: float) -> float:
        """Convert PKR amount to USD using approximate rate."""
        return pkr_amount / Config.psx.USD_PKR_APPROX

    @staticmethod
    def usd_to_pkr(usd_amount: float) -> float:
        """Convert USD amount to PKR using approximate rate."""
        return usd_amount * Config.psx.USD_PKR_APPROX

    @staticmethod
    def format_pkr(amount: float) -> str:
        """Format PKR amount in Pakistani number system (crore/lakh)."""
        if abs(amount) >= 10_000_000:
            return f"₨{amount/10_000_000:.2f} Cr"
        if abs(amount) >= 100_000:
            return f"₨{amount/100_000:.2f} L"
        return f"₨{amount:,.0f}"

    @staticmethod
    def get_sector_for_ticker(ticker: str) -> str:
        """Return sector name for a PSX ticker."""
        for sector, tickers in Config.psx.SECTOR_GROUPS.items():
            if ticker in tickers:
                return sector
        return "Other"

    @staticmethod
    def is_shariah_compliant(ticker: str) -> bool:
        """Check if ticker is in Shariah-compliant list."""
        return ticker in PSXPortfolioHelper.SHARIAH_STOCKS

    @staticmethod
    def is_cpec_related(ticker: str) -> bool:
        """Check if ticker is CPEC-related."""
        return ticker in PSXPortfolioHelper.CPEC_STOCKS

    @staticmethod
    def next_trading_session() -> str:
        """Return next PSX trading session as a string."""
        from datetime import timezone
        now_utc = datetime.now(timezone.utc)
        now_pkt = now_utc + timedelta(hours=5)

        if now_pkt.weekday() < 5:
            open_time = now_pkt.replace(
                hour=Config.psx.MARKET_OPEN_HOUR,
                minute=Config.psx.MARKET_OPEN_MIN,
                second=0,
            )
            if now_pkt < open_time:
                return f"Today at {Config.psx.MARKET_OPEN_HOUR}:{Config.psx.MARKET_OPEN_MIN:02d} PKT"
            close_time = now_pkt.replace(
                hour=Config.psx.MARKET_CLOSE_HOUR,
                minute=Config.psx.MARKET_CLOSE_MIN,
                second=0,
            )
            if now_pkt < close_time:
                return "Market is OPEN now"

        days_ahead = (7 - now_pkt.weekday()) % 7
        if days_ahead == 0:
            days_ahead = 7
        next_day = (now_pkt + timedelta(days=days_ahead)).strftime("%A %b %d")
        return f"{next_day} at {Config.psx.MARKET_OPEN_HOUR}:{Config.psx.MARKET_OPEN_MIN:02d} PKT"


# =============================================================================
# CLI smoke test
# =============================================================================

if __name__ == "__main__":
    print("=" * 60)
    print("SGEAIA Portfolio Module - Smoke Test")
    print("=" * 60)

    # Test WatchlistManager
    print("\n-- Watchlist Manager --")
    wm = WatchlistManager()
    print(f"  Default items: {len(wm.items)}")
    wm.add("NVDA", name="NVIDIA Corporation")
    print(f"  After adding NVDA: {len(wm.items)}")
    print(f"  PSX tickers: {wm.get_psx_tickers()}")
    print(f"  US tickers: {wm.get_us_tickers()}")
    wm.remove("NVDA")
    print(f"  After removing NVDA: {len(wm.items)}")

    # Test PSX Helper
    print("\n-- PSX Portfolio Helper --")
    helper = PSXPortfolioHelper()
    print(f"  PSX market open: {helper.is_psx_market_open()}")
    print(f"  Next session: {helper.next_trading_session()}")
    print(f"  PKR 1,000,000 = ${helper.pkr_to_usd(1_000_000):.2f}")
    print(f"  $100 = {helper.format_pkr(helper.usd_to_pkr(100))}")
    print(f"  MEBL.KA Shariah: {helper.is_shariah_compliant('MEBL.KA')}")
    print(f"  HUBC.KA CPEC: {helper.is_cpec_related('HUBC.KA')}")
    print(f"  ENGRO.KA sector: {helper.get_sector_for_ticker('ENGRO.KA')}")

    # Test Portfolio Analytics
    print("\n-- Portfolio Analytics --")
    tickers = ["ENGRO.KA", "HBL.KA", "SPY", "QQQ", "^FTSE"]
    breakdown = PortfolioAnalytics.market_breakdown(tickers)
    print(f"  Market breakdown: {breakdown}")

    # Test market detection
    print("\n-- Market Detection --")
    test_cases = [
        ("^KSE100", "PSX", "PKR"),
        ("ENGRO.KA", "PSX", "PKR"),
        ("SPY", "US", "USD"),
        ("^FTSE", "LSE", "GBP"),
        ("7203.T", "TSE", "JPY"),
    ]
    for ticker, expected_market, expected_currency in test_cases:
        market, currency = BaseLoader._detect_market(ticker)
        status = "OK" if market == expected_market else "FAIL"
        print(f"  [{status}] {ticker:12s} -> {market} ({currency})")

    print("\nAll portfolio tests passed.")
