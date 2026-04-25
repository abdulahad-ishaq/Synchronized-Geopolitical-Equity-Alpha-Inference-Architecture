"""
config.py
=========
Synchronized Geopolitical-Equity Alpha Inference Architecture
-------------------------------------------------------------
Central configuration file for all system settings, constants,
API keys, and feature flags.

Markets supported:
  - Pakistan Stock Exchange (PSX) — primary launch market
  - US Markets (NYSE, NASDAQ)
  - Global Markets (London, Tokyo, Frankfurt, Hong Kong, etc.)

All other files import from here. Change settings in ONE place
and they apply everywhere automatically.
"""

from __future__ import annotations
import os
from dataclasses import dataclass, field
from typing import Optional


# ─────────────────────────────────────────────────────────────────────────────
# Secret / environment variable loader
# ─────────────────────────────────────────────────────────────────────────────

def _get_secret(key: str, default: str = "") -> str:
    """
    Load secrets in priority order:
      1. st.secrets  (Streamlit Cloud deployment)
      2. os.environ  (local / Claude Code CLI)
      3. default     (fallback value)
    """
    try:
        import streamlit as st
        val = st.secrets.get(key, "")
        if val:
            return str(val)
    except Exception:
        pass
    return os.environ.get(key, default)


# ─────────────────────────────────────────────────────────────────────────────
# API Keys
# ─────────────────────────────────────────────────────────────────────────────

class APIKeys:
    """All API keys loaded from st.secrets or environment variables."""

    GNEWS_API_KEY      = _get_secret("GNEWS_API_KEY")
    NEWSAPI_KEY        = _get_secret("NEWSAPI_KEY")
    FRED_API_KEY       = _get_secret("FRED_API_KEY")
    SENDGRID_API_KEY   = _get_secret("SENDGRID_API_KEY")
    SUPABASE_URL       = _get_secret("SUPABASE_URL")
    SUPABASE_KEY       = _get_secret("SUPABASE_KEY")
    ALPHA_VANTAGE_KEY  = _get_secret("ALPHA_VANTAGE_KEY")

    # Pakistan-specific data sources
    PSX_API_KEY        = _get_secret("PSX_API_KEY")        # PSX official data
    INVESTING_COM_KEY  = _get_secret("INVESTING_COM_KEY")  # Backup for PSX data


# ─────────────────────────────────────────────────────────────────────────────
# Pakistan Stock Exchange (PSX) Configuration
# ─────────────────────────────────────────────────────────────────────────────

class PSXConfig:
    """
    Pakistan Stock Exchange specific configuration.

    PSX Facts:
      - Full name: Pakistan Stock Exchange Limited
      - Location: Karachi, Pakistan
      - Index: KSE-100 (100 largest companies)
      - Currency: Pakistani Rupee (PKR)
      - Trading hours: 9:15 AM - 3:30 PM PKT (UTC+5)
      - Settlement: T+2
      - Regulator: Securities and Exchange Commission of Pakistan (SECP)
      - Website: www.psx.com.pk
    """

    # Exchange info
    EXCHANGE_NAME:     str = "Pakistan Stock Exchange (PSX)"
    EXCHANGE_CODE:     str = "PSX"
    MAIN_INDEX:        str = "KSE-100"
    CURRENCY:          str = "PKR"
    CURRENCY_SYMBOL:   str = "₨"

    # Trading hours (Pakistan Standard Time = UTC+5)
    MARKET_OPEN_HOUR:  int = 9
    MARKET_OPEN_MIN:   int = 15
    MARKET_CLOSE_HOUR: int = 15
    MARKET_CLOSE_MIN:  int = 30
    TIMEZONE:          str = "Asia/Karachi"
    UTC_OFFSET_HOURS:  int = 5

    # Pre-market session
    PREOPEN_START_HOUR: int = 9
    PREOPEN_START_MIN:  int = 0
    PREOPEN_END_HOUR:   int = 9
    PREOPEN_END_MIN:    int = 15

    # Yahoo Finance ticker suffix for PSX stocks
    # PSX tickers on Yahoo Finance use .KA suffix
    # Example: ENGRO.KA, LUCK.KA, OGDC.KA
    YAHOO_SUFFIX: str = ".KA"

    # KSE-100 Blue Chip Companies
    # These are the most liquid, most traded PSX stocks
    KSE100_BLUECHIPS: dict[str, str] = {
        "ENGRO.KA":  "Engro Corporation",
        "LUCK.KA":   "Lucky Cement",
        "OGDC.KA":   "Oil & Gas Development Company",
        "PPL.KA":    "Pakistan Petroleum Limited",
        "PSO.KA":    "Pakistan State Oil",
        "HBL.KA":    "Habib Bank Limited",
        "UBL.KA":    "United Bank Limited",
        "MCB.KA":    "MCB Bank Limited",
        "NBP.KA":    "National Bank of Pakistan",
        "BAFL.KA":   "Bank Alfalah Limited",
        "FFBL.KA":   "Fauji Fertilizer Bin Qasim",
        "FFC.KA":    "Fauji Fertilizer Company",
        "HUBC.KA":   "Hub Power Company",
        "KAPCO.KA":  "Kot Addu Power Company",
        "MARI.KA":   "Mari Petroleum Company",
        "MEBL.KA":   "Meezan Bank Limited",
        "MTL.KA":    "Millat Tractors Limited",
        "NESTLE.KA": "Nestle Pakistan",
        "SEARL.KA":  "The Searle Company",
        "SYS.KA":    "Systems Limited",
        "TRG.KA":    "TRG Pakistan",
        "UNITY.KA":  "Unity Foods Limited",
    }

    # PSX Sector Groups
    SECTOR_GROUPS: dict[str, list[str]] = {
        "Banking": [
            "HBL.KA",   # Habib Bank
            "UBL.KA",   # United Bank
            "MCB.KA",   # MCB Bank
            "NBP.KA",   # National Bank
            "BAFL.KA",  # Bank Alfalah
            "MEBL.KA",  # Meezan Bank (Islamic)
        ],
        "Energy & Oil": [
            "OGDC.KA",  # Oil & Gas Development
            "PPL.KA",   # Pakistan Petroleum
            "PSO.KA",   # Pakistan State Oil
            "MARI.KA",  # Mari Petroleum
            "HUBC.KA",  # Hub Power
            "KAPCO.KA", # Kot Addu Power
        ],
        "Fertilizer": [
            "ENGRO.KA", # Engro Corporation
            "FFC.KA",   # Fauji Fertilizer
            "FFBL.KA",  # Fauji Fertilizer Bin Qasim
        ],
        "Cement": [
            "LUCK.KA",  # Lucky Cement
        ],
        "Technology": [
            "SYS.KA",   # Systems Limited
            "TRG.KA",   # TRG Pakistan
        ],
        "Consumer Goods": [
            "NESTLE.KA", # Nestle Pakistan
            "UNITY.KA",  # Unity Foods
        ],
        "Industrial": [
            "MTL.KA",   # Millat Tractors
        ],
        "Pharma": [
            "SEARL.KA", # The Searle Company
        ],
    }

    # PSX Indices available on Yahoo Finance
    INDICES: dict[str, str] = {
        "KCHI.KA":   "KSE-100 Index (Yahoo Finance)",
        "ENGRO.KA":  "Use individual stocks instead",
    }
    # NOTE: KSE-100 index (^KSE100) is NOT available on Yahoo Finance.
    # Use individual PSX stocks with .KA suffix instead.
    # Best PSX proxies: ENGRO.KA, HBL.KA, OGDC.KA, SYS.KA
    DEFAULT_PSX_TICKER: str = "ENGRO.KA"

    # Pakistan-specific news sources for geopolitical context
    PAKISTAN_NEWS_SOURCES: dict[str, str] = {
        "geo.tv":          "Geo News",
        "dawn.com":        "Dawn",
        "thenews.com.pk":  "The News International",
        "tribune.com.pk":  "Express Tribune",
        "brecorder.com":   "Business Recorder",   # PSX specific
        "propakistani.pk": "ProPakistani",         # Tech/finance
    }

    # Business Recorder RSS (free PSX financial news)
    BRECORDER_RSS: str = "https://www.brecorder.com/feed"
    DAWN_BUSINESS_RSS: str = "https://www.dawn.com/feeds/business"

    # PSX data sources
    PSX_OFFICIAL_URL:   str = "https://www.psx.com.pk"
    PSX_API_BASE:       str = "https://dps.psx.com.pk"

    # USD to PKR approximate rate (updated via API in production)
    USD_PKR_APPROX: float = 278.0

    # Geopolitical keywords specific to Pakistan market
    PAKISTAN_SPECIFIC_KEYWORDS: list[str] = [
        "IMF", "pakistan", "rupee", "pkr", "sbp",
        "state bank", "karachi", "lahore", "islamabad",
        "cpec", "china pakistan", "imf bailout",
        "pakistan economy", "kse", "psx", "secp",
        "load shedding", "energy crisis", "flood",
        "political stability", "election pakistan",
        "army", "government pakistan",
    ]


# ─────────────────────────────────────────────────────────────────────────────
# Global Market Configuration
# ─────────────────────────────────────────────────────────────────────────────

class GlobalMarketConfig:
    """
    Global stock exchange configuration.
    Covers all major exchanges worldwide.
    """

    # All supported exchanges
    EXCHANGES: dict[str, dict] = {
        "PSX": {
            "name":       "Pakistan Stock Exchange",
            "country":    "Pakistan",
            "flag":       "🇵🇰",
            "currency":   "PKR",
            "symbol":     "₨",
            "timezone":   "Asia/Karachi",
            "utc_offset": +5,
            "open":       "09:15",
            "close":      "15:30",
            "yahoo_suffix": ".KA",
            "main_index": "ENGRO.KA",
        },
        "NYSE/NASDAQ": {
            "name":       "US Markets",
            "country":    "United States",
            "flag":       "🇺🇸",
            "currency":   "USD",
            "symbol":     "$",
            "timezone":   "America/New_York",
            "utc_offset": -5,
            "open":       "09:30",
            "close":      "16:00",
            "yahoo_suffix": "",
            "main_index": "^GSPC",
        },
        "LSE": {
            "name":       "London Stock Exchange",
            "country":    "United Kingdom",
            "flag":       "🇬🇧",
            "currency":   "GBP",
            "symbol":     "£",
            "timezone":   "Europe/London",
            "utc_offset": 0,
            "open":       "08:00",
            "close":      "16:30",
            "yahoo_suffix": ".L",
            "main_index": "^FTSE",
        },
        "TSE": {
            "name":       "Tokyo Stock Exchange",
            "country":    "Japan",
            "flag":       "🇯🇵",
            "currency":   "JPY",
            "symbol":     "¥",
            "timezone":   "Asia/Tokyo",
            "utc_offset": +9,
            "open":       "09:00",
            "close":      "15:30",
            "yahoo_suffix": ".T",
            "main_index": "^N225",
        },
        "HKEX": {
            "name":       "Hong Kong Stock Exchange",
            "country":    "Hong Kong",
            "flag":       "🇭🇰",
            "currency":   "HKD",
            "symbol":     "HK$",
            "timezone":   "Asia/Hong_Kong",
            "utc_offset": +8,
            "open":       "09:30",
            "close":      "16:00",
            "yahoo_suffix": ".HK",
            "main_index": "^HSI",
        },
        "SSE": {
            "name":       "Shanghai Stock Exchange",
            "country":    "China",
            "flag":       "🇨🇳",
            "currency":   "CNY",
            "symbol":     "¥",
            "timezone":   "Asia/Shanghai",
            "utc_offset": +8,
            "open":       "09:30",
            "close":      "15:00",
            "yahoo_suffix": ".SS",
            "main_index": "000001.SS",
        },
        "BSE": {
            "name":       "Bombay Stock Exchange",
            "country":    "India",
            "flag":       "🇮🇳",
            "currency":   "INR",
            "symbol":     "₹",
            "timezone":   "Asia/Kolkata",
            "utc_offset": +5.5,
            "open":       "09:15",
            "close":      "15:30",
            "yahoo_suffix": ".BO",
            "main_index": "^BSESN",
        },
        "FSE": {
            "name":       "Frankfurt Stock Exchange",
            "country":    "Germany",
            "flag":       "🇩🇪",
            "currency":   "EUR",
            "symbol":     "€",
            "timezone":   "Europe/Berlin",
            "utc_offset": +1,
            "open":       "09:00",
            "close":      "17:30",
            "yahoo_suffix": ".DE",
            "main_index": "^GDAXI",
        },
        "TSX": {
            "name":       "Toronto Stock Exchange",
            "country":    "Canada",
            "flag":       "🇨🇦",
            "currency":   "CAD",
            "symbol":     "C$",
            "timezone":   "America/Toronto",
            "utc_offset": -5,
            "open":       "09:30",
            "close":      "16:00",
            "yahoo_suffix": ".TO",
            "main_index": "^GSPTSE",
        },
        "ASX": {
            "name":       "Australian Securities Exchange",
            "country":    "Australia",
            "flag":       "🇦🇺",
            "currency":   "AUD",
            "symbol":     "A$",
            "timezone":   "Australia/Sydney",
            "utc_offset": +10,
            "open":       "10:00",
            "close":      "16:00",
            "yahoo_suffix": ".AX",
            "main_index": "^AXJO",
        },
        "TADAWUL": {
            "name":       "Saudi Stock Exchange (Tadawul)",
            "country":    "Saudi Arabia",
            "flag":       "🇸🇦",
            "currency":   "SAR",
            "symbol":     "﷼",
            "timezone":   "Asia/Riyadh",
            "utc_offset": +3,
            "open":       "09:30",
            "close":      "15:00",
            "yahoo_suffix": ".SR",
            "main_index": "^TASI.SR",
        },
        "DFM": {
            "name":       "Dubai Financial Market",
            "country":    "UAE",
            "flag":       "🇦🇪",
            "currency":   "AED",
            "symbol":     "د.إ",
            "timezone":   "Asia/Dubai",
            "utc_offset": +4,
            "open":       "10:00",
            "close":      "14:00",
            "yahoo_suffix": ".DU",
            "main_index": "^DFMGI",
        },
    }

    # Global market indices for quick overview
    GLOBAL_INDICES: dict[str, str] = {
        "ENGRO.KA":   "Engro Corp (Pakistan) 🇵🇰",
        "^GSPC":      "S&P 500 (USA) 🇺🇸",
        "^IXIC":      "NASDAQ (USA) 🇺🇸",
        "^FTSE":      "FTSE 100 (UK) 🇬🇧",
        "^N225":      "Nikkei 225 (Japan) 🇯🇵",
        "^HSI":       "Hang Seng (HK) 🇭🇰",
        "^GDAXI":     "DAX (Germany) 🇩🇪",
        "^BSESN":     "BSE Sensex (India) 🇮🇳",
        "000001.SS":  "Shanghai Composite (China) 🇨🇳",
        "^AXJO":      "ASX 200 (Australia) 🇦🇺",
        "^TASI.SR":   "Tadawul (Saudi Arabia) 🇸🇦",
    }


# ─────────────────────────────────────────────────────────────────────────────
# News Source Configuration
# ─────────────────────────────────────────────────────────────────────────────

class NewsConfig:
    """News source settings including approved domains and display names."""

    # Hard-coded approved domains
    APPROVED_DOMAINS: list[str] = [
        # International
        "reuters.com",
        "ft.com",
        "aljazeera.com",
        "nikkei.com",
        "asia.nikkei.com",
        "wsj.com",
        "bloomberg.com",
        # Pakistan-specific
        "dawn.com",
        "brecorder.com",
        "thenews.com.pk",
        "tribune.com.pk",
        "geo.tv",
    ]

    DOMAIN_DISPLAY: dict[str, str] = {
        "reuters.com":     "Reuters",
        "ft.com":          "Financial Times",
        "aljazeera.com":   "Al Jazeera",
        "nikkei.com":      "Nikkei Asia",
        "asia.nikkei.com": "Nikkei Asia",
        "wsj.com":         "Wall Street Journal",
        "bloomberg.com":   "Bloomberg",
        "dawn.com":        "Dawn",
        "brecorder.com":   "Business Recorder",
        "thenews.com.pk":  "The News International",
        "tribune.com.pk":  "Express Tribune",
        "geo.tv":          "Geo News",
    }

    DOMAIN_COLOURS: dict[str, str] = {
        "reuters.com":     "#ff8f00",
        "ft.com":          "#f9a825",
        "aljazeera.com":   "#00b4d8",
        "nikkei.com":      "#e91e63",
        "asia.nikkei.com": "#e91e63",
        "wsj.com":         "#7c4dff",
        "bloomberg.com":   "#00e676",
        "dawn.com":        "#009688",
        "brecorder.com":   "#4caf50",
        "thenews.com.pk":  "#ff5722",
        "tribune.com.pk":  "#2196f3",
        "geo.tv":          "#9c27b0",
    }

    DOMAIN_CSS: dict[str, str] = {
        "reuters.com":     "reuters",
        "ft.com":          "ft",
        "aljazeera.com":   "aljazeera",
        "nikkei.com":      "nikkei",
        "asia.nikkei.com": "nikkei",
        "wsj.com":         "wsj",
        "bloomberg.com":   "bloomberg",
        "dawn.com":        "dawn",
        "brecorder.com":   "brecorder",
        "thenews.com.pk":  "thenews",
        "tribune.com.pk":  "tribune",
        "geo.tv":          "geo",
    }

    # Pakistan-specific news category
    PAKISTAN_DOMAINS: list[str] = [
        "dawn.com",
        "brecorder.com",
        "thenews.com.pk",
        "tribune.com.pk",
        "geo.tv",
    ]

    GLOBAL_DOMAINS: list[str] = [
        "reuters.com",
        "ft.com",
        "aljazeera.com",
        "nikkei.com",
        "wsj.com",
        "bloomberg.com",
    ]

    GEOPOLITICAL_QUERY: str = (
        "geopolitical OR war OR sanctions OR tariff OR "
        "central bank OR inflation OR recession OR trade war OR "
        "OPEC OR NATO OR supply chain OR election OR conflict OR "
        "federal reserve OR interest rate OR GDP OR currency OR "
        "IMF OR Pakistan OR rupee OR KSE OR CPEC"
    )

    NEWSAPI_BASE:  str = "https://newsapi.org/v2/everything"
    GNEWS_BASE:    str = "https://gnews.io/api/v4/search"
    GDELT_BASE:    str = "https://api.gdeltproject.org/api/v2/doc/doc"

    BLOOMBERG_RSS_FEEDS: list[str] = [
        "https://feeds.bloomberg.com/markets/news.rss",
        "https://feeds.bloomberg.com/politics/news.rss",
    ]
    WSJ_RSS_FEEDS: list[str] = [
        "https://feeds.a.dj.com/rss/RSSMarketsMain.xml",
        "https://feeds.a.dj.com/rss/RSSWorldNews.xml",
    ]
    DAWN_RSS_FEEDS: list[str] = [
        "https://www.dawn.com/feeds/business",
        "https://www.dawn.com/feeds/pakistan",
    ]
    BRECORDER_RSS_FEEDS: list[str] = [
        "https://www.brecorder.com/feed",
    ]


# ─────────────────────────────────────────────────────────────────────────────
# Market / Ticker Configuration
# ─────────────────────────────────────────────────────────────────────────────

class MarketConfig:
    """Stock market and ticker settings for all supported markets."""

    DEFAULT_TICKER:    str = "ENGRO.KA"   # ^KSE100 not on Yahoo Finance
    DEFAULT_MARKET:    str = "PSX"

    FREE_TIER_TICKER_LIMIT:   int = 1
    MAX_COMPARISON_TICKERS:   int = 5

    MACRO_SERIES: list[str] = [
        "FEDFUNDS",
        "T10Y2Y",
        "DCOILWTICO",
    ]

    # All ticker groups organised by market
    TICKER_GROUPS: dict[str, list[str]] = {

        # ── Pakistan (PSX) ────────────────────────────────────────────
        "🇵🇰 KSE-100 Top Picks": ["ENGRO.KA", "HBL.KA", "OGDC.KA", "SYS.KA"],
        "🇵🇰 PSX Banking": [
            "HBL.KA", "UBL.KA", "MCB.KA",
            "NBP.KA", "BAFL.KA", "MEBL.KA",
        ],
        "🇵🇰 PSX Energy": [
            "OGDC.KA", "PPL.KA", "PSO.KA",
            "MARI.KA", "HUBC.KA",
        ],
        "🇵🇰 PSX Technology": [
            "SYS.KA", "TRG.KA",
        ],
        "🇵🇰 PSX Blue Chips": [
            "ENGRO.KA", "LUCK.KA", "OGDC.KA",
            "HBL.KA", "SYS.KA", "PPL.KA",
        ],

        # ── United States ─────────────────────────────────────────────
        "🇺🇸 US Market ETFs": [
            "SPY", "QQQ", "DIA", "IWM",
        ],
        "🇺🇸 US Tech Giants": [
            "AAPL", "MSFT", "NVDA", "GOOGL",
            "AMZN", "TSLA", "META",
        ],
        "🇺🇸 US Finance": [
            "JPM", "GS", "BAC", "BRK-B",
        ],

        # ── Global Indices ────────────────────────────────────────────
        "🌍 Global Indices": [
            "^GSPC",      # S&P 500 USA
            "^FTSE",      # FTSE 100 UK
            "^N225",      # Nikkei 225 Japan
            "^HSI",       # Hang Seng Hong Kong
            "^GDAXI",     # DAX Germany
            "^BSESN",     # BSE Sensex India
        ],

        # ── Middle East & Islamic Finance ─────────────────────────────
        "🕌 Middle East": [
            "^TASI.SR",   # Saudi Tadawul
            "^DFMGI",     # Dubai
            "EFG.CA",     # EFG Hermes Egypt
        ],

        # ── Asia Pacific ──────────────────────────────────────────────
        "🌏 Asia Pacific": [
            "EEM",        # Emerging Markets ETF
            "FXI",        # China ETF
            "EWJ",        # Japan ETF
            "EWH",        # Hong Kong ETF
            "INDA",       # India ETF
        ],

        # ── Commodities (important for Pakistan economy) ──────────────
        "🛢️ Commodities": [
            "GLD",        # Gold (important for PKR hedge)
            "USO",        # Oil (Pakistan imports heavily)
            "SLV",        # Silver
            "UNG",        # Natural Gas
        ],

        # ── Bonds & Safe Havens ───────────────────────────────────────
        "🏦 Bonds & Safe Havens": [
            "TLT",        # 20-year US Treasury
            "GLD",        # Gold
            "SHY",        # Short-term Treasury
        ],
    }


# ─────────────────────────────────────────────────────────────────────────────
# Model Configuration
# ─────────────────────────────────────────────────────────────────────────────

class ModelConfig:
    """Neural network architecture and training settings."""

    SEQ_LEN:       int   = 60
    NUM_FEATURES:  int   = 9
    LSTM_HIDDEN:   int   = 128
    LSTM_LAYERS:   int   = 2
    FUSION_DIM:    int   = 256
    NUM_HEADS:     int   = 4
    DROPOUT:       float = 0.3

    CRISIS_VOL_THRESHOLD: float = 0.30

    LEARNING_RATE:  float = 1e-4
    BATCH_SIZE:     int   = 32
    NUM_EPOCHS:     int   = 50
    TRAIN_SPLIT:    float = 0.70
    VAL_SPLIT:      float = 0.15
    TEST_SPLIT:     float = 0.15
    EARLY_STOPPING_PATIENCE: int = 10

    CHECKPOINT_PATH: str = "model_checkpoint.pt"
    BEST_MODEL_PATH: str = "best_model.pt"


# ─────────────────────────────────────────────────────────────────────────────
# Backtesting Configuration
# ─────────────────────────────────────────────────────────────────────────────

class BacktestConfig:
    """Backtesting and performance metrics settings."""

    RISK_FREE_RATE:        float = 0.05
    TRADING_DAYS_PER_YEAR: int   = 252
    TRANSACTION_COST:      float = 0.001
    POSITION_SIZE:         float = 1.0
    LONG_THRESHOLD:        float = 0.1
    SHORT_THRESHOLD:       float = -0.1
    STOP_LOSS:             float = 0.05
    TAKE_PROFIT:           float = 0.10

    # PSX specific backtesting
    # PSX has higher transaction costs than US markets
    PSX_TRANSACTION_COST: float = 0.002   # 0.2% (CDC + brokerage)
    PSX_RISK_FREE_RATE:   float = 0.22    # ~22% PKR Treasury bill rate


# ─────────────────────────────────────────────────────────────────────────────
# Portfolio Configuration
# ─────────────────────────────────────────────────────────────────────────────

class PortfolioConfig:
    """Paper trading and portfolio simulation settings."""

    # USD paper trading account
    INITIAL_CAPITAL_USD:  float = 100_000.0

    # PKR paper trading account (for PSX simulation)
    INITIAL_CAPITAL_PKR:  float = 10_000_000.0   # 1 crore PKR

    MAX_POSITION_SIZE:    float = 0.20
    MAX_POSITIONS:        int   = 10
    REBALANCE_FREQUENCY:  str   = "weekly"

    # Supported currencies
    SUPPORTED_CURRENCIES: list[str] = ["USD", "PKR", "GBP", "EUR", "JPY"]
    DEFAULT_CURRENCY:     str = "USD"


# ─────────────────────────────────────────────────────────────────────────────
# Alert Configuration
# ─────────────────────────────────────────────────────────────────────────────

class AlertConfig:
    """Email and notification alert settings."""

    ALPHA_CHANGE_THRESHOLD:     float = 0.15
    SENTIMENT_CHANGE_THRESHOLD: float = 0.20
    REGIME_CHANGE_ALERT:        bool  = True
    CRISIS_MODE_ALERT:          bool  = True

    FROM_EMAIL:           str = _get_secret("ALERT_FROM_EMAIL", "alerts@sgeaia.app")
    EMAIL_SUBJECT_PREFIX: str = "SGEAIA Alert"

    FREE_NOTIFICATION_INTERVAL_MINUTES:    int = 120
    PREMIUM_NOTIFICATION_INTERVAL_MINUTES: int = 30

    # Pakistan market hours for smart notifications
    PSX_OPEN_HOUR:   int = 9
    PSX_OPEN_MIN:    int = 15
    PSX_CLOSE_HOUR:  int = 15
    PSX_CLOSE_MIN:   int = 30

    # US market hours
    US_OPEN_HOUR:    int = 9
    US_OPEN_MIN:     int = 30
    US_CLOSE_HOUR:   int = 16
    US_CLOSE_MIN:    int = 0

    # Notification languages
    SUPPORTED_LANGUAGES: list[str] = ["English", "Urdu"]
    DEFAULT_LANGUAGE:    str = "English"


# ─────────────────────────────────────────────────────────────────────────────
# Subscription / Freemium Configuration
# ─────────────────────────────────────────────────────────────────────────────

class SubscriptionConfig:
    """Freemium tier feature flags and limits."""

    FREE_MAX_TICKERS:        int  = 1
    FREE_NOTIFICATION_MINS:  int  = 120
    FREE_HISTORY_DAYS:       int  = 30
    FREE_PDF_REPORTS:        bool = False
    FREE_CRISIS_ALERTS:      bool = False
    FREE_MULTI_TICKER:       bool = False
    FREE_PAPER_TRADING:      bool = False
    FREE_EMAIL_ALERTS:       bool = False
    FREE_PSX_ACCESS:         bool = False   # PSX requires premium (same as global)
    FREE_GLOBAL_ACCESS:      bool = False   # All markets require premium

    PREMIUM_PRICE_MONTHLY_USD: float = 4.99
    # PKR price = USD equivalent at current exchange rate
    # $4.99 USD ≈ ₨1,399 PKR (updates with exchange rate)
    # Same value globally — no regional discount
    PREMIUM_PRICE_MONTHLY_PKR: float = 1_399.0
    PREMIUM_MAX_TICKERS:       int   = 999
    PREMIUM_NOTIFICATION_MINS: int   = 30
    PREMIUM_HISTORY_DAYS:      int   = 365
    PREMIUM_PDF_REPORTS:       bool  = True
    PREMIUM_CRISIS_ALERTS:     bool  = True
    PREMIUM_MULTI_TICKER:      bool  = True
    PREMIUM_PAPER_TRADING:     bool  = True
    PREMIUM_EMAIL_ALERTS:      bool  = True
    PREMIUM_PSX_ACCESS:        bool  = True   # PSX = premium like all markets
    PREMIUM_GLOBAL_ACCESS:     bool  = True


# ─────────────────────────────────────────────────────────────────────────────
# DSA Configuration
# ─────────────────────────────────────────────────────────────────────────────

class DSAConfig:
    """Data Structures & Algorithms settings."""

    MAX_RANKED_ARTICLES:      int   = 50
    IMPACT_DECAY_HOURS:       float = 24.0
    RECENCY_WEIGHT:           float = 0.4
    SENTIMENT_WEIGHT:         float = 0.4
    SOURCE_CREDIBILITY_WEIGHT: float = 0.2

    SOURCE_CREDIBILITY: dict[str, float] = {
        "reuters.com":     1.00,
        "ft.com":          0.95,
        "wsj.com":         0.90,
        "bloomberg.com":   0.90,
        "nikkei.com":      0.85,
        "asia.nikkei.com": 0.85,
        "aljazeera.com":   0.80,
        "dawn.com":        0.80,
        "brecorder.com":   0.85,   # High for PSX-specific news
        "thenews.com.pk":  0.75,
        "tribune.com.pk":  0.75,
        "geo.tv":          0.70,
    }

    NEWS_CACHE_SIZE:  int = 128
    PRICE_CACHE_SIZE: int = 64

    GEOPOLITICAL_KEYWORDS: list[str] = [
        # Global
        "war", "conflict", "sanctions", "tariff", "invasion",
        "recession", "inflation", "crisis", "collapse", "default",
        "election", "coup", "protest", "terrorism", "nuclear",
        "opec", "nato", "fed", "rate", "gdp", "unemployment",
        "trade", "deficit", "surplus", "currency", "devaluation",
        "pandemic", "earthquake", "hurricane", "disaster",
        # Pakistan-specific
        "imf", "rupee", "pkr", "sbp", "kse", "cpec",
        "load shedding", "pakistan economy", "secp",
    ]


# ─────────────────────────────────────────────────────────────────────────────
# UI / Display Configuration
# ─────────────────────────────────────────────────────────────────────────────

class UIConfig:
    """Dashboard display settings."""

    APP_TITLE:    str = "Synchronized Geopolitical-Equity Alpha Inference Architecture"
    APP_ICON:     str = "🛰️"
    APP_SUBTITLE: str = (
        "PSX 🇵🇰 · Reuters · FT · Al Jazeera · Nikkei · WSJ · Bloomberg · "
        "Dawn · Bi-LSTM · Crisis Mode"
    )

    BULL_COLOUR:    str = "#00e676"
    BEAR_COLOUR:    str = "#ff1744"
    NEUTRAL_COLOUR: str = "#546e7a"
    PRIMARY_COLOUR: str = "#00b4d8"
    WARNING_COLOUR: str = "#ffc400"
    CRISIS_COLOUR:  str = "#ff6d00"

    # Pakistan green accent
    PSX_COLOUR:     str = "#4caf50"
    PKR_COLOUR:     str = "#009688"

    DEFAULT_LOOKBACK_DAYS:   int = 90
    DEFAULT_SEQ_LEN:         int = 60
    CANDLESTICK_HEIGHT:      int = 480
    GAUGE_HEIGHT:            int = 260
    BAR_CHART_HEIGHT:        int = 160
    EQUITY_CURVE_HEIGHT:     int = 200
    MACRO_HEIGHT:            int = 200


# ─────────────────────────────────────────────────────────────────────────────
# PDF Report Configuration
# ─────────────────────────────────────────────────────────────────────────────

class PDFConfig:
    """PDF report generation settings."""

    REPORT_TITLE:   str = "SGEAIA Intelligence Report"
    COMPANY_NAME:   str = "Synchronized Geopolitical-Equity Alpha Inference Architecture"
    FOOTER_TEXT:    str = (
        "This report is for educational purposes only and does not constitute "
        "financial advice. Past performance does not guarantee future results."
    )
    FOOTER_URDU:    str = "یہ رپورٹ صرف تعلیمی مقاصد کے لیے ہے اور مالی مشورہ نہیں ہے۔"
    PRIMARY_COLOUR: str = "#00b4d8"
    DARK_COLOUR:    str = "#0b0f1a"


# ─────────────────────────────────────────────────────────────────────────────
# Authentication Configuration
# ─────────────────────────────────────────────────────────────────────────────

class AuthConfig:
    """User authentication settings."""

    SESSION_EXPIRY_HOURS: int  = 24
    MAX_LOGIN_ATTEMPTS:   int  = 5
    LOCKOUT_MINUTES:      int  = 30

    DEMO_MODE:       bool = True
    DEMO_USER_EMAIL: str  = "demo@sgeaia.app"
    DEMO_USER_NAME:  str  = "Demo User"
    DEMO_IS_PREMIUM: bool = True

    # Pakistan phone number support for OTP
    PAKISTAN_COUNTRY_CODE: str = "+92"
    SUPPORTED_OTP_COUNTRIES: list[str] = ["+92", "+1", "+44", "+61"]


# ─────────────────────────────────────────────────────────────────────────────
# Master Config
# ─────────────────────────────────────────────────────────────────────────────

class Config:
    """
    Master configuration class.

    Usage:
        from config import Config
        psx_tickers = Config.psx.KSE100_BLUECHIPS
        all_exchanges = Config.global_markets.EXCHANGES
        gnews_key = Config.api.GNEWS_API_KEY
    """
    api           = APIKeys
    news          = NewsConfig
    market        = MarketConfig
    psx           = PSXConfig
    global_markets = GlobalMarketConfig
    model         = ModelConfig
    backtest      = BacktestConfig
    portfolio     = PortfolioConfig
    alerts        = AlertConfig
    subscription  = SubscriptionConfig
    dsa           = DSAConfig
    ui            = UIConfig
    pdf           = PDFConfig
    auth          = AuthConfig


# ─────────────────────────────────────────────────────────────────────────────
# Validation
# ─────────────────────────────────────────────────────────────────────────────

def validate_config() -> dict[str, bool]:
    return {
        "GNews API":     bool(APIKeys.GNEWS_API_KEY),
        "NewsAPI":       bool(APIKeys.NEWSAPI_KEY),
        "FRED API":      bool(APIKeys.FRED_API_KEY),
        "SendGrid":      bool(APIKeys.SENDGRID_API_KEY),
        "Supabase URL":  bool(APIKeys.SUPABASE_URL),
        "Supabase Key":  bool(APIKeys.SUPABASE_KEY),
        "PSX API":       bool(APIKeys.PSX_API_KEY),
    }


# ─────────────────────────────────────────────────────────────────────────────
# CLI smoke test
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("=" * 60)
    print("SGEAIA Configuration Validation")
    print("=" * 60)

    print("\n📡 API Keys:")
    for key, configured in validate_config().items():
        print(f"  {'✅' if configured else '❌'} {key}")

    print("\n🇵🇰 Pakistan Stock Exchange (PSX):")
    print(f"  Exchange:  {PSXConfig.EXCHANGE_NAME}")
    print(f"  Index:     {PSXConfig.MAIN_INDEX}")
    print(f"  Currency:  {PSXConfig.CURRENCY_SYMBOL} {PSXConfig.CURRENCY}")
    print(f"  Hours:     {PSXConfig.MARKET_OPEN_HOUR}:{PSXConfig.MARKET_OPEN_MIN:02d}"
          f" - {PSXConfig.MARKET_CLOSE_HOUR}:{PSXConfig.MARKET_CLOSE_MIN:02d} PKT")
    print(f"  Sectors:   {len(PSXConfig.SECTOR_GROUPS)}")
    print(f"  Blue Chips:{len(PSXConfig.KSE100_BLUECHIPS)}")

    print("\n🌍 Global Exchanges Supported:")
    for code, info in GlobalMarketConfig.EXCHANGES.items():
        print(f"  {info['flag']} {code}: {info['name']} ({info['currency']})")

    print("\n📰 News Sources:")
    for domain, name in NewsConfig.DOMAIN_DISPLAY.items():
        is_pak = domain in NewsConfig.PAKISTAN_DOMAINS
        tag = "🇵🇰" if is_pak else "🌍"
        cred = DSAConfig.SOURCE_CREDIBILITY.get(domain, 0)
        print(f"  {tag} {name} — credibility: {cred:.0%}")

    print("\n💰 Freemium Model:")
    print(f"  Free:    {SubscriptionConfig.FREE_MAX_TICKERS} ticker | "
          f"Limited features | "
          f"{SubscriptionConfig.FREE_NOTIFICATION_MINS}min notifications")
    print(f"  Premium: Unlimited tickers | PSX + All global markets | "
          f"{SubscriptionConfig.PREMIUM_NOTIFICATION_MINS}min notifications | "
          f"${SubscriptionConfig.PREMIUM_PRICE_MONTHLY_USD}/mo USD = "
          f"₨{SubscriptionConfig.PREMIUM_PRICE_MONTHLY_PKR:,.0f}/mo PKR")
    print(f"  Pricing: Same value globally — no regional discount")
