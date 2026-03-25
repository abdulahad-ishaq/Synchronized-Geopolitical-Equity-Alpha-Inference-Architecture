"""
data_loader.py
==============
Synchronized Geopolitical-Equity Alpha Inference Architecture
--------------------------------------------------------------
OOP-based data pipeline with:
  - Hard-coded domain filtering: reuters.com, ft.com, aljazeera.com
  - st.secrets API key management (os.environ fallback for CLI/Claude Code)
  - 60-day OHLCV ingestion via yfinance
  - Sliding-window temporal alignment with strict zero-lookahead bias
  - Macro indicator ingestion via FRED (pandas_datareader)
"""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Optional

import numpy as np
import pandas as pd
import requests
import yfinance as yf

logger = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s – %(message)s",
)

# ─────────────────────────────────────────────────────────────────────────────
# Constants
# ─────────────────────────────────────────────────────────────────────────────

APPROVED_DOMAINS: list[str] = ["reuters.com", "ft.com", "aljazeera.com"]
DOMAIN_DISPLAY: dict[str, str] = {
    "reuters.com":   "Reuters",
    "ft.com":        "Financial Times",
    "aljazeera.com": "Al Jazeera",
}
GEOPOLITICAL_QUERY = (
    "geopolitical OR war OR sanctions OR tariff OR "
    "central bank OR inflation OR recession OR trade war OR "
    "OPEC OR NATO OR supply chain OR election OR conflict"
)
LOOKBACK_DAYS = 60          # hard spec: 60-day Bi-LSTM window
MACRO_SERIES  = ["FEDFUNDS", "T10Y2Y", "DCOILWTICO"]
NEWSAPI_BASE  = "https://newsapi.org/v2/everything"


# ─────────────────────────────────────────────────────────────────────────────
# Secrets helper (st.secrets → os.environ fallback)
# ─────────────────────────────────────────────────────────────────────────────

def _get_secret(key: str) -> str:
    """
    Priority:
      1. st.secrets  (Streamlit Cloud)
      2. os.environ  (local / Claude Code CLI)
    """
    try:
        import streamlit as st
        val = st.secrets.get(key, "")
        if val:
            return val
    except Exception:
        pass
    import os
    return os.environ.get(key, "")


# ─────────────────────────────────────────────────────────────────────────────
# Data container dataclasses
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class OHLCVBundle:
    """Container for one ticker's OHLCV data and engineered features."""
    ticker:        str
    raw:           pd.DataFrame       # original yfinance columns
    features:      pd.DataFrame       # 9-col engineered matrix for Bi-LSTM
    aligned_index: pd.DatetimeIndex

    @property
    def close(self) -> pd.Series:
        return self.raw["Close"]

    @property
    def num_features(self) -> int:
        return self.features.shape[1]


@dataclass
class NewsArticle:
    """Single news article with domain attribution."""
    published_at:  pd.Timestamp
    title:         str
    description:   str
    url:           str
    source_domain: str

    @property
    def source_label(self) -> str:
        return DOMAIN_DISPLAY.get(self.source_domain, self.source_domain)

    @property
    def full_text(self) -> str:
        return " ".join(
            p for p in [self.title or "", self.description or ""] if p
        ).strip()[:512]


@dataclass
class AlignedBundle:
    """
    Final time-aligned output bundle.
    All DataFrames share the same UTC business-day DatetimeIndex.
    """
    ticker:        str
    ohlcv:         OHLCVBundle
    news_daily:    pd.DataFrame     # headline_count, headline_concat, per-domain counts
    macro:         pd.DataFrame     # FRED macro indicators
    aligned_index: pd.DatetimeIndex
    start:         str
    end:           str
    articles:      list[NewsArticle] = field(default_factory=list)


# ─────────────────────────────────────────────────────────────────────────────
# Abstract base loader
# ─────────────────────────────────────────────────────────────────────────────

class BaseLoader(ABC):
    """Shared interface and utilities for all loader classes."""

    @abstractmethod
    def fetch(self, start: str, end: str) -> object:
        ...

    @staticmethod
    def _bday_index(start: str, end: str) -> pd.DatetimeIndex:
        return pd.bdate_range(start=start, end=end, freq="B", tz="UTC")

    @staticmethod
    def _align_to_index(df: pd.DataFrame, idx: pd.DatetimeIndex) -> pd.DataFrame:
        """
        Reindex with FORWARD-FILL ONLY.
        Back-fill is prohibited: it would introduce future information and
        violate the zero-lookahead-bias requirement.
        """
        return df.reindex(idx, method="ffill")


# ─────────────────────────────────────────────────────────────────────────────
# 1. Equity Loader
# ─────────────────────────────────────────────────────────────────────────────

class EquityLoader(BaseLoader):
    """
    Downloads 60-day OHLCV data for a single ticker and builds the
    9-feature engineering matrix consumed by the Bi-LSTM encoder.

    Features:
        Close_norm, Return_1d, Return_5d, Return_10d,
        Volatility_10d, RSI_14, Volume_norm,
        BB_upper_dist, BB_lower_dist
    """

    FEATURE_COLS: list[str] = [
        "Close_norm", "Return_1d", "Return_5d", "Return_10d",
        "Volatility_10d", "RSI_14", "Volume_norm",
        "BB_upper_dist", "BB_lower_dist",
    ]

    def __init__(self, ticker: str = "SPY"):
        self.ticker = ticker.upper()

    def fetch(self, start: str, end: str) -> OHLCVBundle:
        logger.info("EquityLoader › %s  %s → %s", self.ticker, start, end)
        raw  = self._download(start, end)
        idx  = self._bday_index(start, end)
        raw  = self._align_to_index(raw, idx)
        feat = self._engineer(raw)
        feat = self._align_to_index(feat, idx)
        return OHLCVBundle(ticker=self.ticker, raw=raw, features=feat, aligned_index=idx)

    def _download(self, start: str, end: str) -> pd.DataFrame:
        df = yf.download(self.ticker, start=start, end=end,
                         auto_adjust=True, progress=False)
        if df.empty:
            raise ValueError(f"yfinance returned no data for {self.ticker}.")
        df.index = pd.to_datetime(df.index, utc=True)
        return df

    def _engineer(self, df: pd.DataFrame) -> pd.DataFrame:
        out   = pd.DataFrame(index=df.index)
        close = df["Close"].squeeze()
        vol   = df["Volume"].squeeze() if "Volume" in df.columns else pd.Series(1.0, index=df.index)

        out["Close_norm"]     = (close - close.min()) / (close.max() - close.min() + 1e-9)
        out["Return_1d"]      = close.pct_change(1)
        out["Return_5d"]      = close.pct_change(5)
        out["Return_10d"]     = close.pct_change(10)
        out["Volatility_10d"] = out["Return_1d"].rolling(10).std() * np.sqrt(252)
        out["RSI_14"]         = self._rsi(close, 14) / 100.0
        out["Volume_norm"]    = (vol - vol.min()) / (vol.max() - vol.min() + 1e-9)

        ma20  = close.rolling(20).mean()
        std20 = close.rolling(20).std()
        out["BB_upper_dist"]  = (close - (ma20 + 2 * std20)) / (close + 1e-9)
        out["BB_lower_dist"]  = (close - (ma20 - 2 * std20)) / (close + 1e-9)

        return out.fillna(0.0)

    @staticmethod
    def _rsi(series: pd.Series, period: int = 14) -> pd.Series:
        delta = series.diff()
        gain  = delta.clip(lower=0).rolling(period).mean()
        loss  = (-delta.clip(upper=0)).rolling(period).mean()
        rs    = gain / loss.replace(0, np.nan)
        return (100 - 100 / (1 + rs)).fillna(50.0)


# ─────────────────────────────────────────────────────────────────────────────
# 2. Geopolitical / News Loader — domain-locked
# ─────────────────────────────────────────────────────────────────────────────

class GeopoliticalLoader(BaseLoader):
    """
    Fetches headlines EXCLUSIVELY from reuters.com, ft.com, aljazeera.com
    via the NewsAPI `domains` parameter.

    Produces:
        daily_df  – per-business-day aggregation
        articles  – raw list[NewsArticle] for the Intelligence Feed UI
    """

    def __init__(self, api_key: Optional[str] = None):
        self.api_key = api_key or _get_secret("NEWSAPI_KEY")
        if not self.api_key:
            logger.warning("NEWSAPI_KEY missing – news feed will be empty.")

    def fetch(self, start: str, end: str) -> tuple[pd.DataFrame, list[NewsArticle]]:
        idx      = self._bday_index(start, end)
        articles = self._fetch_articles(start, end)
        daily    = self._aggregate(articles, idx)
        return daily, articles

    def _fetch_articles(self, start: str, end: str) -> list[NewsArticle]:
        if not self.api_key:
            return []
        params = {
            "q":        GEOPOLITICAL_QUERY,
            "domains":  ",".join(APPROVED_DOMAINS),   # ← HARD-CODED domain lock
            "from":     start,
            "to":       end,
            "language": "en",
            "sortBy":   "publishedAt",
            "pageSize": 100,
            "apiKey":   self.api_key,
        }
        out: list[NewsArticle] = []
        try:
            resp = requests.get(NEWSAPI_BASE, params=params, timeout=20)
            resp.raise_for_status()
            for item in resp.json().get("articles", []):
                art = self._parse(item)
                if art:
                    out.append(art)
            logger.info("GeopoliticalLoader › %d articles from approved domains.", len(out))
        except Exception as exc:
            logger.error("NewsAPI error: %s", exc)
        return out

    @staticmethod
    def _parse(item: dict) -> Optional[NewsArticle]:
        try:
            ts = pd.to_datetime(item.get("publishedAt", ""), utc=True)
        except Exception:
            return None
        url    = item.get("url", "")
        domain = next((d for d in APPROVED_DOMAINS if d in url), "unknown")
        return NewsArticle(
            published_at=ts,
            title=item.get("title", "") or "",
            description=item.get("description", "") or "",
            url=url,
            source_domain=domain,
        )

    def _aggregate(self, articles: list[NewsArticle], idx: pd.DatetimeIndex) -> pd.DataFrame:
        """
        Zero-lookahead: each day d only sees articles published on day d.
        No forward-filling of text – missing days stay empty string.
        """
        cols = ["headline_count", "headline_concat",
                "reuters_count", "ft_count", "aljazeera_count"]
        daily = pd.DataFrame(0, index=idx, columns=cols, dtype=object)
        daily["headline_concat"] = ""

        if not articles:
            return daily

        for art in articles:
            day = art.published_at.normalize().tz_convert("UTC")
            if day not in idx:
                continue
            daily.loc[day, "headline_count"]  = int(daily.loc[day, "headline_count"]) + 1
            daily.loc[day, "headline_concat"] = (
                str(daily.loc[day, "headline_concat"]) + " | " + art.full_text
            ).strip(" |")
            col_map = {
                "reuters.com":   "reuters_count",
                "ft.com":        "ft_count",
                "aljazeera.com": "aljazeera_count",
            }
            if art.source_domain in col_map:
                daily.loc[day, col_map[art.source_domain]] = (
                    int(daily.loc[day, col_map[art.source_domain]]) + 1
                )

        for c in ["headline_count", "reuters_count", "ft_count", "aljazeera_count"]:
            daily[c] = pd.to_numeric(daily[c], errors="coerce").fillna(0).astype(int)
        return daily


# ─────────────────────────────────────────────────────────────────────────────
# 3. Macro Loader (FRED)
# ─────────────────────────────────────────────────────────────────────────────

class MacroLoader(BaseLoader):
    """
    Pulls FRED macro series. Uses forward-fill only; zeros on failure.
    """

    def __init__(self, series_ids: list[str] = MACRO_SERIES,
                 api_key: Optional[str] = None):
        self.series_ids = series_ids
        self.api_key    = api_key or _get_secret("FRED_API_KEY")

    def fetch(self, start: str, end: str) -> pd.DataFrame:
        idx = self._bday_index(start, end)
        try:
            import pandas_datareader.data as web  # type: ignore
            frames, kw = [], {"api_key": self.api_key} if self.api_key else {}
            for sid in self.series_ids:
                try:
                    s = web.DataReader(sid, "fred", start, end, **kw)[sid]
                    s.name = sid
                    frames.append(s)
                except Exception as exc:
                    logger.warning("FRED %s: %s", sid, exc)
                    frames.append(pd.Series(0.0, index=idx, name=sid))
            if frames:
                macro = pd.concat(frames, axis=1)
                macro.index = pd.to_datetime(macro.index, utc=True)
                return self._align_to_index(macro, idx).fillna(0.0)
        except ImportError:
            logger.warning("pandas_datareader not installed – macro set to zero.")
        return pd.DataFrame(0.0, index=idx, columns=self.series_ids)


# ─────────────────────────────────────────────────────────────────────────────
# 4. Sliding-Window Builder  (zero-lookahead sequences)
# ─────────────────────────────────────────────────────────────────────────────

class SlidingWindowBuilder:
    """
    Converts a (T, D) feature matrix into (N, seq_len, D) windows.

    Zero-lookahead guarantee:
        Window[i] = rows [i … i+seq_len-1]
        Label[i]  = row [i+seq_len] (next day – never inside the window)
    """

    def __init__(self, seq_len: int = LOOKBACK_DAYS):
        self.seq_len = seq_len

    def build(
        self,
        feature_df: pd.DataFrame,
        target_col: str = "Return_1d",
    ) -> tuple[np.ndarray, np.ndarray, pd.DatetimeIndex]:
        mat = feature_df.values.astype(np.float32)
        T, D = mat.shape
        if T < self.seq_len + 1:
            raise ValueError(
                f"Only {T} rows but seq_len={self.seq_len}. "
                "Increase lookback or reduce seq_len."
            )
        cols = feature_df.columns.tolist()
        ti   = cols.index(target_col) if target_col in cols else 1

        X_list, y_list, date_list = [], [], []
        for i in range(T - self.seq_len):
            X_list.append(mat[i : i + self.seq_len])
            y_list.append(mat[i + self.seq_len, ti])
            date_list.append(feature_df.index[i + self.seq_len])

        X     = np.stack(X_list)
        y     = np.array(y_list, dtype=np.float32)
        dates = pd.DatetimeIndex(date_list)
        logger.info("SlidingWindowBuilder › X=%s  y=%s", X.shape, y.shape)
        return X, y, dates

    def latest_window(self, feature_df: pd.DataFrame) -> np.ndarray:
        mat = feature_df.values.astype(np.float32)
        if len(mat) < self.seq_len:
            raise ValueError("Insufficient rows for latest_window.")
        return mat[-self.seq_len:]   # (seq_len, D)


# ─────────────────────────────────────────────────────────────────────────────
# 5. Master Orchestrator
# ─────────────────────────────────────────────────────────────────────────────

class SynchronizedDataLoader:
    """
    Orchestrates all loaders into a single time-aligned AlignedBundle.

    Streamlit Cloud:
        loader = SynchronizedDataLoader(ticker="SPY")
        bundle = loader.load()

    Claude Code CLI:
        loader = SynchronizedDataLoader(ticker="SPY", newsapi_key="xxx")
        bundle = loader.load(lookback_days=90)
    """

    def __init__(
        self,
        ticker:      str           = "SPY",
        seq_len:     int           = LOOKBACK_DAYS,
        newsapi_key: Optional[str] = None,
        fred_key:    Optional[str] = None,
    ):
        self.ticker      = ticker
        self.seq_len     = seq_len
        self.eq_loader   = EquityLoader(ticker)
        self.geo_loader  = GeopoliticalLoader(api_key=newsapi_key)
        self.mac_loader  = MacroLoader(api_key=fred_key)
        self.win_builder = SlidingWindowBuilder(seq_len=seq_len)

    def load(
        self,
        lookback_days: int           = LOOKBACK_DAYS + 30,
        end_date:      Optional[str] = None,
    ) -> AlignedBundle:
        end   = end_date or datetime.utcnow().strftime("%Y-%m-%d")
        start = (
            datetime.strptime(end, "%Y-%m-%d") - timedelta(days=lookback_days)
        ).strftime("%Y-%m-%d")

        logger.info("SynchronizedDataLoader › %s  %s → %s", self.ticker, start, end)

        ohlcv                = self.eq_loader.fetch(start, end)
        news_daily, articles = self.geo_loader.fetch(start, end)
        macro                = self.mac_loader.fetch(start, end)

        idx        = ohlcv.aligned_index
        news_daily = BaseLoader._align_to_index(news_daily, idx)
        macro      = BaseLoader._align_to_index(macro, idx)
        news_daily["headline_concat"] = news_daily["headline_concat"].fillna("")
        news_daily["headline_count"]  = news_daily["headline_count"].fillna(0).astype(int)

        logger.info(
            "Bundle ready › equity=%d rows | news=%d rows | macro=%d cols | articles=%d",
            len(idx), len(news_daily), macro.shape[1], len(articles),
        )
        return AlignedBundle(
            ticker=self.ticker, ohlcv=ohlcv, news_daily=news_daily, macro=macro,
            aligned_index=idx, articles=articles, start=start, end=end,
        )

    def build_sequences(self, bundle: AlignedBundle) -> tuple[np.ndarray, np.ndarray, pd.DatetimeIndex]:
        return self.win_builder.build(bundle.ohlcv.features)

    def latest_window(self, bundle: AlignedBundle) -> np.ndarray:
        return self.win_builder.latest_window(bundle.ohlcv.features)


# ─────────────────────────────────────────────────────────────────────────────
# CLI smoke-test
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    loader = SynchronizedDataLoader(ticker="SPY", seq_len=60)
    bundle = loader.load(lookback_days=90)

    print(f"\n{'='*60}")
    print(f"Ticker:       {bundle.ticker}")
    print(f"Date range:   {bundle.start} → {bundle.end}")
    print(f"Trading days: {len(bundle.aligned_index)}")
    print(f"OHLCV shape:  {bundle.ohlcv.features.shape}")
    print(f"Articles:     {len(bundle.articles)}")

    print("\n── OHLCV features (last 3 rows) ──")
    print(bundle.ohlcv.features.tail(3).to_string())

    print("\n── News daily (last 5 rows) ──")
    print(bundle.news_daily[
        ["headline_count","reuters_count","ft_count","aljazeera_count"]
    ].tail(5))

    if bundle.articles:
        print("\n── Sample articles ──")
        for a in bundle.articles[:3]:
            print(f"  [{a.source_label}] {a.title[:80]}")

    X, y, dates = loader.build_sequences(bundle)
    print(f"\nSequences › X={X.shape}  y={y.shape}")
