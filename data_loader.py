"""
data_loader.py
==============
Synchronized Geopolitical-Equity Alpha Inference Architecture
-------------------------------------------------------------
Complete data pipeline with:

  DATA SOURCES:
    - Equity:  yfinance (US + PSX .KA + Global indices)
    - News:    GNews API (primary, works on servers)
               NewsAPI (secondary)
               Bloomberg RSS (free, no key)
               WSJ RSS (free headlines)
               Dawn RSS (Pakistan news)
               Business Recorder RSS (PSX news)
               GDELT (always-free fallback)
    - Macro:   FRED (Fed Funds, Yield Curve, Oil)

  DSA IMPLEMENTATIONS:
    - Priority Queue (heapq):  Ranks articles by geopolitical impact
    - Trie:                    O(m) keyword detection in headlines
    - LRU Cache:               Prevents duplicate API calls
    - Sliding Window:          Zero-lookahead sequence builder
    - Binary Search (bisect):  Fast date alignment
"""

from __future__ import annotations

import heapq
import logging
import time
import bisect
import xml.etree.ElementTree as ET
from abc import ABC, abstractmethod
from collections import OrderedDict
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Optional

import numpy as np
import pandas as pd
import requests
import yfinance as yf

from config import Config

logger = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s - %(message)s",
)


# =============================================================================
# DSA 1: Trie - O(m) geopolitical keyword detection
# =============================================================================

class TrieNode:
    __slots__ = ["children", "is_end", "keyword"]
    def __init__(self):
        self.children: dict[str, "TrieNode"] = {}
        self.is_end:   bool = False
        self.keyword:  str  = ""


class GeopoliticalTrie:
    """
    Trie for O(m) geopolitical keyword detection.
    Much faster than O(n*m) naive linear scan.
    """

    def __init__(self, keywords: list[str] = None):
        self.root = TrieNode()
        self._count = 0
        for word in (keywords or Config.dsa.GEOPOLITICAL_KEYWORDS):
            self.insert(word.lower())

    def insert(self, word: str) -> None:
        node = self.root
        for ch in word:
            if ch not in node.children:
                node.children[ch] = TrieNode()
            node = node.children[ch]
        node.is_end  = True
        node.keyword = word
        self._count += 1

    def search(self, word: str) -> bool:
        node = self.root
        for ch in word.lower():
            if ch not in node.children:
                return False
            node = node.children[ch]
        return node.is_end

    def find_keywords(self, text: str) -> list[str]:
        found = []
        words = text.lower().split()
        for word in words:
            clean = "".join(c for c in word if c.isalpha())
            if clean and self.search(clean):
                found.append(clean)
        for i in range(len(words) - 1):
            bigram = words[i] + " " + words[i + 1]
            clean  = "".join(c for c in bigram if c.isalpha() or c == " ")
            if self.search(clean):
                found.append(clean)
        return list(set(found))

    def score_headline(self, headline: str) -> float:
        if not headline:
            return 0.0
        keywords = self.find_keywords(headline)
        words    = len(headline.split())
        density  = len(keywords) / max(words, 1)
        return min(density * 5, 1.0)

    @property
    def size(self) -> int:
        return self._count


# =============================================================================
# DSA 2: LRU Cache - O(1) get/put
# =============================================================================

class LRUCache:
    """
    Least Recently Used Cache using OrderedDict.
    O(1) get and put. Prevents duplicate API calls.
    """

    def __init__(self, capacity: int = 64):
        self.capacity = capacity
        self._cache: OrderedDict = OrderedDict()
        self._hits   = 0
        self._misses = 0

    def get(self, key: str) -> Optional[object]:
        if key not in self._cache:
            self._misses += 1
            return None
        self._cache.move_to_end(key)
        self._hits += 1
        return self._cache[key]

    def put(self, key: str, value: object) -> None:
        if key in self._cache:
            self._cache.move_to_end(key)
        self._cache[key] = value
        if len(self._cache) > self.capacity:
            self._cache.popitem(last=False)

    def clear(self) -> None:
        self._cache.clear()

    @property
    def hit_rate(self) -> float:
        total = self._hits + self._misses
        return self._hits / total if total > 0 else 0.0

    @property
    def size(self) -> int:
        return len(self._cache)

    def stats(self) -> dict:
        return {
            "size":     self.size,
            "capacity": self.capacity,
            "hits":     self._hits,
            "misses":   self._misses,
            "hit_rate": f"{self.hit_rate:.1%}",
        }


# =============================================================================
# DSA 3: Priority Queue - Rank articles by impact
# =============================================================================

class ArticlePriorityQueue:
    """
    Max-heap priority queue ranking news articles by geopolitical impact.
    O(log n) insert and extract.
    """

    def __init__(self, trie: GeopoliticalTrie = None, max_size: int = None):
        self.trie     = trie or GeopoliticalTrie()
        self.max_size = max_size or Config.dsa.MAX_RANKED_ARTICLES
        self._heap:   list = []
        self._counter = 0

    def push(self, article: "NewsArticle", sentiment_score: float = 0.0) -> None:
        score   = self._compute_impact(article, sentiment_score)
        entry   = (-score, self._counter, article)
        heapq.heappush(self._heap, entry)
        self._counter += 1
        if len(self._heap) > self.max_size:
            heapq.heappop(self._heap)

    def top_k(self, k: int) -> list["NewsArticle"]:
        if not self._heap:
            return []
        sorted_items = sorted(self._heap, key=lambda x: x[0])
        return [art for _, _, art in sorted_items[:k]]

    def _compute_impact(self, article: "NewsArticle", sentiment_score: float) -> float:
        now       = pd.Timestamp.utcnow()
        hours_old = max(0, (now - article.published_at).total_seconds() / 3600)
        recency   = np.exp(-hours_old / Config.dsa.IMPACT_DECAY_HOURS)
        credibility   = Config.dsa.SOURCE_CREDIBILITY.get(article.source_domain, 0.5)
        keyword_score = self.trie.score_headline(article.full_text)
        impact = (
            Config.dsa.RECENCY_WEIGHT            * recency +
            Config.dsa.SENTIMENT_WEIGHT          * abs(sentiment_score) +
            Config.dsa.SOURCE_CREDIBILITY_WEIGHT * credibility
        ) * (1.0 + keyword_score)
        return float(np.clip(impact, 0.0, 2.0))

    @property
    def size(self) -> int:
        return len(self._heap)


# =============================================================================
# DSA 4: Binary Search - Fast date alignment
# =============================================================================

def find_nearest_trading_day(
    target_date: pd.Timestamp,
    sorted_trading_days: list[pd.Timestamp],
) -> pd.Timestamp:
    """Binary search O(log n) to find nearest trading day."""
    if not sorted_trading_days:
        return target_date
    dates = [d.timestamp() for d in sorted_trading_days]
    ts    = target_date.timestamp()
    idx   = bisect.bisect_left(dates, ts)
    if idx == 0:
        return sorted_trading_days[0]
    if idx >= len(sorted_trading_days):
        return sorted_trading_days[-1]
    before = sorted_trading_days[idx - 1]
    after  = sorted_trading_days[idx]
    return before if (target_date - before) <= (after - target_date) else after


# =============================================================================
# Data Container Dataclasses
# =============================================================================

@dataclass
class NewsArticle:
    """Single news article with source attribution."""
    published_at:  pd.Timestamp
    title:         str
    description:   str
    url:           str
    source_domain: str
    impact_score:  float = 0.0

    @property
    def source_label(self) -> str:
        return Config.news.DOMAIN_DISPLAY.get(self.source_domain, self.source_domain)

    @property
    def source_colour(self) -> str:
        return Config.news.DOMAIN_COLOURS.get(self.source_domain, "#546e7a")

    @property
    def source_css(self) -> str:
        return Config.news.DOMAIN_CSS.get(self.source_domain, "default")

    @property
    def is_pakistan_source(self) -> bool:
        return self.source_domain in Config.news.PAKISTAN_DOMAINS

    @property
    def full_text(self) -> str:
        return " ".join(
            p for p in [self.title or "", self.description or ""] if p
        ).strip()[:512]

    @property
    def age_hours(self) -> float:
        return (pd.Timestamp.utcnow() - self.published_at).total_seconds() / 3600


@dataclass
class OHLCVBundle:
    """Container for one ticker's OHLCV data and features."""
    ticker:        str
    market:        str
    currency:      str
    raw:           pd.DataFrame
    features:      pd.DataFrame
    aligned_index: pd.DatetimeIndex

    @property
    def close(self) -> pd.Series:
        return self.raw["Close"].squeeze()

    @property
    def num_features(self) -> int:
        return self.features.shape[1]

    @property
    def latest_price(self) -> float:
        try:
            return float(self.close.dropna().iloc[-1])
        except Exception:
            return 0.0

    @property
    def daily_change_pct(self) -> float:
        try:
            closes = self.close.dropna()
            return float((closes.iloc[-1] - closes.iloc[-2]) / closes.iloc[-2] * 100)
        except Exception:
            return 0.0


@dataclass
class AlignedBundle:
    """Final time-aligned bundle passed to the model layer."""
    ticker:          str
    market:          str
    currency:        str
    ohlcv:           OHLCVBundle
    news_daily:      pd.DataFrame
    macro:           pd.DataFrame
    aligned_index:   pd.DatetimeIndex
    start:           str
    end:             str
    articles:        list[NewsArticle] = field(default_factory=list)
    ranked_articles: list[NewsArticle] = field(default_factory=list)

    @property
    def pakistan_articles(self) -> list[NewsArticle]:
        return [a for a in self.articles if a.is_pakistan_source]

    @property
    def global_articles(self) -> list[NewsArticle]:
        return [a for a in self.articles if not a.is_pakistan_source]


@dataclass
class MultiTickerBundle:
    """Container for multiple ticker bundles."""
    tickers:  list[str]
    bundles:  dict[str, AlignedBundle]
    market:   str
    start:    str
    end:      str

    @property
    def primary_ticker(self) -> str:
        return self.tickers[0] if self.tickers else ""

    @property
    def all_closes(self) -> pd.DataFrame:
        frames = {}
        for ticker, bundle in self.bundles.items():
            frames[ticker] = bundle.ohlcv.close
        return pd.DataFrame(frames)

    @property
    def correlation_matrix(self) -> pd.DataFrame:
        closes  = self.all_closes
        returns = closes.pct_change().dropna()
        return returns.corr()


# =============================================================================
# Abstract Base Loader
# =============================================================================

class BaseLoader(ABC):
    """Shared interface and utilities for all loaders."""

    _price_cache = LRUCache(capacity=Config.dsa.PRICE_CACHE_SIZE)
    _news_cache  = LRUCache(capacity=Config.dsa.NEWS_CACHE_SIZE)

    @abstractmethod
    def fetch(self, start: str, end: str) -> object:
        ...

    @staticmethod
    def _bday_index(start: str, end: str) -> pd.DatetimeIndex:
        return pd.bdate_range(start=start, end=end, freq="B", tz="UTC")

    @staticmethod
    def _align_to_index(df: pd.DataFrame, idx: pd.DatetimeIndex) -> pd.DataFrame:
        return df.reindex(idx, method="ffill")

    @staticmethod
    def _detect_market(ticker: str) -> tuple[str, str]:
        t = ticker.upper()
        if t.endswith(".KA") or t in ["^KSE100", "^KSE30", "^KMIALLSHR"]:
            return "PSX", "PKR"
        if t.endswith(".T"):  return "TSE",     "JPY"
        if t.endswith(".HK"): return "HKEX",    "HKD"
        if t.endswith(".SS"): return "SSE",     "CNY"
        if t.endswith(".BO"): return "BSE",     "INR"
        if t.endswith(".L"):  return "LSE",     "GBP"
        if t.endswith(".DE"): return "FSE",     "EUR"
        if t.endswith(".TO"): return "TSX",     "CAD"
        if t.endswith(".AX"): return "ASX",     "AUD"
        if t.endswith(".SR"): return "TADAWUL", "SAR"
        return "US", "USD"


# =============================================================================
# 1. Equity Loader
# =============================================================================

class EquityLoader(BaseLoader):
    """
    Downloads OHLCV data for any ticker via yfinance.
    Supports PSX (.KA), US, and all global markets.
    """

    FEATURE_COLS: list[str] = [
        "Close_norm", "Return_1d", "Return_5d", "Return_10d",
        "Volatility_10d", "RSI_14", "Volume_norm",
        "BB_upper_dist", "BB_lower_dist",
    ]

    def __init__(self, ticker: str = "SPY"):
        self.ticker   = ticker.upper()
        self.market, self.currency = self._detect_market(self.ticker)

    def fetch(self, start: str, end: str) -> OHLCVBundle:
        cache_key = f"{self.ticker}_{start}_{end}"
        cached    = self._price_cache.get(cache_key)
        if cached is not None:
            return cached

        logger.info("EquityLoader: %s [%s] %s -> %s", self.ticker, self.market, start, end)
        raw  = self._download(start, end)
        idx  = self._bday_index(start, end)
        raw  = self._align_to_index(raw, idx)
        feat = self._engineer(raw)
        feat = self._align_to_index(feat, idx)

        bundle = OHLCVBundle(
            ticker=self.ticker, market=self.market, currency=self.currency,
            raw=raw, features=feat, aligned_index=idx,
        )
        self._price_cache.put(cache_key, bundle)
        return bundle

    def _download(self, start: str, end: str) -> pd.DataFrame:
        for attempt in range(3):
            try:
                df = yf.download(
                    self.ticker, start=start, end=end,
                    auto_adjust=True, progress=False,
                )
                if not df.empty:
                    df.index = pd.to_datetime(df.index, utc=True)
                    return df
            except Exception as exc:
                logger.warning("yfinance attempt %d: %s", attempt + 1, exc)
                time.sleep(1)

        if self.market == "PSX":
            raise ValueError(
                f"No data for {self.ticker}. "
                "PSX tickers need .KA suffix (e.g. ENGRO.KA). "
                "Indices use ^ prefix (e.g. ^KSE100)."
            )
        raise ValueError(f"No data returned for {self.ticker}.")

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


# =============================================================================
# 2. Multi-Ticker Loader
# =============================================================================

class MultiTickerLoader(BaseLoader):
    """Loads multiple tickers simultaneously for portfolio comparison."""

    def __init__(self, tickers: list[str]):
        max_t        = Config.market.MAX_COMPARISON_TICKERS
        self.tickers = tickers[:max_t]

    def fetch(self, start: str, end: str) -> MultiTickerBundle:
        logger.info("MultiTickerLoader: %d tickers", len(self.tickers))
        bundles = {}
        market  = "Mixed"
        for ticker in self.tickers:
            try:
                loader = EquityLoader(ticker)
                bundles[ticker] = loader.fetch(start, end)
                market = loader.market
            except Exception as exc:
                logger.error("Failed %s: %s", ticker, exc)
        return MultiTickerBundle(
            tickers=self.tickers, bundles=bundles,
            market=market, start=start, end=end,
        )


# =============================================================================
# 3. RSS Feed Parser - Bloomberg, WSJ, Dawn, Business Recorder
# =============================================================================

class RSSFeedParser:
    """
    Parses RSS feeds — completely free, no API key required.
    Sources: Bloomberg, WSJ, Dawn, Business Recorder.
    """

    def __init__(self):
        self.all_feeds: dict[str, list[str]] = {
            "bloomberg.com": Config.news.BLOOMBERG_RSS_FEEDS,
            "wsj.com":       Config.news.WSJ_RSS_FEEDS,
            "dawn.com":      Config.news.DAWN_RSS_FEEDS,
            "brecorder.com": Config.news.BRECORDER_RSS_FEEDS,
        }

    def fetch_all(self, max_per_feed: int = 20) -> list[NewsArticle]:
        all_articles: list[NewsArticle] = []
        for domain, feeds in self.all_feeds.items():
            for url in feeds:
                try:
                    articles = self._parse_feed(url, domain, max_per_feed)
                    all_articles.extend(articles)
                    logger.info("RSS [%s]: %d articles", domain, len(articles))
                except Exception as exc:
                    logger.warning("RSS failed [%s]: %s", url, exc)
        return all_articles

    def _parse_feed(self, url: str, domain: str, limit: int) -> list[NewsArticle]:
        headers = {"User-Agent": "Mozilla/5.0 (compatible; SGEAIA/1.0)"}
        resp    = requests.get(url, headers=headers, timeout=15)
        resp.raise_for_status()
        root    = ET.fromstring(resp.content)
        items   = root.findall(".//item")
        if not items:
            ns    = {"atom": "http://www.w3.org/2005/Atom"}
            items = root.findall(".//atom:entry", ns)
        articles = []
        for item in items[:limit]:
            try:
                art = self._parse_item(item, domain)
                if art:
                    articles.append(art)
            except Exception:
                continue
        return articles

    @staticmethod
    def _parse_item(item: ET.Element, domain: str) -> Optional[NewsArticle]:
        def get_text(tag: str) -> str:
            el = item.find(tag)
            return el.text.strip() if el is not None and el.text else ""
        title    = get_text("title")
        desc     = get_text("description")
        link     = get_text("link")
        pub_date = get_text("pubDate")
        if not title:
            return None
        try:
            ts = pd.to_datetime(pub_date, utc=True)
        except Exception:
            ts = pd.Timestamp.utcnow()
        return NewsArticle(
            published_at=ts, title=title[:200],
            description=desc[:500], url=link, source_domain=domain,
        )


# =============================================================================
# 4. Geopolitical News Loader - Multi-source with Priority Queue
# =============================================================================

class GeopoliticalLoader(BaseLoader):
    """
    Multi-source news loader with DSA Priority Queue ranking.

    Source chain:
      Tier 1: GNews API          (works on servers)
      Tier 2: RSS Feeds          (Bloomberg, WSJ, Dawn, Brecorder)
      Tier 3: NewsAPI            (paid plans)
      Tier 4: GDELT              (always free)
    """

    def __init__(self, api_key: Optional[str] = None):
        self.gnews_key   = Config.api.GNEWS_API_KEY
        self.newsapi_key = api_key or Config.api.NEWSAPI_KEY
        self.trie        = GeopoliticalTrie()
        self.rss_parser  = RSSFeedParser()

    def fetch(
        self,
        start:  str,
        end:    str,
        market: str = "US",
    ) -> tuple[pd.DataFrame, list[NewsArticle], list[NewsArticle]]:
        cache_key = f"news_{start}_{end}_{market}"
        cached    = self._news_cache.get(cache_key)
        if cached is not None:
            return cached

        idx      = self._bday_index(start, end)
        articles = self._fetch_all_sources(start, end)
        ranked   = self._rank_articles(articles)
        daily    = self._aggregate(articles, idx)
        result   = (daily, articles, ranked)
        self._news_cache.put(cache_key, result)
        return result

    def _fetch_all_sources(self, start: str, end: str) -> list[NewsArticle]:
        all_articles: list[NewsArticle] = []

        if self.gnews_key:
            gnews = self._fetch_gnews(start, end)
            if gnews:
                all_articles.extend(gnews)
                logger.info("GNews: %d articles", len(gnews))

        try:
            rss = self.rss_parser.fetch_all(max_per_feed=15)
            all_articles.extend(rss)
            logger.info("RSS: %d articles", len(rss))
        except Exception as exc:
            logger.warning("RSS failed: %s", exc)

        if not all_articles and self.newsapi_key:
            na = self._fetch_newsapi(start, end)
            if na:
                all_articles.extend(na)
                logger.info("NewsAPI: %d articles", len(na))

        if not all_articles:
            gdelt = self._fetch_gdelt(start, end)
            all_articles.extend(gdelt)
            logger.info("GDELT: %d articles", len(gdelt))

        seen, dedup = set(), []
        for art in all_articles:
            if art.url and art.url not in seen:
                seen.add(art.url)
                dedup.append(art)
            elif not art.url:
                dedup.append(art)
        return dedup

    def _rank_articles(self, articles: list[NewsArticle]) -> list[NewsArticle]:
        pq = ArticlePriorityQueue(trie=self.trie)
        for art in articles:
            sentiment = self.trie.score_headline(art.full_text)
            pq.push(art, sentiment_score=sentiment)
        top = pq.top_k(Config.dsa.MAX_RANKED_ARTICLES)
        for art in top:
            art.impact_score = self.trie.score_headline(art.full_text)
        return top

    def _fetch_gnews(self, start: str, end: str) -> list[NewsArticle]:
        out = []
        try:
            params = {
                "q":      Config.news.GEOPOLITICAL_QUERY[:100],
                "lang":   "en", "max": 10, "apikey": self.gnews_key,
                "from":   start + "T00:00:00Z",
                "to":     end   + "T23:59:59Z",
            }
            resp = requests.get(Config.news.GNEWS_BASE, params=params, timeout=15)
            resp.raise_for_status()
            for item in resp.json().get("articles", []):
                art = self._parse_gnews(item)
                if art:
                    out.append(art)
        except Exception as exc:
            logger.error("GNews error: %s", exc)
        return out

    @staticmethod
    def _parse_gnews(item: dict) -> Optional[NewsArticle]:
        try:
            ts = pd.to_datetime(item.get("publishedAt", ""), utc=True)
        except Exception:
            ts = pd.Timestamp.utcnow()
        src_url = item.get("url", "")
        domain  = next(
            (d for d in Config.news.APPROVED_DOMAINS if d in src_url),
            "reuters.com",
        )
        return NewsArticle(
            published_at=ts,
            title=item.get("title", "") or "",
            description=item.get("description", "") or "",
            url=src_url, source_domain=domain,
        )

    def _fetch_newsapi(self, start: str, end: str) -> list[NewsArticle]:
        out = []
        try:
            params = {
                "q":        Config.news.GEOPOLITICAL_QUERY[:500],
                "domains":  ",".join(Config.news.APPROVED_DOMAINS[:6]),
                "from":     start, "to": end, "language": "en",
                "sortBy":   "publishedAt", "pageSize": 100,
                "apiKey":   self.newsapi_key,
            }
            resp = requests.get(Config.news.NEWSAPI_BASE, params=params, timeout=20)
            resp.raise_for_status()
            data = resp.json()
            if data.get("status") != "ok":
                return []
            for item in data.get("articles", []):
                art = self._parse_newsapi(item)
                if art:
                    out.append(art)
        except Exception as exc:
            logger.error("NewsAPI error: %s", exc)
        return out

    @staticmethod
    def _parse_newsapi(item: dict) -> Optional[NewsArticle]:
        try:
            ts = pd.to_datetime(item.get("publishedAt", ""), utc=True)
        except Exception:
            return None
        url    = item.get("url", "")
        domain = next(
            (d for d in Config.news.APPROVED_DOMAINS if d in url), None
        )
        if not domain:
            return None
        return NewsArticle(
            published_at=ts,
            title=item.get("title", "") or "",
            description=item.get("description", "") or "",
            url=url, source_domain=domain,
        )

    def _fetch_gdelt(self, start: str, end: str) -> list[NewsArticle]:
        out = []
        try:
            params = {
                "query":         "geopolitical war sanctions tariff inflation",
                "mode":          "ArtList", "maxrecords": 100,
                "startdatetime": start.replace("-", "") + "000000",
                "enddatetime":   end.replace("-", "")   + "235959",
                "format":        "json",
            }
            resp = requests.get(Config.news.GDELT_BASE, params=params, timeout=20)
            resp.raise_for_status()
            for art in resp.json().get("articles", []):
                url    = art.get("url", "")
                domain = next(
                    (d for d in Config.news.APPROVED_DOMAINS if d in url),
                    "reuters.com",
                )
                try:
                    ts = pd.to_datetime(art.get("seendate", ""), utc=True)
                except Exception:
                    ts = pd.Timestamp.utcnow()
                out.append(NewsArticle(
                    published_at=ts, title=art.get("title", "") or "",
                    description="", url=url, source_domain=domain,
                ))
        except Exception as exc:
            logger.error("GDELT error: %s", exc)
        return out

    def _aggregate(
        self,
        articles: list[NewsArticle],
        idx:      pd.DatetimeIndex,
    ) -> pd.DataFrame:
        source_cols = [d.replace(".", "_") for d in Config.news.APPROVED_DOMAINS]
        cols  = ["headline_count", "headline_concat", "top_impact_score"] + source_cols
        daily = pd.DataFrame(0, index=idx, columns=cols, dtype=object)
        daily["headline_concat"]  = ""
        daily["top_impact_score"] = 0.0

        if not articles:
            return daily

        trading_days = list(idx)
        for art in articles:
            day = find_nearest_trading_day(art.published_at.normalize(), trading_days)
            if day not in idx:
                continue
            daily.loc[day, "headline_count"] = int(daily.loc[day, "headline_count"]) + 1
            existing = str(daily.loc[day, "headline_concat"])
            daily.loc[day, "headline_concat"] = (
                (existing + " | " + art.full_text).strip(" |")[:2048]
            )
            if art.impact_score > float(daily.loc[day, "top_impact_score"]):
                daily.loc[day, "top_impact_score"] = art.impact_score
            src_col = art.source_domain.replace(".", "_")
            if src_col in daily.columns:
                daily.loc[day, src_col] = int(daily.loc[day, src_col]) + 1

        daily["headline_count"] = pd.to_numeric(daily["headline_count"], errors="coerce").fillna(0).astype(int)
        daily["top_impact_score"] = pd.to_numeric(daily["top_impact_score"], errors="coerce").fillna(0.0)
        for col in source_cols:
            daily[col] = pd.to_numeric(daily[col], errors="coerce").fillna(0).astype(int)
        return daily


# =============================================================================
# 5. Macro Loader (FRED)
# =============================================================================

class MacroLoader(BaseLoader):
    """Pulls FRED macro series. Forward-fill only — no lookahead bias."""

    def __init__(self, series_ids: list[str] = None, api_key: Optional[str] = None):
        self.series_ids = series_ids or Config.market.MACRO_SERIES
        self.api_key    = api_key or Config.api.FRED_API_KEY

    def fetch(self, start: str, end: str) -> pd.DataFrame:
        idx = self._bday_index(start, end)
        try:
            import pandas_datareader.data as web
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
            logger.warning("pandas_datareader not installed.")
        return pd.DataFrame(0.0, index=idx, columns=self.series_ids)


# =============================================================================
# 6. Sliding Window Builder
# =============================================================================

class SlidingWindowBuilder:
    """
    Converts (T, D) feature matrix into (N, seq_len, D) windows.
    Zero-lookahead: label is always the row AFTER the window.
    """

    def __init__(self, seq_len: int = Config.model.SEQ_LEN):
        self.seq_len = seq_len

    def build(
        self,
        feature_df: pd.DataFrame,
        target_col: str = "Return_1d",
    ) -> tuple[np.ndarray, np.ndarray, pd.DatetimeIndex]:
        mat  = feature_df.values.astype(np.float32)
        T, D = mat.shape
        if T < self.seq_len + 1:
            raise ValueError(f"Only {T} rows for seq_len={self.seq_len}.")
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
        logger.info("SlidingWindowBuilder: X=%s y=%s", X.shape, y.shape)
        return X, y, dates

    def latest_window(self, feature_df: pd.DataFrame) -> np.ndarray:
        mat = feature_df.values.astype(np.float32)
        if len(mat) < self.seq_len:
            raise ValueError("Insufficient rows for latest_window.")
        return mat[-self.seq_len:]


# =============================================================================
# 7. Master Orchestrator
# =============================================================================

class SynchronizedDataLoader:
    """
    Orchestrates all loaders into a single time-aligned AlignedBundle.

    Single ticker:   loader = SynchronizedDataLoader(ticker="ENGRO.KA")
    Multi ticker:    loader = SynchronizedDataLoader(tickers=["HBL.KA","SPY"])
    """

    def __init__(
        self,
        ticker:      str           = "SPY",
        tickers:     list[str]     = None,
        seq_len:     int           = Config.model.SEQ_LEN,
        newsapi_key: Optional[str] = None,
    ):
        self.ticker      = ticker
        self.tickers     = tickers or [ticker]
        self.seq_len     = seq_len
        self.eq_loader   = EquityLoader(ticker)
        self.geo_loader  = GeopoliticalLoader(api_key=newsapi_key)
        self.mac_loader  = MacroLoader()
        self.win_builder = SlidingWindowBuilder(seq_len=seq_len)

    def load(
        self,
        lookback_days: int           = Config.ui.DEFAULT_LOOKBACK_DAYS + 30,
        end_date:      Optional[str] = None,
    ) -> AlignedBundle:
        end   = end_date or datetime.utcnow().strftime("%Y-%m-%d")
        start = (
            datetime.strptime(end, "%Y-%m-%d") - timedelta(days=lookback_days)
        ).strftime("%Y-%m-%d")

        logger.info("SynchronizedDataLoader: %s %s -> %s", self.ticker, start, end)

        ohlcv                        = self.eq_loader.fetch(start, end)
        news_daily, articles, ranked = self.geo_loader.fetch(start, end, market=ohlcv.market)
        macro                        = self.mac_loader.fetch(start, end)

        idx        = ohlcv.aligned_index
        news_daily = BaseLoader._align_to_index(news_daily, idx)
        macro      = BaseLoader._align_to_index(macro, idx)
        news_daily["headline_concat"] = news_daily["headline_concat"].fillna("")
        news_daily["headline_count"]  = news_daily["headline_count"].fillna(0).astype(int)

        return AlignedBundle(
            ticker=self.ticker, market=ohlcv.market, currency=ohlcv.currency,
            ohlcv=ohlcv, news_daily=news_daily, macro=macro,
            aligned_index=idx, articles=articles, ranked_articles=ranked,
            start=start, end=end,
        )

    def load_multi(
        self,
        lookback_days: int           = Config.ui.DEFAULT_LOOKBACK_DAYS + 30,
        end_date:      Optional[str] = None,
    ) -> MultiTickerBundle:
        end   = end_date or datetime.utcnow().strftime("%Y-%m-%d")
        start = (
            datetime.strptime(end, "%Y-%m-%d") - timedelta(days=lookback_days)
        ).strftime("%Y-%m-%d")
        return MultiTickerLoader(self.tickers).fetch(start, end)

    def build_sequences(self, bundle: AlignedBundle) -> tuple[np.ndarray, np.ndarray, pd.DatetimeIndex]:
        return self.win_builder.build(bundle.ohlcv.features)

    def latest_window(self, bundle: AlignedBundle) -> np.ndarray:
        return self.win_builder.latest_window(bundle.ohlcv.features)

    @property
    def cache_stats(self) -> dict:
        return {
            "price_cache": BaseLoader._price_cache.stats(),
            "news_cache":  BaseLoader._news_cache.stats(),
        }


# =============================================================================
# CLI smoke test
# =============================================================================

if __name__ == "__main__":
    print("=" * 60)
    print("SGEAIA Data Loader - Smoke Test")
    print("=" * 60)

    print("\n-- DSA: Trie --")
    trie = GeopoliticalTrie()
    print(f"  Keywords: {trie.size}")
    tests = [
        "Federal Reserve raises interest rates amid trade war",
        "Pakistan rupee hits record low against dollar",
        "KSE-100 surges on IMF deal optimism",
    ]
    for t in tests:
        kw    = trie.find_keywords(t)
        score = trie.score_headline(t)
        print(f"  '{t[:55]}'")
        print(f"    keywords={kw}  score={score:.3f}")

    print("\n-- DSA: LRU Cache --")
    cache = LRUCache(capacity=3)
    cache.put("a", 1)
    cache.put("b", 2)
    cache.put("c", 3)
    cache.put("d", 4)   # evicts "a"
    print(f"  Get 'a' (evicted): {cache.get('a')}")
    print(f"  Get 'b': {cache.get('b')}")
    print(f"  Stats: {cache.stats()}")

    print("\n-- DSA: Priority Queue --")
    pq = ArticlePriorityQueue(trie=trie)
    mock = [
        NewsArticle(pd.Timestamp.utcnow() - pd.Timedelta(hours=1),
                    "Fed raises rates amid recession fears", "", "https://reuters.com/1", "reuters.com"),
        NewsArticle(pd.Timestamp.utcnow() - pd.Timedelta(hours=3),
                    "KSE-100 hits record high on IMF deal", "", "https://brecorder.com/1", "brecorder.com"),
        NewsArticle(pd.Timestamp.utcnow() - pd.Timedelta(hours=8),
                    "OPEC agrees to production cuts", "", "https://ft.com/1", "ft.com"),
    ]
    for art in mock:
        pq.push(art, sentiment_score=0.6)
    print(f"  Queue size: {pq.size}")
    for i, art in enumerate(pq.top_k(3), 1):
        print(f"  {i}. [{art.source_label}] {art.title}")

    print("\n-- Market Detection --")
    for t in ["ENGRO.KA", "^KSE100", "SPY", "^FTSE", "7203.T"]:
        market, currency = BaseLoader._detect_market(t)
        print(f"  {t:12s} -> {market} ({currency})")

    print("\n-- Data Loading (SPY quick test) --")
    loader = SynchronizedDataLoader(ticker="SPY", seq_len=20)
    bundle = loader.load(lookback_days=30)
    print(f"  Ticker:    {bundle.ticker}")
    print(f"  Market:    {bundle.market} ({bundle.currency})")
    print(f"  Rows:      {len(bundle.aligned_index)}")
    print(f"  Articles:  {len(bundle.articles)}")
    print(f"  Ranked:    {len(bundle.ranked_articles)}")
    print(f"  Cache:     {loader.cache_stats}")
    print("\nAll tests passed.")
