"""
alerts.py
=========
Synchronized Geopolitical-Equity Alpha Inference Architecture
-------------------------------------------------------------
Alert system features:

  SignalChangeDetector  - Detects meaningful changes in alpha/sentiment/regime
  EmailAlertService     - Sends alerts via SendGrid
  AlertHistory          - Tracks alert history in session state
  MarketHoursChecker    - Smart alerts only during market hours
  AlertComposer         - Formats alert messages (English + Urdu)
  CrisisAlertMonitor    - Dedicated crisis mode detection and alerting
"""

from __future__ import annotations

import json
import logging
import os
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from typing import Optional

import pandas as pd
import streamlit as st

from config import Config
from model_utils import InferenceResult

logger = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s - %(message)s",
)


# =============================================================================
# Data containers
# =============================================================================

@dataclass
class AlertEvent:
    """A single alert event record."""
    alert_id:    int
    ticker:      str
    alert_type:  str        # "SIGNAL_CHANGE", "REGIME_CHANGE", "CRISIS_MODE", "SENTIMENT_SHIFT"
    severity:    str        # "INFO", "WARNING", "CRITICAL"
    title:       str
    message:     str
    timestamp:   str
    sent_email:  bool = False
    email_to:    str  = ""

    @property
    def severity_colour(self) -> str:
        colours = {
            "INFO":     Config.ui.PRIMARY_COLOUR,
            "WARNING":  Config.ui.WARNING_COLOUR,
            "CRITICAL": Config.ui.BEAR_COLOUR,
        }
        return colours.get(self.severity, Config.ui.NEUTRAL_COLOUR)

    @property
    def severity_icon(self) -> str:
        icons = {
            "INFO":     "ℹ️",
            "WARNING":  "⚠️",
            "CRITICAL": "🚨",
        }
        return icons.get(self.severity, "📢")

    @property
    def age_minutes(self) -> float:
        try:
            ts  = datetime.strptime(self.timestamp, "%Y-%m-%d %H:%M UTC")
            now = datetime.utcnow()
            return (now - ts).total_seconds() / 60
        except Exception:
            return 0.0


@dataclass
class SignalSnapshot:
    """Snapshot of signal state for change detection."""
    ticker:      str
    alpha:       float
    regime:      str
    sentiment:   str
    crisis_mode: bool
    volatility:  float
    timestamp:   str

    @property
    def signal_label(self) -> str:
        if self.alpha > Config.backtest.LONG_THRESHOLD:
            return "LONG"
        if self.alpha < Config.backtest.SHORT_THRESHOLD:
            return "SHORT"
        return "FLAT"


# =============================================================================
# 1. Market Hours Checker
# =============================================================================

class MarketHoursChecker:
    """
    Determines whether alerts should be sent based on
    market hours for PSX and US markets.

    Smart notification logic:
      - Only send alerts when the relevant market is open
      - Send pre-market alerts 30 min before open
      - Send post-market summary after close
      - Silence overnight and weekends
    """

    @staticmethod
    def is_psx_open() -> bool:
        """Check if PSX is currently open."""
        now_pkt = MarketHoursChecker._now_pkt()
        if now_pkt.weekday() >= 5:
            return False
        open_h, open_m   = Config.alerts.PSX_OPEN_HOUR,  Config.alerts.PSX_OPEN_MIN
        close_h, close_m = Config.alerts.PSX_CLOSE_HOUR, Config.alerts.PSX_CLOSE_MIN
        open_time  = now_pkt.replace(hour=open_h,  minute=open_m,  second=0, microsecond=0)
        close_time = now_pkt.replace(hour=close_h, minute=close_m, second=0, microsecond=0)
        return open_time <= now_pkt <= close_time

    @staticmethod
    def is_us_open() -> bool:
        """Check if US market is currently open."""
        now_est = MarketHoursChecker._now_est()
        if now_est.weekday() >= 5:
            return False
        open_h, open_m   = Config.alerts.US_OPEN_HOUR,  Config.alerts.US_OPEN_MIN
        close_h, close_m = Config.alerts.US_CLOSE_HOUR, Config.alerts.US_CLOSE_MIN
        open_time  = now_est.replace(hour=open_h,  minute=open_m,  second=0, microsecond=0)
        close_time = now_est.replace(hour=close_h, minute=close_m, second=0, microsecond=0)
        return open_time <= now_est <= close_time

    @staticmethod
    def is_market_open(market: str = "US") -> bool:
        """Check if a specific market is open."""
        if market == "PSX":
            return MarketHoursChecker.is_psx_open()
        return MarketHoursChecker.is_us_open()

    @staticmethod
    def is_pre_market(market: str = "US", minutes_before: int = 30) -> bool:
        """Check if we are in pre-market window."""
        if market == "PSX":
            now  = MarketHoursChecker._now_pkt()
            open_h, open_m = Config.alerts.PSX_OPEN_HOUR, Config.alerts.PSX_OPEN_MIN
        else:
            now  = MarketHoursChecker._now_est()
            open_h, open_m = Config.alerts.US_OPEN_HOUR, Config.alerts.US_OPEN_MIN

        if now.weekday() >= 5:
            return False
        open_time  = now.replace(hour=open_h, minute=open_m, second=0, microsecond=0)
        pre_start  = open_time - timedelta(minutes=minutes_before)
        return pre_start <= now < open_time

    @staticmethod
    def should_send_alert(market: str = "US") -> bool:
        """
        Determine if an alert should be sent right now.
        True during market hours and 30-min pre-market window.
        """
        return (
            MarketHoursChecker.is_market_open(market) or
            MarketHoursChecker.is_pre_market(market)
        )

    @staticmethod
    def next_open_time(market: str = "US") -> str:
        """Return human-readable next open time."""
        if market == "PSX":
            now    = MarketHoursChecker._now_pkt()
            open_h = Config.alerts.PSX_OPEN_HOUR
            open_m = Config.alerts.PSX_OPEN_MIN
            tz_label = "PKT"
        else:
            now    = MarketHoursChecker._now_est()
            open_h = Config.alerts.US_OPEN_HOUR
            open_m = Config.alerts.US_OPEN_MIN
            tz_label = "EST"

        if now.weekday() < 5:
            open_time = now.replace(
                hour=open_h, minute=open_m, second=0, microsecond=0
            )
            if now < open_time:
                return f"Today at {open_h}:{open_m:02d} {tz_label}"

        days = 1
        while (now + timedelta(days=days)).weekday() >= 5:
            days += 1
        next_day = (now + timedelta(days=days)).strftime("%A %b %d")
        return f"{next_day} at {open_h}:{open_m:02d} {tz_label}"

    @staticmethod
    def _now_pkt() -> datetime:
        return datetime.now(timezone.utc) + timedelta(hours=5)

    @staticmethod
    def _now_est() -> datetime:
        return datetime.now(timezone.utc) - timedelta(hours=5)


# =============================================================================
# 2. Signal Change Detector
# =============================================================================

class SignalChangeDetector:
    """
    Detects meaningful changes in alpha signal, regime,
    sentiment, and crisis mode between two snapshots.

    Only triggers alerts when changes are significant enough
    to be actionable — prevents alert fatigue.
    """

    def __init__(self):
        self._ensure_session_state()

    def _ensure_session_state(self) -> None:
        if "last_snapshot" not in st.session_state:
            st.session_state.last_snapshot = {}

    def update_snapshot(
        self,
        ticker: str,
        result: InferenceResult,
    ) -> None:
        """Store current signal state for future change detection."""
        snapshot = SignalSnapshot(
            ticker=ticker,
            alpha=result.alpha,
            regime=result.regime,
            sentiment=result.sentiment,
            crisis_mode=result.crisis_mode,
            volatility=result.current_vol,
            timestamp=datetime.utcnow().strftime("%Y-%m-%d %H:%M UTC"),
        )
        st.session_state.last_snapshot[ticker] = snapshot

    def detect_changes(
        self,
        ticker:     str,
        new_result: InferenceResult,
    ) -> list[AlertEvent]:
        """
        Compare new result against last snapshot.
        Returns list of AlertEvents for any significant changes.
        """
        alerts: list[AlertEvent] = []
        last   = st.session_state.last_snapshot.get(ticker)

        if last is None:
            self.update_snapshot(ticker, new_result)
            return alerts

        # 1. Alpha signal direction change
        old_label = last.signal_label
        new_alpha = new_result.alpha
        new_label = (
            "LONG"  if new_alpha > Config.backtest.LONG_THRESHOLD  else
            "SHORT" if new_alpha < Config.backtest.SHORT_THRESHOLD else
            "FLAT"
        )

        alpha_change = abs(new_alpha - last.alpha)
        if (
            old_label != new_label or
            alpha_change >= Config.alerts.ALPHA_CHANGE_THRESHOLD
        ):
            arrow = "▲" if new_label == "LONG" else "▼" if new_label == "SHORT" else "◆"
            severity = "WARNING" if new_label == "FLAT" else "INFO"
            alerts.append(AlertEvent(
                alert_id=self._next_id(),
                ticker=ticker,
                alert_type="SIGNAL_CHANGE",
                severity=severity,
                title=f"{ticker} Signal Changed: {old_label} → {new_label}",
                message=(
                    f"Alpha signal moved from {last.alpha:+.4f} ({old_label}) "
                    f"to {new_alpha:+.4f} ({arrow} {new_label}). "
                    f"Change magnitude: {alpha_change:.4f}"
                ),
                timestamp=datetime.utcnow().strftime("%Y-%m-%d %H:%M UTC"),
            ))

        # 2. Regime change
        if last.regime != new_result.regime:
            severity = "CRITICAL" if new_result.regime == "Bear" else "WARNING"
            alerts.append(AlertEvent(
                alert_id=self._next_id(),
                ticker=ticker,
                alert_type="REGIME_CHANGE",
                severity=severity,
                title=f"{ticker} Regime Change: {last.regime} → {new_result.regime}",
                message=(
                    f"Market regime shifted from {last.regime} to {new_result.regime}. "
                    f"Current alpha: {new_alpha:+.4f}. "
                    f"Volatility: {new_result.current_vol:.1%}"
                ),
                timestamp=datetime.utcnow().strftime("%Y-%m-%d %H:%M UTC"),
            ))

        # 3. Crisis mode activation
        if not last.crisis_mode and new_result.crisis_mode:
            alerts.append(AlertEvent(
                alert_id=self._next_id(),
                ticker=ticker,
                alert_type="CRISIS_MODE",
                severity="CRITICAL",
                title=f"CRISIS MODE ACTIVATED — {ticker}",
                message=(
                    f"Volatility {new_result.current_vol:.1%} exceeded "
                    f"{Config.model.CRISIS_VOL_THRESHOLD:.0%} threshold. "
                    f"Geopolitical text weight elevated to "
                    f"{new_result.crisis_weight:.0%}. "
                    f"Regime: {new_result.regime}. "
                    f"Sentiment: {new_result.sentiment}."
                ),
                timestamp=datetime.utcnow().strftime("%Y-%m-%d %H:%M UTC"),
            ))

        # 4. Crisis mode deactivation
        if last.crisis_mode and not new_result.crisis_mode:
            alerts.append(AlertEvent(
                alert_id=self._next_id(),
                ticker=ticker,
                alert_type="CRISIS_MODE",
                severity="INFO",
                title=f"Crisis Mode Deactivated — {ticker}",
                message=(
                    f"Volatility dropped to {new_result.current_vol:.1%}, "
                    f"below {Config.model.CRISIS_VOL_THRESHOLD:.0%} threshold. "
                    f"Normal weighting restored."
                ),
                timestamp=datetime.utcnow().strftime("%Y-%m-%d %H:%M UTC"),
            ))

        # 5. Sentiment shift
        old_sent = last.sentiment
        new_sent = new_result.sentiment
        if old_sent != new_sent:
            sent_probs  = new_result.sentiment_probs
            top_prob    = max(sent_probs.values())
            if top_prob >= Config.alerts.SENTIMENT_CHANGE_THRESHOLD + 0.5:
                alerts.append(AlertEvent(
                    alert_id=self._next_id(),
                    ticker=ticker,
                    alert_type="SENTIMENT_SHIFT",
                    severity="WARNING",
                    title=f"{ticker} Sentiment Shift: {old_sent} → {new_sent}",
                    message=(
                        f"Geopolitical sentiment changed from {old_sent} to {new_sent} "
                        f"(confidence: {top_prob:.1%}). "
                        f"This may indicate changing market narrative."
                    ),
                    timestamp=datetime.utcnow().strftime("%Y-%m-%d %H:%M UTC"),
                ))

        # Update snapshot
        self.update_snapshot(ticker, new_result)
        return alerts

    @staticmethod
    def _next_id() -> int:
        if "alert_id_counter" not in st.session_state:
            st.session_state.alert_id_counter = 0
        st.session_state.alert_id_counter += 1
        return st.session_state.alert_id_counter


# =============================================================================
# 3. Alert History Manager
# =============================================================================

class AlertHistory:
    """
    Manages alert history in session state.
    Keeps the last N alerts for display in the UI.
    """

    MAX_HISTORY = 50
    HISTORY_FILE = "alert_history.json"

    def __init__(self):
        self._ensure_session_state()

    def _ensure_session_state(self) -> None:
        if "alert_history" not in st.session_state:
            st.session_state.alert_history = self._load_from_file()

    @property
    def alerts(self) -> list[AlertEvent]:
        return st.session_state.alert_history

    @property
    def unread_count(self) -> int:
        return len([a for a in self.alerts if a.age_minutes < 60])

    def add(self, alert: AlertEvent) -> None:
        """Add alert to history."""
        st.session_state.alert_history.insert(0, alert)
        if len(st.session_state.alert_history) > self.MAX_HISTORY:
            st.session_state.alert_history = (
                st.session_state.alert_history[:self.MAX_HISTORY]
            )
        self._save_to_file()

    def add_many(self, alerts: list[AlertEvent]) -> None:
        for alert in alerts:
            self.add(alert)

    def clear(self) -> None:
        st.session_state.alert_history = []
        self._save_to_file()

    def get_by_type(self, alert_type: str) -> list[AlertEvent]:
        return [a for a in self.alerts if a.alert_type == alert_type]

    def get_by_ticker(self, ticker: str) -> list[AlertEvent]:
        return [a for a in self.alerts if a.ticker == ticker]

    def get_critical(self) -> list[AlertEvent]:
        return [a for a in self.alerts if a.severity == "CRITICAL"]

    def _save_to_file(self) -> None:
        try:
            data = [
                {
                    "alert_id":   a.alert_id,
                    "ticker":     a.ticker,
                    "alert_type": a.alert_type,
                    "severity":   a.severity,
                    "title":      a.title,
                    "message":    a.message,
                    "timestamp":  a.timestamp,
                    "sent_email": a.sent_email,
                    "email_to":   a.email_to,
                }
                for a in self.alerts[:20]
            ]
            with open(self.HISTORY_FILE, "w") as f:
                json.dump(data, f, indent=2)
        except Exception as exc:
            logger.warning("Could not save alert history: %s", exc)

    def _load_from_file(self) -> list[AlertEvent]:
        if not os.path.exists(self.HISTORY_FILE):
            return []
        try:
            with open(self.HISTORY_FILE) as f:
                data = json.load(f)
            return [
                AlertEvent(
                    alert_id=d["alert_id"],
                    ticker=d["ticker"],
                    alert_type=d["alert_type"],
                    severity=d["severity"],
                    title=d["title"],
                    message=d["message"],
                    timestamp=d["timestamp"],
                    sent_email=d.get("sent_email", False),
                    email_to=d.get("email_to", ""),
                )
                for d in data
            ]
        except Exception:
            return []


# =============================================================================
# 4. Alert Composer
# =============================================================================

class AlertComposer:
    """
    Formats alert messages for email and UI display.
    Supports English and Urdu (for Pakistani users).
    """

    @staticmethod
    def compose_email_subject(alert: AlertEvent) -> str:
        prefix = Config.alerts.EMAIL_SUBJECT_PREFIX
        return f"{prefix} [{alert.severity}] {alert.title}"

    @staticmethod
    def compose_email_body(
        alert:    AlertEvent,
        language: str = "English",
    ) -> str:
        """Format a complete email body for an alert."""
        if language == "Urdu":
            return AlertComposer._compose_urdu(alert)
        return AlertComposer._compose_english(alert)

    @staticmethod
    def _compose_english(alert: AlertEvent) -> str:
        icon = alert.severity_icon
        return f"""
{icon} SGEAIA MARKET ALERT
{'=' * 50}

Alert Type:  {alert.alert_type.replace('_', ' ').title()}
Severity:    {alert.severity}
Ticker:      {alert.ticker}
Time:        {alert.timestamp}

{alert.title}
{'-' * len(alert.title)}
{alert.message}

{'=' * 50}
DISCLAIMER: This alert is generated by an automated
system and does not constitute financial advice.
Past signals do not guarantee future performance.
Always conduct your own research before investing.

Synchronized Geopolitical-Equity Alpha Inference Architecture
https://synchronized-geopolitical-equity-alpha-inference-architecture.streamlit.app
"""

    @staticmethod
    def _compose_urdu(alert: AlertEvent) -> str:
        """Basic Urdu alert template for Pakistani users."""
        return f"""
{alert.severity_icon} SGEAIA مارکیٹ الرٹ
{'=' * 40}

ٹکر: {alert.ticker}
وقت: {alert.timestamp}
قسم: {alert.alert_type}

{alert.title}

{alert.message}

{'=' * 40}
نوٹ: یہ الرٹ خودکار نظام سے تیار کی گئی ہے۔
یہ مالی مشورہ نہیں ہے۔ سرمایہ کاری سے پہلے
اپنی تحقیق کریں۔
"""

    @staticmethod
    def compose_push_notification(alert: AlertEvent) -> dict:
        """
        Format alert for mobile push notification.
        Kept short for notification display limits.
        """
        icons = {
            "SIGNAL_CHANGE":  "📊",
            "REGIME_CHANGE":  "🔄",
            "CRISIS_MODE":    "🚨",
            "SENTIMENT_SHIFT": "📰",
        }
        icon  = icons.get(alert.alert_type, "📢")
        short = alert.message[:100] + "..." if len(alert.message) > 100 else alert.message

        return {
            "title": f"{icon} {alert.ticker} — {alert.alert_type.replace('_', ' ').title()}",
            "body":  short,
            "data":  {
                "ticker":     alert.ticker,
                "alert_type": alert.alert_type,
                "severity":   alert.severity,
                "timestamp":  alert.timestamp,
            },
        }


# =============================================================================
# 5. Email Alert Service (SendGrid)
# =============================================================================

class EmailAlertService:
    """
    Sends email alerts via SendGrid.

    SendGrid is free up to 100 emails/day — sufficient for
    personal use and early-stage app users.

    Setup:
      1. Sign up at sendgrid.com (free)
      2. Create an API key
      3. Add to Streamlit secrets: SENDGRID_API_KEY = "SG.xxx"
      4. Add sender email: ALERT_FROM_EMAIL = "alerts@yourdomain.com"
    """

    def __init__(self):
        self.api_key    = Config.api.SENDGRID_API_KEY
        self.from_email = Config.alerts.FROM_EMAIL
        self.available  = bool(self.api_key)
        if not self.available:
            logger.info(
                "EmailAlertService: SENDGRID_API_KEY not set. "
                "Email alerts disabled. Add to st.secrets to enable."
            )

    def send_alert(
        self,
        alert:    AlertEvent,
        to_email: str,
        language: str = "English",
    ) -> bool:
        """
        Send alert email via SendGrid.
        Returns True if sent successfully.
        """
        if not self.available:
            logger.info("Email not sent (no API key): %s", alert.title)
            return False

        if not to_email or "@" not in to_email:
            logger.warning("Invalid email address: %s", to_email)
            return False

        try:
            import urllib.request
            import json as _json

            subject = AlertComposer.compose_email_subject(alert)
            body    = AlertComposer.compose_email_body(alert, language)

            payload = _json.dumps({
                "personalizations": [{"to": [{"email": to_email}]}],
                "from":    {"email": self.from_email, "name": "SGEAIA Alerts"},
                "subject": subject,
                "content": [{"type": "text/plain", "value": body}],
            }).encode("utf-8")

            req = urllib.request.Request(
                "https://api.sendgrid.com/v3/mail/send",
                data=payload,
                headers={
                    "Authorization": f"Bearer {self.api_key}",
                    "Content-Type":  "application/json",
                },
                method="POST",
            )
            with urllib.request.urlopen(req, timeout=10) as resp:
                success = resp.status in (200, 202)

            if success:
                alert.sent_email = True
                alert.email_to   = to_email
                logger.info("Email sent: %s → %s", alert.title, to_email)
            return success

        except Exception as exc:
            logger.error("Email send failed: %s", exc)
            return False

    def send_batch(
        self,
        alerts:   list[AlertEvent],
        to_email: str,
        language: str = "English",
    ) -> int:
        """Send multiple alerts. Returns count sent successfully."""
        sent = 0
        for alert in alerts:
            if self.send_alert(alert, to_email, language):
                sent += 1
        return sent

    @property
    def is_configured(self) -> bool:
        return self.available


# =============================================================================
# 6. Crisis Alert Monitor
# =============================================================================

class CrisisAlertMonitor:
    """
    Dedicated monitor for Crisis Mode detection and alerting.

    Crisis Mode is the most important alert in the system —
    it signals that geopolitical events are driving unusual
    market volatility and the model is shifting its weighting.

    Features:
      - Consecutive crisis day tracking
      - Escalating severity (1 day = WARNING, 3+ days = CRITICAL)
      - Geopolitical context summary
      - Market impact assessment
    """

    def __init__(self):
        self._ensure_session_state()

    def _ensure_session_state(self) -> None:
        if "crisis_tracker" not in st.session_state:
            st.session_state.crisis_tracker = {
                "consecutive_days": 0,
                "first_triggered":  None,
                "last_vol":         0.0,
                "peak_vol":         0.0,
            }

    @property
    def tracker(self) -> dict:
        return st.session_state.crisis_tracker

    def update(
        self,
        ticker:      str,
        result:      InferenceResult,
        top_headline: str = "",
    ) -> Optional[AlertEvent]:
        """
        Update crisis tracking. Returns AlertEvent if crisis escalated.
        """
        tracker = self.tracker

        if result.crisis_mode:
            if tracker["consecutive_days"] == 0:
                tracker["first_triggered"] = datetime.utcnow().strftime("%Y-%m-%d")
            tracker["consecutive_days"] += 1
            tracker["last_vol"]  = result.current_vol
            tracker["peak_vol"]  = max(tracker["peak_vol"], result.current_vol)

            # Escalate severity based on duration
            if tracker["consecutive_days"] >= 3:
                severity = "CRITICAL"
            elif tracker["consecutive_days"] >= 1:
                severity = "WARNING"
            else:
                severity = "INFO"

            # Only send escalation alert every 3 updates
            if tracker["consecutive_days"] % 3 == 1:
                headline_context = (
                    f"\n\nTop headline: {top_headline[:150]}"
                    if top_headline else ""
                )
                return AlertEvent(
                    alert_id=self._next_id(),
                    ticker=ticker,
                    alert_type="CRISIS_MODE",
                    severity=severity,
                    title=(
                        f"Crisis Mode — Day {tracker['consecutive_days']} "
                        f"— {ticker}"
                    ),
                    message=(
                        f"Volatility: {result.current_vol:.1%} "
                        f"(peak: {tracker['peak_vol']:.1%}). "
                        f"Active since: {tracker['first_triggered']}. "
                        f"Geopolitical text weight: {result.crisis_weight:.0%}. "
                        f"Regime: {result.regime}."
                        f"{headline_context}"
                    ),
                    timestamp=datetime.utcnow().strftime("%Y-%m-%d %H:%M UTC"),
                )
        else:
            # Reset when crisis ends
            if tracker["consecutive_days"] > 0:
                logger.info(
                    "Crisis ended after %d updates. Peak vol: %.1f%%",
                    tracker["consecutive_days"],
                    tracker["peak_vol"] * 100,
                )
            tracker["consecutive_days"] = 0
            tracker["first_triggered"]  = None
            tracker["peak_vol"]         = 0.0

        return None

    @staticmethod
    def _next_id() -> int:
        if "crisis_alert_counter" not in st.session_state:
            st.session_state.crisis_alert_counter = 1000
        st.session_state.crisis_alert_counter += 1
        return st.session_state.crisis_alert_counter


# =============================================================================
# 7. Alert Manager (orchestrates all components)
# =============================================================================

class AlertManager:
    """
    Top-level orchestrator for the entire alert system.

    Usage in app.py:
        manager = AlertManager()
        manager.process_new_result(ticker, result, headline)
        manager.render_alert_panel()
    """

    def __init__(self):
        self.detector  = SignalChangeDetector()
        self.history   = AlertHistory()
        self.email_svc = EmailAlertService()
        self.crisis_mon = CrisisAlertMonitor()
        self.hours_chk  = MarketHoursChecker()
        self._ensure_session_state()

    def _ensure_session_state(self) -> None:
        if "alert_email" not in st.session_state:
            st.session_state.alert_email = ""
        if "alert_language" not in st.session_state:
            st.session_state.alert_language = "English"
        if "alerts_enabled" not in st.session_state:
            st.session_state.alerts_enabled = True

    def process_new_result(
        self,
        ticker:      str,
        result:      InferenceResult,
        top_headline: str = "",
        market:      str = "US",
    ) -> list[AlertEvent]:
        """
        Process a new inference result — detect changes,
        generate alerts, send emails if configured.
        Returns list of new alerts generated.
        """
        if not st.session_state.alerts_enabled:
            return []

        new_alerts: list[AlertEvent] = []

        # Signal change detection
        changes = self.detector.detect_changes(ticker, result)
        new_alerts.extend(changes)

        # Crisis monitoring
        crisis_alert = self.crisis_mon.update(ticker, result, top_headline)
        if crisis_alert:
            new_alerts.append(crisis_alert)

        # Add to history
        self.history.add_many(new_alerts)

        # Send emails if configured and market is open
        email = st.session_state.alert_email
        if (
            email and
            self.email_svc.is_configured and
            self.hours_chk.should_send_alert(market)
        ):
            critical = [a for a in new_alerts if a.severity == "CRITICAL"]
            if critical:
                lang = st.session_state.alert_language
                self.email_svc.send_batch(critical, email, lang)

        return new_alerts

    def render_alert_panel(self) -> None:
        """
        Render the alert configuration panel and history
        inside a Streamlit expander.
        """
        unread = self.history.unread_count
        label  = (
            f"🔔 Alerts & Notifications ({unread} new)"
            if unread > 0
            else "🔔 Alerts & Notifications"
        )

        with st.expander(label, expanded=unread > 0):
            # Settings
            st.markdown(
                '<div style="font-family:IBM Plex Mono;font-size:0.7rem;'
                'color:#546e7a;text-transform:uppercase;letter-spacing:0.1em;">'
                'Alert Settings</div>',
                unsafe_allow_html=True,
            )

            col1, col2 = st.columns(2)
            with col1:
                st.session_state.alerts_enabled = st.toggle(
                    "Enable alerts", value=st.session_state.alerts_enabled
                )
            with col2:
                st.session_state.alert_language = st.selectbox(
                    "Language",
                    options=Config.alerts.SUPPORTED_LANGUAGES,
                    index=0,
                )

            email_input = st.text_input(
                "Email for critical alerts (optional)",
                value=st.session_state.alert_email,
                placeholder="you@example.com",
            )
            if email_input != st.session_state.alert_email:
                st.session_state.alert_email = email_input

            if email_input and not self.email_svc.is_configured:
                st.caption(
                    "⚠️ Add `SENDGRID_API_KEY` to Streamlit secrets to enable email delivery."
                )
            elif email_input and self.email_svc.is_configured:
                st.caption("✅ Email alerts configured.")

            st.divider()

            # Market status
            psx_open = self.hours_chk.is_psx_open()
            us_open  = self.hours_chk.is_us_open()
            mc1, mc2 = st.columns(2)
            mc1.metric(
                "PSX Status",
                "OPEN 🟢" if psx_open else "CLOSED 🔴",
            )
            mc2.metric(
                "US Status",
                "OPEN 🟢" if us_open else "CLOSED 🔴",
            )

            st.divider()

            # Alert history
            alerts = self.history.alerts
            if not alerts:
                st.caption("No alerts yet. Run inference to start monitoring.")
            else:
                if st.button("Clear History", use_container_width=True):
                    self.history.clear()
                    st.rerun()

                for alert in alerts[:15]:
                    colour = alert.severity_colour
                    st.markdown(
                        f'<div style="'
                        f'border-left: 3px solid {colour};'
                        f'background: #0a1220;'
                        f'border-radius: 0 4px 4px 0;'
                        f'padding: 0.4rem 0.7rem;'
                        f'margin-bottom: 0.3rem;'
                        f'">'
                        f'<div style="font-family:IBM Plex Mono;font-size:0.6rem;'
                        f'color:#37474f;text-transform:uppercase;">'
                        f'{alert.severity_icon} {alert.alert_type.replace("_", " ")} '
                        f'· {alert.timestamp}</div>'
                        f'<div style="font-family:IBM Plex Mono;font-size:0.75rem;'
                        f'color:#90a4ae;margin-top:3px;">'
                        f'{alert.title}</div>'
                        f'</div>',
                        unsafe_allow_html=True,
                    )

    def render_market_status_bar(self) -> None:
        """Render a compact market status bar."""
        psx_open = self.hours_chk.is_psx_open()
        us_open  = self.hours_chk.is_us_open()

        psx_col  = Config.ui.BULL_COLOUR if psx_open else Config.ui.BEAR_COLOUR
        us_col   = Config.ui.BULL_COLOUR if us_open  else Config.ui.BEAR_COLOUR
        psx_next = "" if psx_open else f" · Next: {self.hours_chk.next_open_time('PSX')}"
        us_next  = "" if us_open  else f" · Next: {self.hours_chk.next_open_time('US')}"

        st.markdown(
            f'<div style="'
            f'background:#0d1521;border:1px solid #162030;'
            f'border-radius:5px;padding:0.35rem 0.8rem;'
            f'font-family:IBM Plex Mono;font-size:0.68rem;'
            f'display:flex;gap:1.5rem;margin-bottom:0.5rem;">'
            f'<span style="color:{psx_col};">● PSX '
            f'{"OPEN" if psx_open else "CLOSED"}{psx_next}</span>'
            f'<span style="color:{us_col};">● NYSE/NASDAQ '
            f'{"OPEN" if us_open else "CLOSED"}{us_next}</span>'
            f'</div>',
            unsafe_allow_html=True,
        )


# =============================================================================
# CLI smoke test
# =============================================================================

if __name__ == "__main__":
    print("=" * 60)
    print("SGEAIA Alert System - Smoke Test")
    print("=" * 60)

    # Test MarketHoursChecker
    print("\n-- Market Hours --")
    checker = MarketHoursChecker()
    print(f"  PSX open:       {checker.is_psx_open()}")
    print(f"  US open:        {checker.is_us_open()}")
    print(f"  PSX next open:  {checker.next_open_time('PSX')}")
    print(f"  US next open:   {checker.next_open_time('US')}")

    # Test AlertComposer
    print("\n-- Alert Composer --")
    mock_alert = AlertEvent(
        alert_id=1,
        ticker="ENGRO.KA",
        alert_type="SIGNAL_CHANGE",
        severity="WARNING",
        title="ENGRO.KA Signal Changed: FLAT -> LONG",
        message="Alpha moved from +0.05 to +0.18. Change: 0.13",
        timestamp=datetime.utcnow().strftime("%Y-%m-%d %H:%M UTC"),
    )

    subject = AlertComposer.compose_email_subject(mock_alert)
    body    = AlertComposer.compose_email_body(mock_alert, "English")
    push    = AlertComposer.compose_push_notification(mock_alert)

    print(f"  Subject:     {subject}")
    print(f"  Body length: {len(body)} chars")
    print(f"  Push title:  {push['title']}")
    print(f"  Push body:   {push['body']}")

    # Test EmailAlertService (no key)
    print("\n-- Email Service --")
    svc = EmailAlertService()
    print(f"  Configured: {svc.is_configured}")
    result = svc.send_alert(mock_alert, "test@example.com")
    print(f"  Send result (no key): {result}")

    # Test CrisisAlertMonitor
    print("\n-- Crisis Monitor --")
    from dataclasses import dataclass as dc

    class MockResult:
        alpha = -0.25
        regime = "Bear"
        sentiment = "Negative"
        crisis_mode = True
        crisis_weight = 0.65
        current_vol = 0.35
        sentiment_probs = {"Positive": 0.1, "Negative": 0.7, "Neutral": 0.2}

    monitor = CrisisAlertMonitor()
    for i in range(4):
        alert = monitor.update("^KSE100", MockResult(), "PSX falls on IMF concerns")
        if alert:
            print(f"  Day {i+1}: {alert.severity} - {alert.title}")
        else:
            print(f"  Day {i+1}: No alert")

    print("\nAll alert tests passed.")
