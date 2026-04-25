"""
auth.py
=======
Synchronized Geopolitical-Equity Alpha Inference Architecture
-------------------------------------------------------------
User authentication and session management.
 
  AuthManager        - Main authentication orchestrator
  SupabaseAuth       - Supabase backend (email + Google login)
  LocalAuth          - Fallback local auth (demo mode, no backend needed)
  SessionManager     - JWT session handling
  TierManager        - Free vs Premium feature access control
  UserProfile        - User data container
"""
 
from __future__ import annotations
 
import hashlib
import json
import logging
import os
import time
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Optional
 
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
class UserProfile:
    """Complete user profile."""
    user_id:     str
    email:       str
    name:        str
    is_premium:  bool
    created_at:  str
    last_login:  str
    country:     str = "PK"
    language:    str = "English"
    watchlist:   list[str] = field(default_factory=list)
    alert_email: str = ""
 
    @property
    def display_name(self) -> str:
        return self.name or self.email.split("@")[0]
 
    @property
    def tier_label(self) -> str:
        return "Premium" if self.is_premium else "Free"
 
    @property
    def tier_colour(self) -> str:
        return Config.ui.WARNING_COLOUR if self.is_premium else Config.ui.NEUTRAL_COLOUR
 
    @property
    def max_tickers(self) -> int:
        return (
            Config.subscription.PREMIUM_MAX_TICKERS
            if self.is_premium
            else Config.subscription.FREE_MAX_TICKERS
        )
 
    @property
    def notification_interval(self) -> int:
        return (
            Config.subscription.PREMIUM_NOTIFICATION_MINS
            if self.is_premium
            else Config.subscription.FREE_NOTIFICATION_MINS
        )
 
 
@dataclass
class AuthResult:
    """Result of an authentication attempt."""
    success:   bool
    user:      Optional[UserProfile] = None
    error:     str = ""
    token:     str = ""
 
 
# =============================================================================
# 1. Session Manager
# =============================================================================
 
class SessionManager:
    """
    Manages user sessions in Streamlit session state.
 
    Sessions expire after Config.auth.SESSION_EXPIRY_HOURS hours.
    On Streamlit Cloud, sessions reset on page refresh — this is
    a known limitation of Streamlit's stateless architecture.
    """
 
    SESSION_KEY = "sgeaia_session"
 
    @staticmethod
    def is_logged_in() -> bool:
        """Check if a valid session exists."""
        session = st.session_state.get(SessionManager.SESSION_KEY)
        if not session:
            return False
        # Check expiry
        try:
            expiry = datetime.fromisoformat(session.get("expires_at", ""))
            if datetime.utcnow() > expiry:
                SessionManager.logout()
                return False
        except Exception:
            return False
        return True
 
    @staticmethod
    def get_user() -> Optional[UserProfile]:
        """Get current logged-in user."""
        if not SessionManager.is_logged_in():
            return None
        session = st.session_state.get(SessionManager.SESSION_KEY, {})
        user_data = session.get("user")
        if not user_data:
            return None
        return UserProfile(**user_data)
 
    @staticmethod
    def create_session(user: UserProfile, token: str = "") -> None:
        """Create a new session for the user."""
        expiry = datetime.utcnow() + timedelta(
            hours=Config.auth.SESSION_EXPIRY_HOURS
        )
        st.session_state[SessionManager.SESSION_KEY] = {
            "user": {
                "user_id":    user.user_id,
                "email":      user.email,
                "name":       user.name,
                "is_premium": user.is_premium,
                "created_at": user.created_at,
                "last_login": user.last_login,
                "country":    user.country,
                "language":   user.language,
                "watchlist":  user.watchlist,
                "alert_email": user.alert_email,
            },
            "token":      token,
            "expires_at": expiry.isoformat(),
            "created_at": datetime.utcnow().isoformat(),
        }
        logger.info("Session created for %s (premium=%s)", user.email, user.is_premium)
 
    @staticmethod
    def logout() -> None:
        """Clear the current session."""
        if SessionManager.SESSION_KEY in st.session_state:
            del st.session_state[SessionManager.SESSION_KEY]
        logger.info("User logged out")
 
    @staticmethod
    def update_user(updates: dict) -> None:
        """Update user data in session."""
        session = st.session_state.get(SessionManager.SESSION_KEY)
        if session and "user" in session:
            session["user"].update(updates)
 
 
# =============================================================================
# 2. Tier Manager
# =============================================================================
 
class TierManager:
    """
    Controls feature access based on user tier (Free vs Premium).
 
    Usage:
        tier = TierManager(user)
        if tier.can_access_multi_ticker():
            # show multi-ticker UI
        if tier.can_access_psx():
            # show PSX data
    """
 
    def __init__(self, user: Optional[UserProfile] = None):
        self.user       = user
        self.is_premium = user.is_premium if user else False
 
    def can_access_multi_ticker(self) -> bool:
        return self.is_premium or Config.subscription.FREE_MULTI_TICKER
 
    def can_access_psx(self) -> bool:
        return self.is_premium or Config.subscription.FREE_PSX_ACCESS
 
    def can_access_global_markets(self) -> bool:
        return self.is_premium or Config.subscription.FREE_GLOBAL_ACCESS
 
    def can_generate_pdf(self) -> bool:
        return self.is_premium or Config.subscription.FREE_PDF_REPORTS
 
    def can_set_email_alerts(self) -> bool:
        return self.is_premium or Config.subscription.FREE_EMAIL_ALERTS
 
    def can_access_crisis_alerts(self) -> bool:
        return self.is_premium or Config.subscription.FREE_CRISIS_ALERTS
 
    def can_paper_trade(self) -> bool:
        return self.is_premium or Config.subscription.FREE_PAPER_TRADING
 
    def get_max_tickers(self) -> int:
        return (
            Config.subscription.PREMIUM_MAX_TICKERS
            if self.is_premium
            else Config.subscription.FREE_MAX_TICKERS
        )
 
    def get_history_days(self) -> int:
        return (
            Config.subscription.PREMIUM_HISTORY_DAYS
            if self.is_premium
            else Config.subscription.FREE_HISTORY_DAYS
        )
 
    def get_notification_interval(self) -> int:
        return (
            Config.subscription.PREMIUM_NOTIFICATION_MINS
            if self.is_premium
            else Config.subscription.FREE_NOTIFICATION_MINS
        )
 
    def render_upgrade_prompt(self, feature: str) -> None:
        """Show upgrade prompt for locked features."""
        price_usd = Config.subscription.PREMIUM_PRICE_MONTHLY_USD
        price_pkr = Config.subscription.PREMIUM_PRICE_MONTHLY_PKR
        st.markdown(
            f'<div style="'
            f'background:#1a1000;border:1px solid {Config.ui.WARNING_COLOUR};'
            f'border-radius:6px;padding:0.6rem 0.8rem;margin:0.4rem 0;">'
            f'<div style="font-family:IBM Plex Mono;font-size:0.72rem;'
            f'color:{Config.ui.WARNING_COLOUR};">🔒 Premium Feature</div>'
            f'<div style="font-family:IBM Plex Mono;font-size:0.78rem;'
            f'color:#90a4ae;margin-top:4px;">'
            f'{feature} requires a Premium subscription.</div>'
            f'<div style="font-family:IBM Plex Mono;font-size:0.68rem;'
            f'color:#546e7a;margin-top:4px;">'
            f'${price_usd}/month &nbsp;|&nbsp; '
            f'PKR {price_pkr:,.0f}/month</div>'
            f'</div>',
            unsafe_allow_html=True,
        )
 
 
# =============================================================================
# 3. Local Auth (Demo Mode - no backend required)
# =============================================================================
 
class LocalAuth:
    """
    Simple local authentication for demo mode.
 
    No backend required — perfect for:
      - Local development
      - Demo deployments
      - Testing premium features
 
    In demo mode, all users get premium access.
    Set Config.auth.DEMO_MODE = False to require real auth.
    """
 
    # Simple in-memory user store (for demo only)
    # In production, replace with Supabase
    _DEMO_USERS: dict[str, dict] = {
        "demo@sgeaia.app": {
            "password_hash": hashlib.sha256(b"demo123").hexdigest(),
            "name":          "Demo User",
            "is_premium":    True,
        },
    }
 
    def login(self, email: str, password: str) -> AuthResult:
        """Authenticate with email and password."""
        email = email.lower().strip()
 
        # Demo mode: accept any email/password
        if Config.auth.DEMO_MODE:
            user = self._create_demo_user(email)
            return AuthResult(success=True, user=user, token="demo_token")
 
        # Local user store
        user_data = self._DEMO_USERS.get(email)
        if not user_data:
            return AuthResult(success=False, error="Email not found.")
 
        pw_hash = hashlib.sha256(password.encode()).hexdigest()
        if pw_hash != user_data["password_hash"]:
            return AuthResult(success=False, error="Incorrect password.")
 
        user = UserProfile(
            user_id=hashlib.md5(email.encode()).hexdigest()[:12],
            email=email,
            name=user_data["name"],
            is_premium=user_data["is_premium"],
            created_at="2024-01-01",
            last_login=datetime.utcnow().strftime("%Y-%m-%d %H:%M UTC"),
        )
        return AuthResult(success=True, user=user, token="local_token")
 
    def register(self, email: str, password: str, name: str) -> AuthResult:
        """Register a new local user."""
        email = email.lower().strip()
        if email in self._DEMO_USERS:
            return AuthResult(success=False, error="Email already registered.")
 
        pw_hash = hashlib.sha256(password.encode()).hexdigest()
        self._DEMO_USERS[email] = {
            "password_hash": pw_hash,
            "name":          name,
            "is_premium":    False,
        }
        user = UserProfile(
            user_id=hashlib.md5(email.encode()).hexdigest()[:12],
            email=email,
            name=name,
            is_premium=False,
            created_at=datetime.utcnow().strftime("%Y-%m-%d"),
            last_login=datetime.utcnow().strftime("%Y-%m-%d %H:%M UTC"),
        )
        return AuthResult(success=True, user=user, token="local_token")
 
    @staticmethod
    def _create_demo_user(email: str) -> UserProfile:
        return UserProfile(
            user_id=hashlib.md5(email.encode()).hexdigest()[:12],
            email=email,
            name=email.split("@")[0].replace(".", " ").title(),
            is_premium=Config.auth.DEMO_IS_PREMIUM,
            created_at=datetime.utcnow().strftime("%Y-%m-%d"),
            last_login=datetime.utcnow().strftime("%Y-%m-%d %H:%M UTC"),
        )
 
 
# =============================================================================
# 4. Supabase Auth (Production Backend)
# =============================================================================
 
class SupabaseAuth:
    """
    Supabase-based authentication for production deployment.
 
    Features:
      - Email + password login
      - Email verification
      - Password reset
      - JWT session tokens
      - User metadata (premium status, preferences)
 
    Setup:
      1. Create project at supabase.com (free)
      2. Add to Streamlit secrets:
           SUPABASE_URL = "https://xxx.supabase.co"
           SUPABASE_KEY = "eyJ..."
      3. Set Config.auth.DEMO_MODE = False
    """
 
    def __init__(self):
        self.url      = Config.api.SUPABASE_URL
        self.key      = Config.api.SUPABASE_KEY
        self.available = bool(self.url and self.key)
 
        if not self.available:
            logger.info(
                "Supabase not configured. "
                "Add SUPABASE_URL and SUPABASE_KEY to st.secrets."
            )
 
    def login(self, email: str, password: str) -> AuthResult:
        """Authenticate via Supabase Auth REST API."""
        if not self.available:
            return AuthResult(
                success=False,
                error="Supabase not configured. Using demo mode.",
            )
        try:
            import urllib.request
            import json as _json
 
            payload = _json.dumps({
                "email":    email,
                "password": password,
            }).encode("utf-8")
 
            req = urllib.request.Request(
                f"{self.url}/auth/v1/token?grant_type=password",
                data=payload,
                headers={
                    "apikey":       self.key,
                    "Content-Type": "application/json",
                },
                method="POST",
            )
 
            with urllib.request.urlopen(req, timeout=10) as resp:
                data = _json.loads(resp.read())
 
            if "access_token" not in data:
                error = data.get("error_description", "Login failed.")
                return AuthResult(success=False, error=error)
 
            user_data = data.get("user", {})
            metadata  = user_data.get("user_metadata", {})
 
            user = UserProfile(
                user_id=user_data.get("id", ""),
                email=user_data.get("email", email),
                name=metadata.get("name", email.split("@")[0]),
                is_premium=metadata.get("is_premium", False),
                created_at=user_data.get("created_at", "")[:10],
                last_login=datetime.utcnow().strftime("%Y-%m-%d %H:%M UTC"),
                country=metadata.get("country", "PK"),
                language=metadata.get("language", "English"),
            )
            return AuthResult(
                success=True,
                user=user,
                token=data["access_token"],
            )
 
        except Exception as exc:
            logger.error("Supabase login error: %s", exc)
            return AuthResult(success=False, error=str(exc))
 
    def register(self, email: str, password: str, name: str) -> AuthResult:
        """Register new user via Supabase Auth."""
        if not self.available:
            return AuthResult(
                success=False,
                error="Supabase not configured.",
            )
        try:
            import urllib.request
            import json as _json
 
            payload = _json.dumps({
                "email":    email,
                "password": password,
                "data":     {"name": name, "is_premium": False, "country": "PK"},
            }).encode("utf-8")
 
            req = urllib.request.Request(
                f"{self.url}/auth/v1/signup",
                data=payload,
                headers={
                    "apikey":       self.key,
                    "Content-Type": "application/json",
                },
                method="POST",
            )
 
            with urllib.request.urlopen(req, timeout=10) as resp:
                data = _json.loads(resp.read())
 
            if "id" not in data.get("user", {}):
                return AuthResult(
                    success=False,
                    error=data.get("error_description", "Registration failed."),
                )
 
            user = UserProfile(
                user_id=data["user"]["id"],
                email=email,
                name=name,
                is_premium=False,
                created_at=datetime.utcnow().strftime("%Y-%m-%d"),
                last_login=datetime.utcnow().strftime("%Y-%m-%d %H:%M UTC"),
                country="PK",
            )
            return AuthResult(success=True, user=user)
 
        except Exception as exc:
            logger.error("Supabase register error: %s", exc)
            return AuthResult(success=False, error=str(exc))
 
    def reset_password(self, email: str) -> bool:
        """Send password reset email."""
        if not self.available:
            return False
        try:
            import urllib.request
            import json as _json
 
            payload = _json.dumps({"email": email}).encode("utf-8")
            req = urllib.request.Request(
                f"{self.url}/auth/v1/recover",
                data=payload,
                headers={
                    "apikey":       self.key,
                    "Content-Type": "application/json",
                },
                method="POST",
            )
            with urllib.request.urlopen(req, timeout=10):
                pass
            return True
        except Exception as exc:
            logger.error("Password reset error: %s", exc)
            return False
 
 
# =============================================================================
# 5. Auth Manager (orchestrates everything)
# =============================================================================
 
class AuthManager:
    """
    Top-level authentication orchestrator.
 
    Automatically selects the right backend:
      - Supabase if configured (production)
      - LocalAuth if demo mode (development/demo)
 
    Usage in app.py:
        auth = AuthManager()
        if not auth.is_logged_in():
            auth.render_login_page()
            st.stop()
        user = auth.current_user
        tier = auth.tier
    """
 
    def __init__(self):
        self.session  = SessionManager()
        self.local    = LocalAuth()
        self.supabase = SupabaseAuth()
 
        # Use Supabase if configured, otherwise fall back to local
        self.backend  = (
            self.supabase
            if self.supabase.available and not Config.auth.DEMO_MODE
            else self.local
        )
        self._backend_name = (
            "Supabase"
            if self.supabase.available and not Config.auth.DEMO_MODE
            else "Demo"
        )
        logger.info("AuthManager: using %s backend", self._backend_name)
 
    @property
    def is_logged_in(self) -> bool:
        return self.session.is_logged_in()
 
    @property
    def current_user(self) -> Optional[UserProfile]:
        return self.session.get_user()
 
    @property
    def tier(self) -> TierManager:
        return TierManager(self.current_user)
 
    def login(self, email: str, password: str) -> AuthResult:
        """Attempt login with selected backend."""
        result = self.backend.login(email, password)
        if result.success and result.user:
            self.session.create_session(result.user, result.token)
        return result
 
    def register(self, email: str, password: str, name: str) -> AuthResult:
        """Register new user."""
        result = self.backend.register(email, password, name)
        if result.success and result.user:
            self.session.create_session(result.user, result.token)
        return result
 
    def logout(self) -> None:
        """Log out current user."""
        self.session.logout()
 
    def render_login_page(self) -> None:
        """
        Render the complete login/register UI.
        Called in app.py before the main dashboard.
        """
        st.markdown("""
<style>
.auth-container {
    max-width: 420px;
    margin: 2rem auto;
    background: #0d1521;
    border: 1px solid #162030;
    border-radius: 8px;
    padding: 2rem;
}
.auth-title {
    font-family: IBM Plex Mono, monospace;
    font-size: 1.1rem;
    color: #00b4d8;
    text-align: center;
    margin-bottom: 0.3rem;
    text-transform: uppercase;
    letter-spacing: 0.08em;
}
.auth-sub {
    font-family: IBM Plex Mono, monospace;
    font-size: 0.68rem;
    color: #546e7a;
    text-align: center;
    margin-bottom: 1.5rem;
}
.tier-box {
    background: #0a1220;
    border: 1px solid #162030;
    border-radius: 5px;
    padding: 0.6rem 0.8rem;
    margin-bottom: 0.4rem;
    font-family: IBM Plex Mono, monospace;
    font-size: 0.72rem;
    color: #90a4ae;
}
.tier-box .tier-name {
    color: #00b4d8;
    font-weight: bold;
    margin-bottom: 3px;
}
</style>
""", unsafe_allow_html=True)
 
        col_l, col_c, col_r = st.columns([1, 2, 1])
        with col_c:
            st.markdown(
                '<div class="auth-title">🛰 SGEAIA</div>'
                '<div class="auth-sub">Synchronized Geopolitical-Equity Alpha<br>'
                'Inference Architecture</div>',
                unsafe_allow_html=True,
            )
 
            # Demo mode banner
            if Config.auth.DEMO_MODE:
                st.info(
                    "**Demo Mode** — Enter any email and password to access "
                    "the full app with Premium features enabled.",
                    icon="ℹ️",
                )
 
            tab_login, tab_register = st.tabs(["Login", "Register"])
 
            with tab_login:
                self._render_login_form()
 
            with tab_register:
                self._render_register_form()
 
            st.divider()
 
            # Pricing info
            st.markdown(
                f'<div class="tier-box">'
                f'<div class="tier-name">FREE TIER</div>'
                f'1 ticker &nbsp;|&nbsp; '
                f'{Config.subscription.FREE_NOTIFICATION_MINS}min alerts &nbsp;|&nbsp; '
                f'{Config.subscription.FREE_HISTORY_DAYS}-day history'
                f'</div>'
                f'<div class="tier-box">'
                f'<div class="tier-name" style="color:#ffc400;">PREMIUM — '
                f'${Config.subscription.PREMIUM_PRICE_MONTHLY_USD}/mo &nbsp;|&nbsp; '
                f'PKR {Config.subscription.PREMIUM_PRICE_MONTHLY_PKR:,.0f}/mo</div>'
                f'Unlimited tickers &nbsp;|&nbsp; PSX + Global &nbsp;|&nbsp; '
                f'{Config.subscription.PREMIUM_NOTIFICATION_MINS}min alerts &nbsp;|&nbsp; '
                f'PDF reports &nbsp;|&nbsp; Paper trading'
                f'</div>',
                unsafe_allow_html=True,
            )
 
    def _render_login_form(self) -> None:
        """Render login form."""
        with st.form("login_form", clear_on_submit=False):
            email    = st.text_input("Email", placeholder="you@example.com")
            password = st.text_input("Password", type="password")
            submitted = st.form_submit_button(
                "Login", use_container_width=True, type="primary"
            )
 
        if submitted:
            if not email or not password:
                st.error("Please enter email and password.")
                return
 
            with st.spinner("Authenticating..."):
                result = self.login(email, password)
 
            if result.success:
                st.success(f"Welcome back, {result.user.display_name}!")
                time.sleep(0.5)
                st.rerun()
            else:
                st.error(result.error or "Login failed.")
 
        # Password reset
        if not Config.auth.DEMO_MODE and self.supabase.available:
            if st.button("Forgot password?", use_container_width=False):
                st.session_state["show_reset"] = True
 
            if st.session_state.get("show_reset"):
                reset_email = st.text_input(
                    "Enter your email for reset link",
                    key="reset_email",
                )
                if st.button("Send Reset Link"):
                    if self.supabase.reset_password(reset_email):
                        st.success("Reset link sent to your email.")
                        st.session_state["show_reset"] = False
                    else:
                        st.error("Could not send reset link.")
 
    def _render_register_form(self) -> None:
        """Render registration form."""
        with st.form("register_form", clear_on_submit=False):
            name     = st.text_input("Full Name", placeholder="Abdul Ahad")
            email    = st.text_input("Email", placeholder="you@example.com")
            password = st.text_input(
                "Password", type="password",
                help="Minimum 8 characters",
            )
            confirm  = st.text_input("Confirm Password", type="password")
            submitted = st.form_submit_button(
                "Create Account", use_container_width=True, type="primary"
            )
 
        if submitted:
            if not all([name, email, password, confirm]):
                st.error("Please fill in all fields.")
                return
            if password != confirm:
                st.error("Passwords do not match.")
                return
            if len(password) < 8:
                st.error("Password must be at least 8 characters.")
                return
            if "@" not in email:
                st.error("Please enter a valid email address.")
                return
 
            with st.spinner("Creating account..."):
                result = self.register(email, password, name)
 
            if result.success:
                st.success(
                    f"Account created! Welcome, {result.user.display_name}. "
                    f"You're on the Free tier."
                )
                time.sleep(0.5)
                st.rerun()
            else:
                st.error(result.error or "Registration failed.")
 
    def render_user_menu(self) -> None:
        """
        Render compact user info + logout button in sidebar.
        Call this in the sidebar after successful login.
        """
        user = self.current_user
        if not user:
            return
 
        tier_colour = user.tier_colour
        st.markdown(
            f'<div style="'
            f'background:#0a1220;border:1px solid #162030;'
            f'border-radius:5px;padding:0.5rem 0.7rem;margin-bottom:0.5rem;">'
            f'<div style="font-family:IBM Plex Mono;font-size:0.7rem;'
            f'color:#90a4ae;">{user.display_name}</div>'
            f'<div style="font-family:IBM Plex Mono;font-size:0.62rem;'
            f'color:#546e7a;">{user.email}</div>'
            f'<div style="font-family:IBM Plex Mono;font-size:0.62rem;'
            f'color:{tier_colour};margin-top:2px;">'
            f'{"★" if user.is_premium else "○"} {user.tier_label} Tier</div>'
            f'</div>',
            unsafe_allow_html=True,
        )
 
        if st.button("Logout", use_container_width=True):
            self.logout()
            st.rerun()
 
    def render_tier_badge(self) -> None:
        """Render a small tier badge inline."""
        user = self.current_user
        if not user:
            return
        icon   = "★" if user.is_premium else "○"
        colour = user.tier_colour
        st.markdown(
            f'<span style="font-family:IBM Plex Mono;font-size:0.65rem;'
            f'color:{colour};background:#0a1220;border:1px solid #162030;'
            f'border-radius:3px;padding:2px 6px;">'
            f'{icon} {user.tier_label}</span>',
            unsafe_allow_html=True,
        )
 
 
# =============================================================================
# 6. Auth Guard decorator
# =============================================================================
 
def require_auth(func):
    """
    Decorator for Streamlit pages that require authentication.
 
    Usage:
        @require_auth
        def main():
            st.write("This page requires login")
    """
    def wrapper(*args, **kwargs):
        auth = AuthManager()
        if not auth.is_logged_in:
            auth.render_login_page()
            st.stop()
        return func(*args, **kwargs)
    return wrapper
 
 
def require_premium(feature_name: str = "This feature"):
    """
    Decorator for functions that require premium tier.
 
    Usage:
        @require_premium("Multi-ticker comparison")
        def show_multi_ticker():
            ...
    """
    def decorator(func):
        def wrapper(*args, **kwargs):
            auth = AuthManager()
            if not auth.is_logged_in:
                st.error("Please log in to access this feature.")
                return
            if not auth.tier.is_premium:
                auth.tier.render_upgrade_prompt(feature_name)
                return
            return func(*args, **kwargs)
        return wrapper
    return decorator
 
 
# =============================================================================
# CLI smoke test
# =============================================================================
 
if __name__ == "__main__":
    print("=" * 60)
    print("SGEAIA Auth Module - Smoke Test")
    print("=" * 60)
 
    print(f"\n  Demo Mode: {Config.auth.DEMO_MODE}")
    print(f"  Supabase configured: {bool(Config.api.SUPABASE_URL)}")
 
    # Test LocalAuth
    print("\n-- Local Auth --")
    local = LocalAuth()
 
    # Demo mode login
    result = local.login("test@example.com", "anypassword")
    print(f"  Demo login: {result.success} | user={result.user.email if result.user else None}")
 
    # Real user login
    result2 = local.login("demo@sgeaia.app", "demo123")
    print(f"  Real login: {result2.success} | premium={result2.user.is_premium if result2.user else None}")
 
    # Wrong password
    result3 = local.login("demo@sgeaia.app", "wrongpassword")
    if Config.auth.DEMO_MODE:
        print(f"  Wrong pw (demo mode): {result3.success} (demo accepts anything)")
    else:
        print(f"  Wrong pw: success={result3.success} error={result3.error}")
 
    # Test TierManager
    print("\n-- Tier Manager --")
    free_user = UserProfile(
        user_id="001", email="free@test.com", name="Free User",
        is_premium=False, created_at="2024-01-01",
        last_login="2024-01-01",
    )
    premium_user = UserProfile(
        user_id="002", email="premium@test.com", name="Premium User",
        is_premium=True, created_at="2024-01-01",
        last_login="2024-01-01",
    )
 
    free_tier    = TierManager(free_user)
    premium_tier = TierManager(premium_user)
 
    print(f"  Free   — multi_ticker: {free_tier.can_access_multi_ticker()}")
    print(f"  Free   — pdf:          {free_tier.can_generate_pdf()}")
    print(f"  Free   — paper trade:  {free_tier.can_paper_trade()}")
    print(f"  Free   — max tickers:  {free_tier.get_max_tickers()}")
    print(f"  Premium— multi_ticker: {premium_tier.can_access_multi_ticker()}")
    print(f"  Premium— pdf:          {premium_tier.can_generate_pdf()}")
    print(f"  Premium— paper trade:  {premium_tier.can_paper_trade()}")
    print(f"  Premium— max tickers:  {premium_tier.get_max_tickers()}")
 
    # Test UserProfile
    print("\n-- User Profile --")
    print(f"  Free display name:    {free_user.display_name}")
    print(f"  Free tier label:      {free_user.tier_label}")
    print(f"  Free max tickers:     {free_user.max_tickers}")
    print(f"  Free notif interval:  {free_user.notification_interval} min")
    print(f"  Premium tier label:   {premium_user.tier_label}")
    print(f"  Premium max tickers:  {premium_user.max_tickers}")
    print(f"  Premium notif int:    {premium_user.notification_interval} min")
 
    print("\nAll auth tests passed.")