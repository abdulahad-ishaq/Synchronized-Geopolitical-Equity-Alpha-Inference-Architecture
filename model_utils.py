"""
model_utils.py
==============
Synchronized Geopolitical-Equity Alpha Inference Architecture
--------------------------------------------------------------
OOP model components:

  BiLSTMEncoder         – 60-day OHLCV numeric encoding
  FinBERTEncoder        – yiyanghkust/finbert-tone text encoding (cached)
  WeightedFusionLayer   – Cross-modal attention with Crisis Mode weighting
  AlphaInferenceModel   – End-to-end orchestrator
  PriceOnlyBaseline     – Ablation baseline (no text signal)
  BacktestEngine        – Sharpe, Max Drawdown, RMSE, Ablation Study
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Optional

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import streamlit as st
from sklearn.preprocessing import StandardScaler
from transformers import AutoTokenizer, AutoModelForSequenceClassification

logger = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s – %(message)s",
)

DEVICE          = torch.device("cuda" if torch.cuda.is_available() else "cpu")
FINBERT_MODEL   = "yiyanghkust/finbert-tone"   # spec: finbert-tone
CRISIS_VOL_THRESHOLD = 0.30   # annualised vol above this → Crisis Mode


# ─────────────────────────────────────────────────────────────────────────────
# Result dataclass
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class InferenceResult:
    alpha:           float          # directional signal in [-1, +1]
    confidence:      float          # model confidence in [ 0,  1]
    regime:          str            # "Bull" | "Bear" | "Neutral"
    regime_probs:    dict           # {"Bull": f, "Bear": f, "Neutral": f}
    sentiment:       str            # "Positive" | "Negative" | "Neutral"
    sentiment_probs: dict           # {"Positive": f, "Negative": f, "Neutral": f}
    crisis_mode:     bool           # True when volatility > threshold
    crisis_weight:   float          # text-stream weight in [0, 1]
    current_vol:     float          # realised annualised volatility

    @property
    def signal_label(self) -> str:
        if self.alpha > 0.1:  return "▲ LONG"
        if self.alpha < -0.1: return "▼ SHORT"
        return "◆ FLAT"


@dataclass
class BacktestResult:
    sharpe_ratio:    float
    max_drawdown:    float
    rmse:            float
    annualised_return: float
    total_return:    float
    n_trades:        int
    equity_curve:    pd.Series


@dataclass
class AblationResult:
    hybrid:     BacktestResult
    price_only: BacktestResult

    @property
    def sharpe_improvement(self) -> float:
        return self.hybrid.sharpe_ratio - self.price_only.sharpe_ratio

    @property
    def rmse_improvement(self) -> float:
        return self.price_only.rmse - self.hybrid.rmse   # positive = hybrid is better


# ─────────────────────────────────────────────────────────────────────────────
# 1. Bi-LSTM Encoder
# ─────────────────────────────────────────────────────────────────────────────

class BiLSTMEncoder(nn.Module):
    """
    Bidirectional LSTM that encodes a 60-day multivariate time-series.

    Input:  (batch, seq_len=60, input_dim)
    Output: (batch, hidden_dim * 2)  – concatenated fwd + bwd final hidden states
    """

    def __init__(
        self,
        input_dim:  int   = 9,
        hidden_dim: int   = 128,
        num_layers: int   = 2,
        dropout:    float = 0.3,
    ):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.layer_norm = nn.LayerNorm(hidden_dim * 2)
        self.dropout    = nn.Dropout(dropout)
        self.output_dim = hidden_dim * 2

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (B, T, D)  →  (B, hidden*2)"""
        _, (h_n, _) = self.lstm(x)
        fwd = h_n[-2]                       # last layer, forward direction
        bwd = h_n[-1]                       # last layer, backward direction
        out = torch.cat([fwd, bwd], dim=-1)
        return self.dropout(self.layer_norm(out))


# ─────────────────────────────────────────────────────────────────────────────
# 2. FinBERT Encoder  (yiyanghkust/finbert-tone, @st.cache_resource)
# ─────────────────────────────────────────────────────────────────────────────

@st.cache_resource(show_spinner="Loading FinBERT (finbert-tone)…")
def _load_finbert() -> tuple:
    """
    Cached loader for yiyanghkust/finbert-tone.

    @st.cache_resource ensures the ~440 MB model is loaded only once per
    Streamlit session, preventing repeated RAM allocation.

    Python 3.12+ / Streamlit Cloud compatibility note:
        The fast (Rust) tokenizer for finbert-tone ships no binary wheel for
        Python >= 3.12. We therefore force `use_fast=False` so the pure-Python
        BertTokenizer is used instead. This is ~10% slower on long batches but
        fully correct and compatible with all CPython versions.
    """
    import sys
    logger.info(
        "Loading tokenizer & model: %s  (Python %s, use_fast=False)",
        FINBERT_MODEL, sys.version.split()[0],
    )
    tokenizer = AutoTokenizer.from_pretrained(
        FINBERT_MODEL,
        use_fast=False,          # ← required: no Rust wheel for Python ≥ 3.12
    )
    model = AutoModelForSequenceClassification.from_pretrained(
        FINBERT_MODEL,
        output_hidden_states=True,
        ignore_mismatched_sizes=True,   # silences harmless head-size warnings
    )
    model.eval()
    logger.info("FinBERT loaded. Labels: %s", model.config.id2label)
    return tokenizer, model


class FinBERTEncoder(nn.Module):
    """
    Wraps yiyanghkust/finbert-tone as a frozen feature extractor.
    Label order for finbert-tone: {0: Positive, 1: Negative, 2: Neutral}
    """

    LABEL_MAP = {0: "Positive", 1: "Negative", 2: "Neutral"}

    def __init__(self, max_length: int = 256):
        super().__init__()
        self.max_length     = max_length
        self.tokenizer, self.bert = _load_finbert()
        self.bert.to(DEVICE)
        # Freeze all weights – feature extractor only
        for p in self.bert.parameters():
            p.requires_grad = False
        self.hidden_size = self.bert.config.hidden_size  # 768

    def forward(self, texts: list[str]) -> torch.Tensor:
        """texts: list[str] (length=B)  →  (B, 768)"""
        enc = self.tokenizer(
            texts, padding=True, truncation=True,
            max_length=self.max_length, return_tensors="pt",
        ).to(DEVICE)
        with torch.no_grad():
            out = self.bert(**enc)
        return out.hidden_states[-1][:, 0, :]   # [CLS] embedding

    def sentiment_scores(self, texts: list[str]) -> pd.DataFrame:
        """Return per-text sentiment probability DataFrame."""
        enc = self.tokenizer(
            texts, padding=True, truncation=True,
            max_length=self.max_length, return_tensors="pt",
        ).to(DEVICE)
        with torch.no_grad():
            logits = self.bert(**enc).logits
        probs  = F.softmax(logits, dim=-1).cpu().numpy()
        labels = [self.LABEL_MAP[int(np.argmax(p))] for p in probs]
        return pd.DataFrame(
            probs, columns=["Positive", "Negative", "Neutral"]
        ).assign(predicted=labels)


# ─────────────────────────────────────────────────────────────────────────────
# 3. Weighted Fusion Layer  (Crisis Mode attention)
# ─────────────────────────────────────────────────────────────────────────────

class WeightedFusionLayer(nn.Module):
    """
    Fuses Bi-LSTM numeric embeddings with FinBERT text embeddings using
    a Weighted Attention Mechanism.

    Crisis Mode:
        When realised annualised volatility > CRISIS_VOL_THRESHOLD,
        the text (geopolitical) stream's attention weight is amplified
        via a learnable gating scalar α, shifting the fusion toward
        qualitative geopolitical signals.

    Outputs:
        alpha_signal  : (B, 1)  in [-1, +1]
        confidence    : (B, 1)  in [ 0,  1]
        regime_logits : (B, 3)  Bull / Bear / Neutral
        text_weight   : float   – actual text-stream weight applied
    """

    def __init__(
        self,
        lstm_dim:   int   = 256,
        bert_dim:   int   = 768,
        fusion_dim: int   = 256,
        num_heads:  int   = 4,
        dropout:    float = 0.3,
    ):
        super().__init__()
        self.proj_num  = nn.Linear(lstm_dim, fusion_dim)
        self.proj_text = nn.Linear(bert_dim, fusion_dim)

        self.attn_n2t = nn.MultiheadAttention(fusion_dim, num_heads, dropout=dropout, batch_first=True)
        self.attn_t2n = nn.MultiheadAttention(fusion_dim, num_heads, dropout=dropout, batch_first=True)
        self.norm1    = nn.LayerNorm(fusion_dim)
        self.norm2    = nn.LayerNorm(fusion_dim)

        # Crisis Mode: learnable amplification of text weight in [0, 1]
        self.crisis_gate = nn.Parameter(torch.tensor(0.7))

        mlp_in = fusion_dim * 2
        self.mlp = nn.Sequential(
            nn.Linear(mlp_in, fusion_dim), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(fusion_dim, fusion_dim // 2), nn.GELU(),
        )
        half = fusion_dim // 2
        self.head_alpha  = nn.Linear(half, 1)
        self.head_conf   = nn.Linear(half, 1)
        self.head_regime = nn.Linear(half, 3)
        self.dropout     = nn.Dropout(dropout)

    def forward(
        self,
        lstm_emb:    torch.Tensor,   # (B, lstm_dim)
        bert_emb:    torch.Tensor,   # (B, bert_dim)
        crisis_mode: bool = False,
    ) -> dict[str, torch.Tensor | float]:

        h_n = self.proj_num(lstm_emb).unsqueeze(1)    # (B, 1, F)
        h_t = self.proj_text(bert_emb).unsqueeze(1)   # (B, 1, F)

        f_n, _ = self.attn_n2t(h_n, h_t, h_t)
        f_t, _ = self.attn_t2n(h_t, h_n, h_n)
        f_n = self.norm1((h_n + f_n).squeeze(1))
        f_t = self.norm2((h_t + f_t).squeeze(1))

        # Weighted mix: in crisis mode, up-weight text stream
        text_weight  = float(torch.sigmoid(self.crisis_gate))
        if crisis_mode:
            text_weight = min(text_weight * 1.4, 0.95)   # amplify, cap at 0.95

        num_weight = 1.0 - text_weight
        combined   = torch.cat([f_n * num_weight, f_t * text_weight], dim=-1)

        features     = self.mlp(self.dropout(combined))
        alpha_signal = torch.tanh(self.head_alpha(features))
        confidence   = torch.sigmoid(self.head_conf(features))
        regime_logits= self.head_regime(features)

        return {
            "alpha":         alpha_signal,
            "confidence":    confidence,
            "regime_logits": regime_logits,
            "text_weight":   text_weight,
        }


# ─────────────────────────────────────────────────────────────────────────────
# 4. End-to-end model
# ─────────────────────────────────────────────────────────────────────────────

class AlphaInferenceModel(nn.Module):
    """
    Full pipeline: BiLSTMEncoder → FinBERTEncoder → WeightedFusionLayer

    forward() params:
        numeric_seq : (B, seq_len, num_features)
        texts       : list[str]
        current_vol : float  – annualised volatility for crisis detection
    """

    def __init__(
        self,
        num_features: int   = 9,
        lstm_hidden:  int   = 128,
        lstm_layers:  int   = 2,
        fusion_dim:   int   = 256,
        num_heads:    int   = 4,
        dropout:      float = 0.3,
    ):
        super().__init__()
        self.lstm_encoder = BiLSTMEncoder(num_features, lstm_hidden, lstm_layers, dropout)
        self.bert_encoder = FinBERTEncoder()
        self.fusion       = WeightedFusionLayer(
            lstm_dim=lstm_hidden * 2,
            bert_dim=self.bert_encoder.hidden_size,
            fusion_dim=fusion_dim,
            num_heads=num_heads,
            dropout=dropout,
        )

    def forward(
        self,
        numeric_seq: torch.Tensor,
        texts:       list[str],
        current_vol: float = 0.0,
    ) -> dict:
        crisis_mode = current_vol > CRISIS_VOL_THRESHOLD
        lstm_emb    = self.lstm_encoder(numeric_seq)
        bert_emb    = self.bert_encoder(texts)
        out         = self.fusion(lstm_emb, bert_emb, crisis_mode=crisis_mode)
        out["regime_probs"] = F.softmax(out["regime_logits"], dim=-1)
        out["crisis_mode"]  = crisis_mode
        return out


# ─────────────────────────────────────────────────────────────────────────────
# 5. Price-Only Baseline (ablation – no text signal)
# ─────────────────────────────────────────────────────────────────────────────

class PriceOnlyBaseline(nn.Module):
    """
    Ablation baseline: Bi-LSTM only, no FinBERT, no fusion.
    Used to quantify the incremental value of the geopolitical text stream.
    """

    def __init__(
        self,
        num_features: int   = 9,
        lstm_hidden:  int   = 128,
        lstm_layers:  int   = 2,
        dropout:      float = 0.3,
    ):
        super().__init__()
        self.lstm_encoder = BiLSTMEncoder(num_features, lstm_hidden, lstm_layers, dropout)
        half = lstm_hidden
        self.mlp = nn.Sequential(
            nn.Linear(lstm_hidden * 2, lstm_hidden), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(lstm_hidden, half), nn.GELU(),
        )
        self.head_alpha  = nn.Linear(half, 1)
        self.head_conf   = nn.Linear(half, 1)
        self.head_regime = nn.Linear(half, 3)

    def forward(self, numeric_seq: torch.Tensor) -> dict:
        emb      = self.lstm_encoder(numeric_seq)
        feat     = self.mlp(emb)
        alpha    = torch.tanh(self.head_alpha(feat))
        conf     = torch.sigmoid(self.head_conf(feat))
        regime_l = self.head_regime(feat)
        return {
            "alpha":         alpha,
            "confidence":    conf,
            "regime_logits": regime_l,
            "regime_probs":  F.softmax(regime_l, dim=-1),
            "text_weight":   0.0,
            "crisis_mode":   False,
        }


# ─────────────────────────────────────────────────────────────────────────────
# 6. Inference helper
# ─────────────────────────────────────────────────────────────────────────────

@torch.no_grad()
def run_inference(
    model:         AlphaInferenceModel,
    numeric_seq:   np.ndarray,    # (seq_len, num_features)
    headline_text: str,
    current_vol:   float = 0.0,
) -> InferenceResult:
    """Single-sample inference returning a structured InferenceResult."""
    model.eval().to(DEVICE)
    x    = torch.tensor(numeric_seq, dtype=torch.float32).unsqueeze(0).to(DEVICE)
    text = headline_text.strip() or "No significant geopolitical events."

    out      = model(x, [text], current_vol=current_vol)
    alpha    = float(out["alpha"].cpu().item())
    conf     = float(out["confidence"].cpu().item())
    reg_p    = out["regime_probs"].cpu().numpy()[0]
    reg_map  = {0: "Bull", 1: "Bear", 2: "Neutral"}
    regime   = reg_map[int(np.argmax(reg_p))]

    sent_df  = model.bert_encoder.sentiment_scores([text])
    s_row    = sent_df.iloc[0]

    return InferenceResult(
        alpha=alpha,
        confidence=conf,
        regime=regime,
        regime_probs={"Bull": float(reg_p[0]), "Bear": float(reg_p[1]), "Neutral": float(reg_p[2])},
        sentiment=s_row["predicted"],
        sentiment_probs={
            "Positive": float(s_row["Positive"]),
            "Negative": float(s_row["Negative"]),
            "Neutral":  float(s_row["Neutral"]),
        },
        crisis_mode=bool(out["crisis_mode"]),
        crisis_weight=float(out["text_weight"]),
        current_vol=current_vol,
    )


# ─────────────────────────────────────────────────────────────────────────────
# 7. Backtest Engine  (Sharpe, Max Drawdown, RMSE, Ablation Study)
# ─────────────────────────────────────────────────────────────────────────────

class BacktestEngine:
    """
    Simulates a simple long/short strategy driven by alpha signals and
    computes professional performance metrics.

    Also runs the Ablation Study comparing:
        Hybrid Architecture  (BiLSTM + FinBERT + Fusion)
        Price-Only Baseline  (BiLSTM alone)
    """

    RISK_FREE_RATE = 0.05   # 5% annual

    def __init__(
        self,
        model:        AlphaInferenceModel,
        baseline:     PriceOnlyBaseline,
        num_features: int = 9,
    ):
        self.model    = model
        self.baseline = baseline
        self.n_feat   = num_features

    # ── Public ───────────────────────────────────────────────────────────

    def run(
        self,
        X:        np.ndarray,           # (N, seq_len, D)
        y_true:   np.ndarray,           # (N,)  actual next-day returns
        texts:    list[str],            # length N
        vols:     Optional[np.ndarray] = None,  # (N,) annualised vols
    ) -> BacktestResult:
        predictions = self._predict_hybrid(X, texts, vols)
        return self._compute_metrics(predictions, y_true)

    def ablation_study(
        self,
        X:      np.ndarray,
        y_true: np.ndarray,
        texts:  list[str],
        vols:   Optional[np.ndarray] = None,
    ) -> AblationResult:
        logger.info("BacktestEngine › running ablation study …")
        hybrid_preds   = self._predict_hybrid(X, texts, vols)
        baseline_preds = self._predict_baseline(X)
        return AblationResult(
            hybrid=self._compute_metrics(hybrid_preds, y_true),
            price_only=self._compute_metrics(baseline_preds, y_true),
        )

    # ── Prediction helpers ────────────────────────────────────────────────

    @torch.no_grad()
    def _predict_hybrid(
        self,
        X:     np.ndarray,
        texts: list[str],
        vols:  Optional[np.ndarray],
    ) -> np.ndarray:
        self.model.eval().to(DEVICE)
        preds = []
        for i in range(len(X)):
            xi  = torch.tensor(X[i:i+1], dtype=torch.float32).to(DEVICE)
            vol = float(vols[i]) if vols is not None else 0.0
            txt = texts[i] if i < len(texts) else ""
            out = self.model(xi, [txt or "No headlines."], current_vol=vol)
            preds.append(float(out["alpha"].cpu().item()))
        return np.array(preds, dtype=np.float32)

    @torch.no_grad()
    def _predict_baseline(self, X: np.ndarray) -> np.ndarray:
        self.baseline.eval().to(DEVICE)
        preds = []
        for i in range(len(X)):
            xi  = torch.tensor(X[i:i+1], dtype=torch.float32).to(DEVICE)
            out = self.baseline(xi)
            preds.append(float(out["alpha"].cpu().item()))
        return np.array(preds, dtype=np.float32)

    # ── Metric computation ────────────────────────────────────────────────

    def _compute_metrics(
        self,
        predictions: np.ndarray,
        y_true:      np.ndarray,
    ) -> BacktestResult:
        # Strategy: go long when alpha > 0, short when alpha < 0
        direction     = np.sign(predictions)
        strat_returns = direction * y_true

        # Equity curve
        equity_curve = pd.Series(
            (1 + strat_returns).cumprod(),
            name="equity_curve",
        )

        total_return      = float(equity_curve.iloc[-1] - 1.0)
        n_trading_days    = len(strat_returns)
        annualised_return = float((1 + total_return) ** (252 / max(n_trading_days, 1)) - 1)

        # Sharpe Ratio
        daily_rf   = self.RISK_FREE_RATE / 252
        excess     = strat_returns - daily_rf
        sharpe     = float(
            (excess.mean() / (excess.std() + 1e-9)) * np.sqrt(252)
        )

        # Maximum Drawdown
        roll_max  = equity_curve.cummax()
        drawdown  = (equity_curve - roll_max) / (roll_max + 1e-9)
        max_dd    = float(drawdown.min())

        # RMSE (model's alpha vs actual next-day return)
        rmse = float(np.sqrt(np.mean((predictions - y_true) ** 2)))

        n_trades = int(np.sum(direction != 0))

        return BacktestResult(
            sharpe_ratio=sharpe,
            max_drawdown=max_dd,
            rmse=rmse,
            annualised_return=annualised_return,
            total_return=total_return,
            n_trades=n_trades,
            equity_curve=equity_curve,
        )


# ─────────────────────────────────────────────────────────────────────────────
# 8. Model factory
# ─────────────────────────────────────────────────────────────────────────────

@st.cache_resource(show_spinner="Initialising Alpha Inference Model…")
def build_model(num_features: int = 9) -> tuple[AlphaInferenceModel, PriceOnlyBaseline]:
    """
    @st.cache_resource: builds models once per Streamlit session.
    Returns both the hybrid model and the price-only baseline.
    """
    model    = AlphaInferenceModel(num_features=num_features).to(DEVICE)
    baseline = PriceOnlyBaseline(num_features=num_features).to(DEVICE)
    logger.info("Models built on %s", DEVICE)
    return model, baseline


# ─────────────────────────────────────────────────────────────────────────────
# CLI smoke-test
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import warnings
    warnings.filterwarnings("ignore")

    print(f"Device: {DEVICE}")
    model, baseline = build_model(num_features=9)

    dummy_seq = np.random.randn(60, 9).astype(np.float32)
    headline  = (
        "Federal Reserve signals further rate hikes amid escalating "
        "US-China trade tensions and Middle East conflict."
    )

    result = run_inference(model, dummy_seq, headline, current_vol=0.35)
    print(f"\n{'='*55}")
    print(f"Alpha signal:    {result.alpha:+.4f}  ({result.signal_label})")
    print(f"Confidence:      {result.confidence:.1%}")
    print(f"Regime:          {result.regime}")
    print(f"Sentiment:       {result.sentiment}")
    print(f"Crisis Mode:     {result.crisis_mode}  (vol={result.current_vol:.2%})")
    print(f"Text weight:     {result.crisis_weight:.2f}")

    # Quick ablation sanity check
    N = 20
    X_dummy  = np.random.randn(N, 60, 9).astype(np.float32)
    y_dummy  = np.random.randn(N).astype(np.float32) * 0.01
    texts    = ["Central bank rate decision"] * N
    vols     = np.random.uniform(0.1, 0.5, N).astype(np.float32)

    engine   = BacktestEngine(model, baseline)
    ablation = engine.ablation_study(X_dummy, y_dummy, texts, vols)

    print(f"\n{'='*55}")
    print("ABLATION STUDY RESULTS")
    print(f"{'─'*55}")
    print(f"{'Metric':<22} {'Hybrid':>12} {'Price-Only':>12}")
    print(f"{'─'*55}")
    h, p = ablation.hybrid, ablation.price_only
    print(f"{'Sharpe Ratio':<22} {h.sharpe_ratio:>12.3f} {p.sharpe_ratio:>12.3f}")
    print(f"{'Max Drawdown':<22} {h.max_drawdown:>12.1%} {p.max_drawdown:>12.1%}")
    print(f"{'RMSE':<22} {h.rmse:>12.6f} {p.rmse:>12.6f}")
    print(f"{'Annual Return':<22} {h.annualised_return:>12.1%} {p.annualised_return:>12.1%}")
    print(f"{'─'*55}")
    print(f"Sharpe improvement (Hybrid − Baseline): {ablation.sharpe_improvement:+.3f}")
    print(f"RMSE  improvement  (Baseline − Hybrid): {ablation.rmse_improvement:+.6f}")
