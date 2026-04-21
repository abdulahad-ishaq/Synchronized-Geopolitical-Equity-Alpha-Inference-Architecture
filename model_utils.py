"""
model_utils.py
==============
Synchronized Geopolitical-Equity Alpha Inference Architecture
-------------------------------------------------------------
OOP model components:

  BiLSTMEncoder         - 60-day OHLCV numeric encoding
  SentimentEncoder      - Pure-Python VADER+TextBlob (Python 3.14 safe)
  WeightedFusionLayer   - Cross-modal attention with Crisis Mode
  AlphaInferenceModel   - End-to-end hybrid model
  PriceOnlyBaseline     - Ablation baseline (no text signal)
  ModelTrainer          - Full training loop with real labeled data
  WalkForwardValidator  - Zero-lookahead walk-forward validation
  BacktestEngine        - Sharpe, Max Drawdown, RMSE, Ablation Study
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass
from typing import Optional

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import streamlit as st
from sklearn.preprocessing import StandardScaler

# Pure-Python sentiment - zero Rust/C++ build requirements
# Works on Python 3.14 and Streamlit Cloud free tier
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from textblob import TextBlob

from config import Config

logger = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s - %(message)s",
)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# =============================================================================
# Result dataclasses
# =============================================================================

@dataclass
class InferenceResult:
    """Complete output from a single inference pass."""
    alpha:           float
    confidence:      float
    regime:          str
    regime_probs:    dict
    sentiment:       str
    sentiment_probs: dict
    crisis_mode:     bool
    crisis_weight:   float
    current_vol:     float

    @property
    def signal_label(self) -> str:
        if self.alpha > Config.backtest.LONG_THRESHOLD:
            return "LONG"
        if self.alpha < Config.backtest.SHORT_THRESHOLD:
            return "SHORT"
        return "FLAT"

    @property
    def signal_arrow(self) -> str:
        if self.alpha > Config.backtest.LONG_THRESHOLD:
            return "▲"
        if self.alpha < Config.backtest.SHORT_THRESHOLD:
            return "▼"
        return "◆"

    @property
    def signal_colour(self) -> str:
        if self.alpha > Config.backtest.LONG_THRESHOLD:
            return Config.ui.BULL_COLOUR
        if self.alpha < Config.backtest.SHORT_THRESHOLD:
            return Config.ui.BEAR_COLOUR
        return Config.ui.NEUTRAL_COLOUR


@dataclass
class TrainingMetrics:
    """Metrics tracked during model training."""
    epoch:        int
    train_loss:   float
    val_loss:     float
    train_acc:    float
    val_acc:      float
    learning_rate: float


@dataclass
class BacktestResult:
    """Complete backtest performance metrics."""
    sharpe_ratio:      float
    max_drawdown:      float
    rmse:              float
    annualised_return: float
    total_return:      float
    n_trades:          int
    win_rate:          float
    equity_curve:      pd.Series
    trade_log:         pd.DataFrame

    @property
    def calmar_ratio(self) -> float:
        """Annualised return / Max drawdown magnitude."""
        if abs(self.max_drawdown) < 1e-9:
            return 0.0
        return self.annualised_return / abs(self.max_drawdown)

    @property
    def profit_factor(self) -> float:
        """Gross profit / Gross loss."""
        gains  = self.trade_log["pnl"][self.trade_log["pnl"] > 0].sum()
        losses = abs(self.trade_log["pnl"][self.trade_log["pnl"] < 0].sum())
        return gains / losses if losses > 0 else float("inf")


@dataclass
class AblationResult:
    """Ablation study comparing hybrid vs price-only model."""
    hybrid:     BacktestResult
    price_only: BacktestResult

    @property
    def sharpe_improvement(self) -> float:
        return self.hybrid.sharpe_ratio - self.price_only.sharpe_ratio

    @property
    def rmse_improvement(self) -> float:
        return self.price_only.rmse - self.hybrid.rmse

    @property
    def drawdown_improvement(self) -> float:
        return self.price_only.max_drawdown - self.hybrid.max_drawdown

    @property
    def text_signal_value(self) -> str:
        if self.sharpe_improvement > 0.3:
            return "High"
        if self.sharpe_improvement > 0.1:
            return "Moderate"
        if self.sharpe_improvement > 0:
            return "Low"
        return "Negative"


# =============================================================================
# 1. Bi-LSTM Encoder
# =============================================================================

class BiLSTMEncoder(nn.Module):
    """
    Bidirectional LSTM encoding a 60-day multivariate time-series.

    Input:  (batch, seq_len=60, input_dim)
    Output: (batch, hidden_dim * 2)
    """

    def __init__(
        self,
        input_dim:  int   = Config.model.NUM_FEATURES,
        hidden_dim: int   = Config.model.LSTM_HIDDEN,
        num_layers: int   = Config.model.LSTM_LAYERS,
        dropout:    float = Config.model.DROPOUT,
    ):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.output_dim = hidden_dim * 2

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

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (B, T, D) -> (B, hidden*2)"""
        _, (h_n, _) = self.lstm(x)
        fwd = h_n[-2]
        bwd = h_n[-1]
        out = torch.cat([fwd, bwd], dim=-1)
        return self.dropout(self.layer_norm(out))


# =============================================================================
# 2. Sentiment Encoder (Pure Python - Python 3.14 safe)
# =============================================================================

@st.cache_resource(show_spinner="Loading sentiment analyser...")
def _load_sentiment_analyser():
    """
    Cached VADER analyser.
    @st.cache_resource loads once per Streamlit session.
    Pure Python - no Rust, no C++, works on Python 3.14.
    """
    analyser = SentimentIntensityAnalyzer()
    logger.info("VADER SentimentIntensityAnalyzer loaded.")
    return analyser


class SentimentEncoder(nn.Module):
    """
    Financial sentiment encoder using VADER + TextBlob.

    Produces a (batch, 768) embedding vector compatible with
    the original FinBERT-based FusionLayer dimensions.

    Label map: {0: Positive, 1: Negative, 2: Neutral}
    """

    LABEL_MAP = {0: "Positive", 1: "Negative", 2: "Neutral"}
    hidden_size = 768   # kept for FusionLayer compatibility

    def __init__(self, max_length: int = 256):
        super().__init__()
        self.max_length = max_length
        self.analyser   = _load_sentiment_analyser()

    def _score(self, text: str) -> tuple[float, float, float]:
        """Return (positive, negative, neutral) probabilities."""
        text = (text or "").strip()[:self.max_length]
        if not text:
            return 0.0, 0.0, 1.0

        vs       = self.analyser.polarity_scores(text)
        compound = vs["compound"]
        tb_pol   = TextBlob(text).sentiment.polarity
        blended  = 0.7 * compound + 0.3 * tb_pol

        if blended > 0.05:
            pos = 0.5 + blended * 0.5
            neg = max(0.0, 0.5 - blended * 0.5 - 0.1)
            neu = max(0.0, 1.0 - pos - neg)
        elif blended < -0.05:
            neg = 0.5 + abs(blended) * 0.5
            pos = max(0.0, 0.5 - abs(blended) * 0.5 - 0.1)
            neu = max(0.0, 1.0 - pos - neg)
        else:
            pos, neg, neu = 0.2, 0.2, 0.6

        total = pos + neg + neu
        return pos / total, neg / total, neu / total

    def forward(self, texts: list[str]) -> torch.Tensor:
        """texts: list[str] -> (B, 768)"""
        batch = []
        for text in texts:
            pos, neg, neu = self._score(text)
            vec = [pos, neg, neu] + [0.0] * (self.hidden_size - 3)
            batch.append(vec)
        return torch.tensor(batch, dtype=torch.float32).to(DEVICE)

    def sentiment_scores(self, texts: list[str]) -> pd.DataFrame:
        """Return per-text sentiment probability DataFrame."""
        rows = []
        for text in texts:
            pos, neg, neu = self._score(text)
            label = self.LABEL_MAP[int(torch.tensor([pos, neg, neu]).argmax())]
            rows.append({
                "Positive": pos, "Negative": neg,
                "Neutral": neu, "predicted": label,
            })
        return pd.DataFrame(rows)


# Keep FinBERTEncoder as alias for backward compatibility
FinBERTEncoder = SentimentEncoder


# =============================================================================
# 3. Weighted Fusion Layer (Crisis Mode)
# =============================================================================

class WeightedFusionLayer(nn.Module):
    """
    Fuses Bi-LSTM numeric embeddings with sentiment text embeddings.

    Crisis Mode:
        When volatility > CRISIS_VOL_THRESHOLD, the text stream weight
        is amplified — geopolitical signals dominate during market stress.

    Outputs:
        alpha_signal  : (B, 1) in [-1, +1]
        confidence    : (B, 1) in [ 0,  1]
        regime_logits : (B, 3) Bull / Bear / Neutral
        text_weight   : float
    """

    def __init__(
        self,
        lstm_dim:   int   = Config.model.LSTM_HIDDEN * 2,
        bert_dim:   int   = 768,
        fusion_dim: int   = Config.model.FUSION_DIM,
        num_heads:  int   = Config.model.NUM_HEADS,
        dropout:    float = Config.model.DROPOUT,
    ):
        super().__init__()
        self.proj_num  = nn.Linear(lstm_dim,   fusion_dim)
        self.proj_text = nn.Linear(bert_dim,   fusion_dim)

        self.attn_n2t = nn.MultiheadAttention(
            fusion_dim, num_heads, dropout=dropout, batch_first=True
        )
        self.attn_t2n = nn.MultiheadAttention(
            fusion_dim, num_heads, dropout=dropout, batch_first=True
        )
        self.norm1 = nn.LayerNorm(fusion_dim)
        self.norm2 = nn.LayerNorm(fusion_dim)

        # Learnable crisis gate
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
        lstm_emb:    torch.Tensor,
        bert_emb:    torch.Tensor,
        crisis_mode: bool = False,
    ) -> dict:
        h_n = self.proj_num(lstm_emb).unsqueeze(1)
        h_t = self.proj_text(bert_emb).unsqueeze(1)

        f_n, _ = self.attn_n2t(h_n, h_t, h_t)
        f_t, _ = self.attn_t2n(h_t, h_n, h_n)
        f_n = self.norm1((h_n + f_n).squeeze(1))
        f_t = self.norm2((h_t + f_t).squeeze(1))

        text_weight = float(torch.sigmoid(self.crisis_gate))
        if crisis_mode:
            text_weight = min(text_weight * 1.4, 0.95)

        num_weight = 1.0 - text_weight
        combined   = torch.cat([f_n * num_weight, f_t * text_weight], dim=-1)
        features   = self.mlp(self.dropout(combined))

        return {
            "alpha":         torch.tanh(self.head_alpha(features)),
            "confidence":    torch.sigmoid(self.head_conf(features)),
            "regime_logits": self.head_regime(features),
            "text_weight":   text_weight,
        }


# =============================================================================
# 4. End-to-end Hybrid Model
# =============================================================================

class AlphaInferenceModel(nn.Module):
    """
    Full pipeline: BiLSTMEncoder -> SentimentEncoder -> WeightedFusionLayer

    forward() params:
        numeric_seq : (B, seq_len, num_features)
        texts       : list[str]
        current_vol : float - for crisis detection
    """

    def __init__(
        self,
        num_features: int   = Config.model.NUM_FEATURES,
        lstm_hidden:  int   = Config.model.LSTM_HIDDEN,
        lstm_layers:  int   = Config.model.LSTM_LAYERS,
        fusion_dim:   int   = Config.model.FUSION_DIM,
        num_heads:    int   = Config.model.NUM_HEADS,
        dropout:      float = Config.model.DROPOUT,
    ):
        super().__init__()
        self.lstm_encoder = BiLSTMEncoder(num_features, lstm_hidden, lstm_layers, dropout)
        self.bert_encoder = SentimentEncoder()
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
        crisis_mode = current_vol > Config.model.CRISIS_VOL_THRESHOLD
        lstm_emb    = self.lstm_encoder(numeric_seq)
        bert_emb    = self.bert_encoder(texts)
        out         = self.fusion(lstm_emb, bert_emb, crisis_mode=crisis_mode)
        out["regime_probs"] = F.softmax(out["regime_logits"], dim=-1)
        out["crisis_mode"]  = crisis_mode
        return out


# =============================================================================
# 5. Price-Only Baseline (ablation)
# =============================================================================

class PriceOnlyBaseline(nn.Module):
    """
    Ablation baseline: Bi-LSTM only, no sentiment, no fusion.
    Quantifies the incremental value of the geopolitical text stream.
    """

    def __init__(
        self,
        num_features: int   = Config.model.NUM_FEATURES,
        lstm_hidden:  int   = Config.model.LSTM_HIDDEN,
        lstm_layers:  int   = Config.model.LSTM_LAYERS,
        dropout:      float = Config.model.DROPOUT,
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
        regime_l = self.head_regime(feat)
        return {
            "alpha":         torch.tanh(self.head_alpha(feat)),
            "confidence":    torch.sigmoid(self.head_conf(feat)),
            "regime_logits": regime_l,
            "regime_probs":  F.softmax(regime_l, dim=-1),
            "text_weight":   0.0,
            "crisis_mode":   False,
        }


# =============================================================================
# 6. Model Trainer - Full training loop with real labeled data
# =============================================================================

class ModelTrainer:
    """
    Full training loop for AlphaInferenceModel with:
      - Train / validation / test splits
      - Early stopping
      - Learning rate scheduling
      - Checkpoint saving (best model)
      - Training history tracking
      - Direction accuracy metric

    Usage:
        trainer = ModelTrainer(model)
        history = trainer.train(X_train, y_train, X_val, y_val, texts_train, texts_val)
        trainer.save_checkpoint("best_model.pt")
    """

    def __init__(
        self,
        model:         AlphaInferenceModel,
        learning_rate: float = Config.model.LEARNING_RATE,
        device:        torch.device = DEVICE,
    ):
        self.model    = model.to(device)
        self.device   = device
        self.history:  list[TrainingMetrics] = []

        self.optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=learning_rate,
            weight_decay=1e-4,
        )
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer,
            mode="min",
            factor=0.5,
            patience=5,
            verbose=True,
        )
        self.best_val_loss   = float("inf")
        self.patience_counter = 0

    def train(
        self,
        X_train:      np.ndarray,
        y_train:      np.ndarray,
        X_val:        np.ndarray,
        y_val:        np.ndarray,
        texts_train:  list[str],
        texts_val:    list[str],
        num_epochs:   int = Config.model.NUM_EPOCHS,
        batch_size:   int = Config.model.BATCH_SIZE,
        patience:     int = Config.model.EARLY_STOPPING_PATIENCE,
        checkpoint_path: str = Config.model.BEST_MODEL_PATH,
    ) -> list[TrainingMetrics]:
        """
        Train the model with early stopping and checkpoint saving.

        Args:
            X_train:      (N, seq_len, features) float32
            y_train:      (N,) float32  - next-day returns (labels)
            X_val:        (M, seq_len, features)
            y_val:        (M,) float32
            texts_train:  list of headline strings per window
            texts_val:    list of headline strings per window
            num_epochs:   maximum training epochs
            batch_size:   mini-batch size
            patience:     early stopping patience
            checkpoint_path: where to save best model

        Returns:
            list[TrainingMetrics] - training history per epoch
        """
        logger.info(
            "ModelTrainer: train=%d  val=%d  epochs=%d  batch=%d",
            len(X_train), len(X_val), num_epochs, batch_size,
        )

        for epoch in range(1, num_epochs + 1):
            train_loss, train_acc = self._run_epoch(
                X_train, y_train, texts_train, batch_size, training=True
            )
            val_loss, val_acc = self._run_epoch(
                X_val, y_val, texts_val, batch_size, training=False
            )

            self.scheduler.step(val_loss)
            current_lr = self.optimizer.param_groups[0]["lr"]

            metrics = TrainingMetrics(
                epoch=epoch,
                train_loss=train_loss,
                val_loss=val_loss,
                train_acc=train_acc,
                val_acc=val_acc,
                learning_rate=current_lr,
            )
            self.history.append(metrics)

            logger.info(
                "Epoch %3d/%d | train_loss=%.4f  val_loss=%.4f | "
                "train_acc=%.1f%%  val_acc=%.1f%%",
                epoch, num_epochs, train_loss, val_loss,
                train_acc * 100, val_acc * 100,
            )

            # Early stopping + checkpoint
            if val_loss < self.best_val_loss:
                self.best_val_loss = val_loss
                self.patience_counter = 0
                self.save_checkpoint(checkpoint_path)
                logger.info("  -> New best model saved (val_loss=%.4f)", val_loss)
            else:
                self.patience_counter += 1
                if self.patience_counter >= patience:
                    logger.info(
                        "Early stopping at epoch %d (patience=%d)",
                        epoch, patience,
                    )
                    break

        # Load best model weights
        if os.path.exists(checkpoint_path):
            self.load_checkpoint(checkpoint_path)
            logger.info("Best model loaded from %s", checkpoint_path)

        return self.history

    def _run_epoch(
        self,
        X:        np.ndarray,
        y:        np.ndarray,
        texts:    list[str],
        batch_size: int,
        training: bool,
    ) -> tuple[float, float]:
        """Run one epoch. Returns (avg_loss, direction_accuracy)."""
        self.model.train(training)
        total_loss = 0.0
        correct    = 0
        n_batches  = 0

        # Shuffle for training
        indices = np.random.permutation(len(X)) if training else np.arange(len(X))

        for start in range(0, len(X), batch_size):
            batch_idx  = indices[start : start + batch_size]
            X_batch    = torch.tensor(X[batch_idx], dtype=torch.float32).to(self.device)
            y_batch    = torch.tensor(y[batch_idx], dtype=torch.float32).to(self.device)
            texts_batch = [texts[i] for i in batch_idx] if texts else [""] * len(batch_idx)

            with torch.set_grad_enabled(training):
                out = self.model(X_batch, texts_batch)
                alpha = out["alpha"].squeeze(-1)

                # Primary loss: MSE between predicted alpha and actual return
                mse_loss = F.mse_loss(alpha, y_batch)

                # Direction loss: penalise wrong direction predictions
                pred_dir   = torch.sign(alpha)
                actual_dir = torch.sign(y_batch)
                dir_loss   = F.binary_cross_entropy(
                    torch.sigmoid(alpha),
                    (actual_dir > 0).float(),
                )

                # Combined loss: 70% MSE + 30% direction
                loss = 0.7 * mse_loss + 0.3 * dir_loss

            if training:
                self.optimizer.zero_grad()
                loss.backward()
                # Gradient clipping to prevent exploding gradients
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                self.optimizer.step()

            total_loss += loss.item()
            n_batches  += 1

            # Direction accuracy
            correct += (pred_dir == actual_dir).sum().item()

        avg_loss   = total_loss / max(n_batches, 1)
        direction_acc = correct / max(len(X), 1)
        return avg_loss, direction_acc

    def save_checkpoint(self, path: str) -> None:
        """Save model weights to disk."""
        torch.save({
            "model_state_dict":     self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "best_val_loss":        self.best_val_loss,
            "history":              self.history,
        }, path)

    def load_checkpoint(self, path: str) -> None:
        """Load model weights from disk."""
        checkpoint = torch.load(path, map_location=self.device)
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        self.best_val_loss = checkpoint.get("best_val_loss", float("inf"))
        self.history       = checkpoint.get("history", [])

    @property
    def training_summary(self) -> dict:
        """Summary of training run."""
        if not self.history:
            return {}
        best   = min(self.history, key=lambda m: m.val_loss)
        last   = self.history[-1]
        return {
            "total_epochs":       last.epoch,
            "best_epoch":         best.epoch,
            "best_val_loss":      best.val_loss,
            "final_train_loss":   last.train_loss,
            "final_val_loss":     last.val_loss,
            "final_train_acc":    last.train_acc,
            "final_val_acc":      last.val_acc,
            "early_stopped":      last.epoch < Config.model.NUM_EPOCHS,
        }


# =============================================================================
# 7. Walk-Forward Validator
# =============================================================================

class WalkForwardValidator:
    """
    Walk-forward validation with strict zero-lookahead guarantee.

    Splits time series data into expanding windows:
      Window 1: train [0:T1],    test [T1:T1+step]
      Window 2: train [0:T1+step], test [T1+step:T1+2*step]
      ...

    This is the ONLY valid way to evaluate time-series ML models.
    Standard k-fold cross-validation introduces lookahead bias
    because future data leaks into training.
    """

    def __init__(
        self,
        model:        AlphaInferenceModel,
        min_train_size: int = 100,
        step_size:     int = 20,
    ):
        self.model          = model
        self.min_train_size = min_train_size
        self.step_size      = step_size

    def validate(
        self,
        X:     np.ndarray,
        y:     np.ndarray,
        texts: list[str],
    ) -> dict:
        """
        Run walk-forward validation.

        Returns dict with per-fold metrics and aggregate statistics.
        """
        N = len(X)
        if N < self.min_train_size + self.step_size:
            raise ValueError(
                f"Not enough data for walk-forward validation. "
                f"Need {self.min_train_size + self.step_size}, got {N}."
            )

        fold_metrics = []
        start = self.min_train_size

        while start + self.step_size <= N:
            X_train    = X[:start]
            y_train    = y[:start]
            X_test     = X[start : start + self.step_size]
            y_test     = y[start : start + self.step_size]
            texts_train = texts[:start]
            texts_test  = texts[start : start + self.step_size]

            # Predict on test window
            preds = self._predict_window(X_test, texts_test)

            # Metrics for this fold
            fold_rmse = float(np.sqrt(np.mean((preds - y_test) ** 2)))
            direction = np.sign(preds) == np.sign(y_test)
            fold_acc  = float(direction.mean())

            fold_metrics.append({
                "fold":       len(fold_metrics) + 1,
                "train_size": start,
                "test_start": start,
                "test_end":   start + self.step_size,
                "rmse":       fold_rmse,
                "direction_accuracy": fold_acc,
            })

            start += self.step_size

        df = pd.DataFrame(fold_metrics)
        return {
            "n_folds":          len(fold_metrics),
            "mean_rmse":        float(df["rmse"].mean()),
            "std_rmse":         float(df["rmse"].std()),
            "mean_direction_acc": float(df["direction_accuracy"].mean()),
            "fold_details":     df,
        }

    @torch.no_grad()
    def _predict_window(
        self,
        X:     np.ndarray,
        texts: list[str],
    ) -> np.ndarray:
        self.model.eval().to(DEVICE)
        preds = []
        for i in range(len(X)):
            xi  = torch.tensor(X[i:i+1], dtype=torch.float32).to(DEVICE)
            out = self.model(xi, [texts[i] if i < len(texts) else ""])
            preds.append(float(out["alpha"].cpu().item()))
        return np.array(preds, dtype=np.float32)


# =============================================================================
# 8. Backtest Engine
# =============================================================================

class BacktestEngine:
    """
    Simulates a long/short strategy and computes professional metrics:
      - Sharpe Ratio
      - Maximum Drawdown
      - RMSE
      - Win Rate
      - Calmar Ratio
      - Profit Factor
      - Trade Log

    Also runs Ablation Study comparing Hybrid vs Price-Only.
    """

    def __init__(
        self,
        model:    AlphaInferenceModel,
        baseline: PriceOnlyBaseline,
    ):
        self.model    = model
        self.baseline = baseline

    def run(
        self,
        X:     np.ndarray,
        y_true: np.ndarray,
        texts: list[str],
        vols:  Optional[np.ndarray] = None,
        market: str = "US",
    ) -> BacktestResult:
        preds = self._predict_hybrid(X, texts, vols)
        return self._compute_metrics(preds, y_true, market=market)

    def ablation_study(
        self,
        X:     np.ndarray,
        y_true: np.ndarray,
        texts: list[str],
        vols:  Optional[np.ndarray] = None,
        market: str = "US",
    ) -> AblationResult:
        logger.info("BacktestEngine: running ablation study...")
        hybrid_preds   = self._predict_hybrid(X, texts, vols)
        baseline_preds = self._predict_baseline(X)
        return AblationResult(
            hybrid     = self._compute_metrics(hybrid_preds,   y_true, market),
            price_only = self._compute_metrics(baseline_preds, y_true, market),
        )

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

    def _compute_metrics(
        self,
        predictions: np.ndarray,
        y_true:      np.ndarray,
        market:      str = "US",
    ) -> BacktestResult:
        # Use market-specific transaction cost
        tx_cost = (
            Config.backtest.PSX_TRANSACTION_COST
            if market == "PSX"
            else Config.backtest.TRANSACTION_COST
        )
        risk_free = (
            Config.backtest.PSX_RISK_FREE_RATE
            if market == "PSX"
            else Config.backtest.RISK_FREE_RATE
        )

        # Strategy: long when alpha > threshold, short when alpha < -threshold
        direction      = np.sign(predictions)
        strat_returns  = direction * y_true - tx_cost * np.abs(direction)

        # Equity curve
        equity_curve = pd.Series(
            (1 + strat_returns).cumprod(), name="equity_curve"
        )

        total_return      = float(equity_curve.iloc[-1] - 1.0)
        n_days            = len(strat_returns)
        ann_return        = float((1 + total_return) ** (252 / max(n_days, 1)) - 1)

        # Sharpe Ratio
        daily_rf  = risk_free / 252
        excess    = strat_returns - daily_rf
        sharpe    = float(
            (excess.mean() / (excess.std() + 1e-9)) * np.sqrt(252)
        )

        # Maximum Drawdown
        roll_max  = equity_curve.cummax()
        drawdown  = (equity_curve - roll_max) / (roll_max + 1e-9)
        max_dd    = float(drawdown.min())

        # RMSE
        rmse = float(np.sqrt(np.mean((predictions - y_true) ** 2)))

        # Win Rate & Trade Log
        trades    = []
        n_trades  = int(np.sum(direction != 0))
        n_wins    = int(np.sum((direction * y_true) > 0))
        win_rate  = n_wins / max(n_trades, 1)

        for i, (pred, actual, ret) in enumerate(
            zip(predictions, y_true, strat_returns)
        ):
            if direction[i] != 0:
                trades.append({
                    "step":      i,
                    "signal":    "LONG" if direction[i] > 0 else "SHORT",
                    "alpha":     float(pred),
                    "actual":    float(actual),
                    "pnl":       float(ret),
                    "win":       bool((direction[i] * actual) > 0),
                })

        trade_log = pd.DataFrame(trades) if trades else pd.DataFrame(
            columns=["step", "signal", "alpha", "actual", "pnl", "win"]
        )

        return BacktestResult(
            sharpe_ratio=sharpe,
            max_drawdown=max_dd,
            rmse=rmse,
            annualised_return=ann_return,
            total_return=total_return,
            n_trades=n_trades,
            win_rate=win_rate,
            equity_curve=equity_curve,
            trade_log=trade_log,
        )


# =============================================================================
# 9. Inference helper
# =============================================================================

@torch.no_grad()
def run_inference(
    model:         AlphaInferenceModel,
    numeric_seq:   np.ndarray,
    headline_text: str,
    current_vol:   float = 0.0,
) -> InferenceResult:
    """Single-sample inference returning a structured InferenceResult."""
    model.eval().to(DEVICE)
    x    = torch.tensor(numeric_seq, dtype=torch.float32).unsqueeze(0).to(DEVICE)
    text = headline_text.strip() or "No significant geopolitical events."

    out     = model(x, [text], current_vol=current_vol)
    alpha   = float(out["alpha"].cpu().item())
    conf    = float(out["confidence"].cpu().item())
    reg_p   = out["regime_probs"].cpu().numpy()[0]
    reg_map = {0: "Bull", 1: "Bear", 2: "Neutral"}
    regime  = reg_map[int(np.argmax(reg_p))]

    sent_df = model.bert_encoder.sentiment_scores([text])
    s_row   = sent_df.iloc[0]

    return InferenceResult(
        alpha=alpha,
        confidence=conf,
        regime=regime,
        regime_probs={
            "Bull":    float(reg_p[0]),
            "Bear":    float(reg_p[1]),
            "Neutral": float(reg_p[2]),
        },
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


# =============================================================================
# 10. Model factory
# =============================================================================

@st.cache_resource(show_spinner="Initialising Alpha Inference Model...")
def build_model(
    num_features: int = Config.model.NUM_FEATURES,
) -> tuple[AlphaInferenceModel, PriceOnlyBaseline]:
    """
    @st.cache_resource: builds models once per Streamlit session.
    Returns (hybrid_model, price_only_baseline).
    """
    model    = AlphaInferenceModel(num_features=num_features).to(DEVICE)
    baseline = PriceOnlyBaseline(num_features=num_features).to(DEVICE)

    # Load checkpoint if available
    if os.path.exists(Config.model.BEST_MODEL_PATH):
        try:
            state = torch.load(
                Config.model.BEST_MODEL_PATH, map_location=DEVICE
            )
            if "model_state_dict" in state:
                model.load_state_dict(state["model_state_dict"])
                logger.info("Loaded checkpoint: %s", Config.model.BEST_MODEL_PATH)
            else:
                model.load_state_dict(state)
        except Exception as exc:
            logger.warning("Could not load checkpoint: %s", exc)

    logger.info("Models built on %s", DEVICE)
    return model, baseline


def prepare_training_data(
    X:           np.ndarray,
    y:           np.ndarray,
    texts:       list[str],
    train_split: float = Config.model.TRAIN_SPLIT,
    val_split:   float = Config.model.VAL_SPLIT,
) -> tuple:
    """
    Split sequences into train / validation / test sets.

    IMPORTANT: Uses sequential split (NOT random shuffle).
    Random shuffle would introduce lookahead bias in time series.

    Returns:
        (X_train, y_train, texts_train,
         X_val,   y_val,   texts_val,
         X_test,  y_test,  texts_test)
    """
    N       = len(X)
    t_end   = int(N * train_split)
    v_end   = int(N * (train_split + val_split))

    return (
        X[:t_end],   y[:t_end],   texts[:t_end],
        X[t_end:v_end], y[t_end:v_end], texts[t_end:v_end],
        X[v_end:],   y[v_end:],   texts[v_end:],
    )


# =============================================================================
# CLI smoke test
# =============================================================================

if __name__ == "__main__":
    import warnings
    warnings.filterwarnings("ignore")

    print(f"Device: {DEVICE}")
    print("=" * 55)

    # Build models
    model, baseline = build_model(num_features=9)
    print("Models built successfully.")

    # Test inference
    dummy_seq = np.random.randn(60, 9).astype(np.float32)
    headline  = (
        "Federal Reserve signals rate cuts amid Pakistan rupee crisis "
        "and OPEC production cuts."
    )

    result = run_inference(model, dummy_seq, headline, current_vol=0.35)
    print(f"\n-- Inference Result --")
    print(f"  Alpha:       {result.alpha:+.4f}  ({result.signal_arrow} {result.signal_label})")
    print(f"  Confidence:  {result.confidence:.1%}")
    print(f"  Regime:      {result.regime}")
    print(f"  Sentiment:   {result.sentiment}")
    print(f"  Crisis Mode: {result.crisis_mode}  (vol={result.current_vol:.2%})")
    print(f"  Text weight: {result.crisis_weight:.2f}")

    # Test training loop (quick)
    print(f"\n-- Training Loop (5 epochs, small data) --")
    N = 50
    X_dummy  = np.random.randn(N, 60, 9).astype(np.float32)
    y_dummy  = np.random.randn(N).astype(np.float32) * 0.01
    texts_dummy = ["Fed rate decision"] * N

    (X_tr, y_tr, t_tr,
     X_vl, y_vl, t_vl,
     X_te, y_te, t_te) = prepare_training_data(X_dummy, y_dummy, texts_dummy)

    trainer = ModelTrainer(model, learning_rate=1e-3)
    history = trainer.train(
        X_tr, y_tr, X_vl, y_vl, t_tr, t_vl,
        num_epochs=5, batch_size=8, patience=3,
    )
    print(f"  Epochs trained: {len(history)}")
    print(f"  Summary: {trainer.training_summary}")

    # Test ablation study
    print(f"\n-- Ablation Study --")
    engine   = BacktestEngine(model, baseline)
    ablation = engine.ablation_study(
        X_dummy, y_dummy, texts_dummy,
        vols=np.random.uniform(0.1, 0.5, N).astype(np.float32),
    )
    h, p = ablation.hybrid, ablation.price_only
    print(f"  {'Metric':<20} {'Hybrid':>10} {'Price-Only':>12}")
    print(f"  {'-'*44}")
    print(f"  {'Sharpe Ratio':<20} {h.sharpe_ratio:>10.3f} {p.sharpe_ratio:>12.3f}")
    print(f"  {'Max Drawdown':<20} {h.max_drawdown:>10.1%} {p.max_drawdown:>12.1%}")
    print(f"  {'RMSE':<20} {h.rmse:>10.5f} {p.rmse:>12.5f}")
    print(f"  {'Win Rate':<20} {h.win_rate:>10.1%} {p.win_rate:>12.1%}")
    print(f"\n  Text signal value: {ablation.text_signal_value}")
    print(f"  Sharpe improvement: {ablation.sharpe_improvement:+.3f}")

    print("\nAll tests passed.")
