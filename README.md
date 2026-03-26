# Synchronized-Geopolitical-Equity-Alpha-Inference-Architecture
SGEAIA is an advanced AI-driven financial intelligence engine designed to bridge the gap between global geopolitical events and equity market movements. By fusing real-time news sentiment with historical price action, SGEAIA identifies "Alpha" opportunities that traditional technical analysis often misses.

 **[Live Demo](https://synchronized-geopolitical-equity-alpha-inference-architecture.streamlit.app/)**

### Core Features

### Dual-Engine Inference: 

Combines FinBERT (Natural Language Processing) for news sentiment and Bi-LSTM (Neural Networks) for price trend prediction.

### Temporal Synchronization: 

Aligns market ticks with real-time headlines from Reuters, Financial Times, and Al Jazeera.

### Macro-Context Integration: 

Pulls economic indicators (CPI, Interest Rates) directly from FRED (Federal Reserve Economic Data).

### Alpha Inference Dashboard: 

A minimalist, high-contrast Streamlit interface for real-time visualization and backtesting.

### Tech Stack

### Language: 

Python 3.12+

### AI/ML: 

PyTorch, HuggingFace (FinBERT), Scikit-learn

### Data: 

yFinance, NewsAPI, Pandas, NumPy

### Frontend: 

Streamlit (Custom CSS Branding)

### Technical Interpretation & User Guide
To effectively utilize the SGEAIA dashboard, observe the following data clusters:

#### 1. The Sentiment-Price Convergence
The Pulse: If the Geopolitical Sentiment Pulse is "Negative" while the Alpha Signal remains "Flat," the model suggests that while news is bearish, the market hasn't yet priced in the shock.

The Overlay: Divergence between price action and Alpha bars identifies non-random trends identified by the Bi-LSTM core.

### 2. The Ablation Study (Hybrid vs. Price-Only)
The Ablation table serves as the "Proof of Concept" for the AI:

Hybrid (Blue): Performance using both FinBERT news data and historical prices.

Price-Only (Grey): Performance using only historical price action.

Goal: Improved Sharpe Ratio or RMSE in the Hybrid column validates that geopolitical news adds measurable predictive value.

### 3. Risk Thresholds
Crisis Mode: If Volatility crosses the red Crisis Threshold, the system triggers a defensive "FLAT" Alpha signal to prioritize capital preservation.

### Setup & Installation
Clone & Environment:

git clone https://github.com/abdulahadishaq512/SGEAIA.git && cd SGEAIA
python -m venv venv
.\venv\Scripts\Activate.ps1
pip install -r requirements.txt
Keys: Add NEWSAPI_KEY and FRED_API_KEY to .streamlit/secrets.toml.

Launch: streamlit run app.py
