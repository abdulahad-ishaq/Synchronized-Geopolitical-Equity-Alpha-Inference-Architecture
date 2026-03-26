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

###  Setup
1. **Clone & Env:**
   `git clone https://github.com/abdulahadishaq512/SGEAIA.git && cd SGEAIA`
   `python -m venv venv && .\venv\Scripts\Activate.ps1`
   `pip install -r requirements.txt`

2. **Keys:** Add `NEWSAPI_KEY` and `FRED_API_KEY` to `.streamlit/secrets.toml`.

3. **Launch:** `streamlit run app.py`
