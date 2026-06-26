# Time-Series Demand/Price Forecasting

Comparison of classical (ARIMA/SARIMA), modern (Prophet, LSTM) and ensemble
forecasting models against naive baselines, validated by walk-forward rolling-origin
cross-validation on two real datasets.

## Datasets

| Dataset | Source | Length | Frequency |
|---------|--------|--------|-----------|
| Energy demand | energy.csv (gov/Kaggle) | 1433 rows | Daily |
| Crude oil (CL=F) | Yahoo Finance (yfinance) | 2610 rows | Business-day |

## Results

### Energy Demand

| Model | MAPE | sMAPE | RMSE | MAE | MASE |
|-------|------|-------|------|-----|------|
| naive | 31.87 | 25.92 | 0.3144 | 0.2477 | 1.1804 |
| seasonal_naive | 27.63 | 25.77 | 0.3298 | 0.2518 | 1.2002 |
| arima | NaN | NaN | NaN | NaN | NaN |
| prophet | 51.28 | 36.65 | 0.4741 | 0.3898 | 1.8580 |
| lstm | 30.21 | 29.30 | 0.3397 | 0.2774 | 1.3222 |
| ensemble | N/A | N/A | N/A | N/A | N/A |

Best model: **seasonal_naive** (MAPE = 27.63%)

> ARIMA hit numerical instability on the daily energy series (SARIMA m=12 at scale).
> Ensemble failed because ARIMA NaN propagated into NNLS weight learning.

### Commodity Price (Crude Oil CL=F)

| Model | MAPE | sMAPE | RMSE | MAE | MASE |
|-------|------|-------|------|-----|------|
| naive | 11.21 | 11.59 | 12.059 | 8.534 | 6.118 |
| arima | 11.26 | 11.61 | 12.034 | 8.551 | 6.131 |
| prophet | 39.38 | 54.11 | 35.849 | 29.173 | 20.914 |
| lstm | 15.54 | 18.37 | 19.214 | 12.539 | 8.989 |
| ensemble | 10.37 | 11.16 | 12.818 | 8.255 | 5.918 |

Best model: **ensemble** (MAPE = 10.37%)

> Prophet severely underperformed on this non-stationary price series (MAPE 39.38%).
> Ensemble (ARIMA + Prophet, NNLS weights) learned to down-weight Prophet,
> achieving the best overall MAPE (10.37%).

## Project Structure

```
time_series_forecasting/
├── data/{raw,processed}/  data/download.py
├── analysis/{eda,decomposition,stationarity}.py
├── models/{arima,prophet_model,lstm_model,ensemble}.py
├── evaluation/{metrics,backtester,benchmark}.py
├── visualisation/plots.py
├── dashboard/app.py
├── results/               <- plots and summary
└── requirements.txt
```

## Quick Start

```bash
# 1. Setup
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

# 2. Download data
python -m data.download

# 3. Run benchmark
python -m evaluation.benchmark

# 4. Generate plots
python -m visualisation.plots

# 5. Launch dashboard
streamlit run dashboard/app.py
```

## Key Design Decisions

- **Walk-forward validation** (>= 3 windows) prevents lookahead bias
- **Scaler fit on train-only** prevents leakage in LSTM normalization
- **pmdarima fallback**: if auto_arima fails, statsmodels AIC grid search is used
- **Ensemble weights** learned via NNLS on val set (not test set)
- **RANDOM_SEED = 42** throughout for reproducibility
