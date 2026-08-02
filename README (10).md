# Weather Prediction Using Time-Series Neural Networks
### Multi-Step Forecasting with LSTM / GRU

## Overview

This project builds a neural network that learns from historical hourly
weather measurements and forecasts **multiple future time steps at once**
(multi-step forecasting), rather than just the next single reading. It
uses an LSTM/GRU-based sequence model trained on the **Weather in Szeged
2006–2016** dataset from Kaggle.

## Dataset

**File:** `weatherHistory.csv`

## Pipeline

1. **Load & clean data**
   - Parse `Formatted Date` as a timezone-aware datetime index
   - Fill missing `Precip Type` values with the most frequent category
   - Drop the constant `Loud Cover` column and free-text `Summary` /
     `Daily Summary` columns
   - Time-interpolate any remaining numeric gaps

2. **Feature engineering**
   - Label-encode `Precip Type`
   - Add cyclical (sine/cosine) encodings for hour-of-day and
     day-of-year so the model can learn daily/seasonal periodicity

3. **Chronological train / val / test split (70% / 15% / 15%)**
   - No shuffling before splitting — required for valid time-series
     evaluation
   - `MinMaxScaler` fit only on the training set to avoid data leakage

4. **Sliding-window sequence construction**
   - **Input (X):** last **72 hours** (3 days) of all features
   - **Output (y):** next **24 hours** of the target variables
   - Direct (one-shot) multi-step formulation — avoids the error
     accumulation of recursive one-step-at-a-time forecasting

5. **Model architecture**
   - Stacked LSTM (or GRU) encoder → Dropout → Dense decoder → reshaped
     to `(24 time steps, n_targets)`
   - Trained with Adam optimizer, MSE loss, early stopping, and
     learning-rate reduction on plateau

6. **Targets forecasted:** Temperature (°C), Humidity, Pressure (millibars)

7. **LSTM vs GRU comparison**
   - Both architectures are trained back-to-back on the same data/split
     and compared directly on R², MAE, RMSE, parameter count, and
     training time

## Evaluation

Metrics are reported per target and per forecast step (1–24 hours ahead):

- **R²** — proportion of variance explained
- **MAE** and **RMSE** — average error in each variable's real units

### Results (72h lookback → 24h horizon, after Pressure sensor-error fix)

| Target | LSTM R² | LSTM MAE | LSTM RMSE | GRU R² | GRU MAE | GRU RMSE |
|---|---|---|---|---|---|---|
| Temperature (°C) | 0.930 | 1.85 | 2.44 | 0.936 | 1.75 | 2.33 |
| Humidity | 0.734 | 0.071 | 0.098 | 0.748 | 0.069 | 0.096 |
| Pressure (millibars) | 0.875 | 1.75 | 2.58 | 0.880 | 1.68 | 2.53 |

**Efficiency:**

| | LSTM | GRU |
|---|---|---|
| Parameters | 126,280 | 96,456 |
| Training time | ~160s | ~134s |

**Notes on results:**
- All three targets now explain 73–94% of variance — a strong result for
  a 24-hour-ahead multi-step forecast.
- Fixing the `Pressure (millibars) == 0` sensor-error rows (treating them
  as missing and interpolating) raised Pressure's R² from ~0.20 to
  ~0.88 — the largest single improvement in the project.
- Humidity is the hardest target to forecast, likely because it's noisier
  and less cyclical than temperature or pressure.
- GRU outperformed LSTM on every metric here while using ~24% fewer
  parameters and training ~16% faster.

## Visualizations included

- Training/validation loss curve
- Per-forecast-step MAE/RMSE (error growth across the horizon)
- R² per target (bar chart) and R² vs. forecast horizon (line chart)
- Single-sample forecast plot (history + true vs. predicted future)
- Grid of multi-sample forecasts across all targets
- Predicted-vs-actual scatter plots (with R² annotated)
- Residual (error) distribution histograms
- RMSE heatmap across forecast step × target
- Side-by-side LSTM vs GRU comparison (R², RMSE, training curves, efficiency)
