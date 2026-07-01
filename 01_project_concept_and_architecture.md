# Drought Prediction System — Concepts & Architecture

## What Is This Project?

This project predicts **how severe a drought will be** in a US county based on the past 7 days of weather data. It uses deep learning (a GRU neural network) trained on 19 million rows of real historical weather data.

The final product is a REST API — you send 7 days of weather, it returns a drought severity score.

---

## The Problem

Drought does not happen overnight. It builds slowly over days and weeks. So to predict drought, you need to look at a sequence of days — not just today's weather. This is a **time-series regression problem**.

- Input: 7 consecutive days of weather data (18 features per day)
- Output: 1 number — the predicted drought severity score for the next day
- Score range: 0 (no drought) → 5 (exceptional drought)

---

## Data Sources

| File | What It Contains | Size |
|---|---|---|
| `train_timeseries.csv` | Daily weather per US county (2000–2020) | 19.3M rows |
| `soil_data.csv` | Fixed land info per US county | 3,109 rows |

Both are joined on `fips` — the US county ID code.

After joining and removing rows with no score label, we end up with **~2.75 million usable rows**.

---

## The 18 Features

These are the columns the model learns from, grouped by category:

### Weather (7 features)
| Feature | Meaning |
|---|---|
| PS | Air pressure (hPa) |
| QV2M | Humidity (g/kg) |
| T2M_MAX | Max temperature that day (°C) |
| T2MDEW | Dew point — temp where dew forms (°C) |
| T2MWET | Wet bulb temp — how hot it feels (°C) |
| T2M_RANGE | Difference between max and min temp (°C) |
| TS | Ground surface temperature (°C) |

### Wind (2 features)
| Feature | Meaning |
|---|---|
| WS10M_RANGE | How much wind speed varied at 10m height (m/s) |
| WS50M_RANGE | How much wind speed varied at 50m height (m/s) |

### Geography (3 features)
| Feature | Meaning |
|---|---|
| lat | Latitude of the county |
| lon | Longitude of the county |
| elevation | Height above sea level (meters) |

### Land Cover (6 features)
| Feature | Meaning |
|---|---|
| NVG_LAND | % of county that is bare/no vegetation |
| GRS_LAND | % grassland |
| FOR_LAND | % forest |
| CULTRF_LAND | % rainfed farmland (rain-dependent crops) |
| CULTIR_LAND | % irrigated farmland (has water supply) |
| CULT_LAND | % total farmland (rainfed + irrigated) |

---

## Target — The Drought Score

The `score` column follows the **US Drought Monitor** scale:

| Score | Drought Level |
|---|---|
| 0 | No drought |
| 1 | Abnormally dry |
| 2 | Moderate drought |
| 3 | Severe drought |
| 4 | Extreme drought |
| 5 | Exceptional drought |

This is treated as **regression** (predicting a continuous number), not classification. The model outputs any decimal value like `2.34`, not a fixed category label.

---

## Pipeline Overview

```
Raw CSV files
     ↓
Notebook 1 — Preprocessing (PySpark)
  • Join train + soil on fips
  • Drop rows with no score
  • Sort by fips → date
  • StandardScaler on 18 features
  • Save as Parquet
     ↓
Notebook 2 — EDA (Exploratory Data Analysis)
  • Correlation heatmaps
  • Feature selection → pick best 18 columns
     ↓
Notebook 3 — Model Training (PyTorch)
  • Load scaled Parquet data
  • Sliding window: 7 days → predict day 8
  • Train/test split 80/20 (chronological)
  • GRU neural network
  • Save model as .pth file
     ↓
FastAPI — Production API
  • Load model + scaler on startup
  • Receive 7 days of raw weather via POST /predict
  • Apply same scaler → run model → return score
```

---

## Data Preprocessing (Notebook 1)

**Tool used:** PySpark (handles large data that doesn't fit in RAM)

Steps:
1. Load both CSV files into Spark DataFrames
2. Left join on `fips` — keeps all daily rows, attaches soil info
3. Convert `date` column from string to real date type
4. Drop any row where `score` is NULL (no label = can't train)
5. Sort by `fips` then `date` — time order is critical for time-series
6. Fill missing feature values with 0
7. Pack 18 features into one vector column using `VectorAssembler`
8. Scale using `StandardScaler` (mean=0, std=1 for each feature)
9. Save scaler stats (mean + std) as `scaler.pkl`
10. Save final data as `deep_learning_data.parquet`

**Why scale?** Features have very different ranges. For example, `elevation` can be 0–4000 while `QV2M` is 0–30. Without scaling, the model pays too much attention to large-number features. Scaling puts everything on the same playing field.

---

## Feature Engineering (Notebook 2)

EDA (Exploratory Data Analysis) was done to understand the data and pick the most useful features. Correlation heatmaps were used to find which features have the strongest relationship with the drought score. The final 18 features were selected based on this analysis and then applied back in Notebook 1.

---

## Model Training (Notebook 3)

**Tool used:** PyTorch + CUDA GPU

### Sliding Window
The raw data is converted into sequences using a sliding window of 7 days:

```
Day 1, 2, 3, 4, 5, 6, 7  →  predict Day 8 score
Day 2, 3, 4, 5, 6, 7, 8  →  predict Day 9 score
...
```

This gives ~2.75 million sequences total.

### Train/Test Split
Split is **chronological** (never shuffle time-series data):
- 80% train → 2,205,431 sequences
- 20% test  → 551,358 sequences

### The GRU Model

GRU (Gated Recurrent Unit) is a type of neural network designed for sequences. It has memory — it reads each day one by one and remembers important patterns.

```
Input: (batch, 7 days, 18 features)
         ↓
GRU Layer (hidden size = 64)
  reads day 1 → day 2 → ... → day 7
         ↓
Take only day 7's output (the final memory)
         ↓
Linear Layer → 1 number (drought score)
```

### Training
- Loss function: MSELoss (Mean Squared Error) — measures how far off predictions are
- Optimizer: Adam (lr=0.001)
- Epochs: 5 (results show room for improvement)
- Batch size: 256

### Training Results

| Epoch | Train Loss (MSE) | Val Loss (MSE) | Train RMSE | Val RMSE |
|---|---|---|---|---|
| 1 | 0.8399 | 1.0052 | ~0.92 | ~1.00 |
| 2 | 0.6631 | 1.0293 | ~0.81 | ~1.01 |
| 3 | 0.5754 | 0.9820 | ~0.76 | ~0.99 |
| 4 | 0.5297 | 1.0882 | ~0.73 | ~1.04 |
| 5 | 0.5009 | 0.9980 | ~0.71 | ~1.00 |

RMSE of ~1.0 means the model is off by about 1 drought level on average. The gap between train and val loss indicates **overfitting** — the model memorizes training data but doesn't generalize perfectly. This can be improved with dropout, more epochs, and early stopping.

---

## FastAPI (Production API)

Three files inside the `api/` folder:

### `model.py`
Defines the GRU neural network class — the blueprint of the brain.

### `main.py`
- Starts the FastAPI app
- On startup: loads `drought_gru_model.pth` and `scaler.pkl` into memory
- Both are stored in `app.state` so all routes can access them

### `route.py`
- `POST /predict` endpoint
- Accepts a JSON body with exactly 7 days of weather data
- Applies the scaler: `scaled = (raw - mean) / std`
- Passes scaled data to the model
- Returns the predicted drought score

### Why scaling matters in the API
The model was trained on scaled data. If you send raw values (like `PS=100.5`) without scaling, the model receives numbers it has never seen before and gives wrong predictions. The scaler must be applied every time before prediction.

---

## Known Improvements (Planned)

| # | Improvement | Why |
|---|---|---|
| 1 | Fix train/val/test split (70/15/15) | Current code has no true validation set |
| 2 | Add Dropout (0.3) | Reduce overfitting gap |
| 3 | Early stopping (patience=5) | Stop training at best val loss |
| 4 | More GRU layers | Learn more complex patterns |
| 5 | Larger window (14 or 30 days) | Drought builds slowly over weeks |
| 6 | Add RMSE + MAE metrics | MSE alone is hard to interpret |
| 7 | Learning rate scheduler | Fine-tune near end of training |
| 8 | MLflow experiment tracking | Professional portfolio logging |

---

## Tech Stack

| Layer | Tool |
|---|---|
| Big data processing | PySpark |
| Deep learning | PyTorch |
| API framework | FastAPI |
| Data format | Parquet |
| Model serialization | PyTorch `.pth` + Pickle |
| GPU acceleration | CUDA |
