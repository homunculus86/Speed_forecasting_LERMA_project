# ARIMA Time Series Forecasting: Speed Prediction

## 📌 Objective

This project uses ARIMA models to forecast vehicle speed from time series data collected in a CSV file. The goal is to predict the next 10 values based on previous data using a sliding window approach, with automatic model retraining if forecast quality drops.

---

## 📥 Dataset

The dataset is a CSV file (`T1_06_30_2021_2.csv`) containing the following columns:
- `Date`: The date of the observation
- `Time`: The time of the observation
- `Speed`: The speed value (in Km/h)

---

## 🚀 How It Works

1. **Data Preprocessing:**
   - Combines `Date` and `Time` into a single datetime column.
   - Converts it to a time series format.

2. **Modeling:**
   - Uses `auto_arima()` to select the best ARIMA `(p, d, q)` parameters.
   - Trains an ARIMA model on a rolling window of past values (default: 200).
   - Predicts the next 10 values.

3. **Dynamic Retraining:**
   - Evaluates the prediction using SMAPE (Symmetric Mean Absolute Percentage Error).
   - If SMAPE > 15%, the model is retrained using all data from the start up to the current point.
   - Otherwise, the forecasted values are appended and used for the next prediction window.

4. **Visualization:**
   - For each prediction, plots:
     - The training data
     - Actual next values
     - Forecasted values

---

## 📦 Imports Required

```python
import warnings
import time
import pandas as pd
import numpy as np
import statsmodels.api as sm
from statsmodels.tsa.arima.model import ARIMA
from pmdarima import auto_arima
from sklearn.metrics import mean_squared_error, mean_absolute_error
import matplotlib.pyplot as plt
