import warnings
import time
import pandas as pd
import numpy as np
import statsmodels.api as sm
from statsmodels.tsa.arima.model import ARIMA
from pmdarima import auto_arima
from sklearn.metrics import mean_squared_error, mean_absolute_error
import matplotlib.pyplot as plt

df = pd.read_csv("T1_06_30_2021_2 (1).csv")
df["datetime"] = pd.to_datetime(df["Date"] + " " + df["Time"])
plt.figure(figsize=(15, 8))
plt.plot(df["datetime"], df["Speed"], label="Speed (Km/h)", color="b")
plt.xlabel("Time")
plt.ylabel("Speed (Km/h)")
plt.title("Speed over Time(COMPLETE)")
plt.xticks(rotation=45)
plt.grid(True)
plt.legend()
plt.show()


# Suppress warnings
warnings.simplefilter(action="ignore", category=FutureWarning)
warnings.simplefilter(action="ignore", category=UserWarning)


def evaluate_forecast(actual, predicted):
    """Computes evaluation metrics for time series forecasting."""
    mse = mean_squared_error(actual, predicted)
    rmse = mse**0.5
    mae = mean_absolute_error(actual, predicted)
    mape = np.mean(np.abs((actual - predicted) / actual)) * 100
    smape = 100 * np.mean(
        2 * np.abs(actual - predicted) / (np.abs(actual) + np.abs(predicted))
    )

    return {"RMSE": rmse, "MAE": mae, "MAPE": mape, "SMAPE": smape}


def auto_select_arima(series, p_range=(0, 5), d_range=(0, 2), q_range=(0, 5)):
    """Uses AutoARIMA to find the best (p, d, q) values based on BIC."""
    start_time = time.time()

    model = auto_arima(
        series,
        start_p=p_range[0],
        max_p=p_range[1],
        start_d=d_range[0],
        max_d=d_range[1],
        start_q=q_range[0],
        max_q=q_range[1],
        seasonal=False,
        stepwise=True,
        suppress_warnings=True,
        trace=True,
    )

    end_time = time.time()
    print(f"AutoARIMA selected order {model.order} in {end_time - start_time:.4f} sec")

    return model.order


def test_arima(
    df, target_col="Speed", interval=200, p_range=(0, 5), d_range=(0, 2), q_range=(0, 5)
):
    forecast_horizon = 10
    total_start_time = time.time()

    if "datetime" not in df.columns:
        raise ValueError("Column 'datetime' is missing from the dataframe.")

    df["datetime"] = pd.to_datetime(df["datetime"], errors="coerce")
    df = df.dropna(subset=["datetime"]).sort_values(by="datetime").set_index("datetime")

    detected_freq = pd.infer_freq(df.index) or "s"
    full_series = df[target_col].astype(float)
    forecasts = []
    single_step_times = []

    p, d, q = None, None, None
    model = None
    fitted_model = None

    start = 0
    while start + interval + forecast_horizon < len(full_series):
        end = start + interval
        train_data = full_series.iloc[start:end]
        test_data = full_series.iloc[end : end + forecast_horizon]

        if fitted_model is None:
            p, d, q = auto_select_arima(train_data, p_range, d_range, q_range)
            model = ARIMA(train_data, order=(p, d, q))
            fitted_model = model.fit()

        forecast_start = time.time()
        forecast = fitted_model.forecast(steps=forecast_horizon)
        forecast_end = time.time()

        t1 = time.time()
        _ = fitted_model.forecast(steps=1)
        t2 = time.time()
        single_step_times.append(t2 - t1)

        forecast_index = pd.date_range(
            start=full_series.index[end], periods=forecast_horizon, freq=detected_freq
        )
        forecast_series = pd.Series(forecast.values, index=forecast_index)
        errors = evaluate_forecast(test_data.values, forecast_series.values)
        smape = errors["SMAPE"]

        print(f"Interval {start}-{end}: Best ARIMA({p},{d},{q}) Forecast")
        print(f"SMAPE: {smape:.2f}%")

        forecasts.append((start, forecast_series, errors))

        # --- ⬇️ PLOT HERE
        plt.figure(figsize=(15, 5))
        plt.plot(
            full_series.index[start:end],
            full_series.iloc[start:end],
            label="Training Data",
            color="blue",
        )
        plt.plot(
            test_data.index,
            test_data.values,
            label="Actual (Next 10)",
            color="green",
            marker="o",
        )
        plt.plot(
            forecast_series.index,
            forecast_series.values,
            label="Forecast (Next 10)",
            color="red",
            linestyle="dashed",
            marker="x",
        )
        plt.title(f"Prediction at Interval {start}-{end}")
        plt.xlabel("Time")
        plt.ylabel("Speed")
        plt.legend()
        plt.grid(True)
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.show()

        time.sleep(1)

        # Decide whether to retrain or not
        if smape > 15:
            print(
                " SMAPE > 15% — Retraining model with all data up to this point and using actual values"
            )
            start += forecast_horizon
            p, d, q = auto_select_arima(
                full_series.iloc[: start + interval], p_range, d_range, q_range
            )
            model = ARIMA(full_series.iloc[: start + interval], order=(p, d, q))
            fitted_model = model.fit()

        else:
            print(" SMAPE ≤ 15% — Reusing model and using forecasted values")
            new_values = pd.Series(forecast.values, index=forecast_index)
            full_series = pd.concat([full_series, new_values])
            start += forecast_horizon

        print(f"Model Fit Time: {forecast_end - forecast_start:.4f} sec\n")

    total_end_time = time.time()
    avg_step_time = np.mean(single_step_times) if single_step_times else None

    print(f"\nTotal execution time: {total_end_time - total_start_time:.2f} sec")
    print(
        f"Average 1-step forecast time: {avg_step_time:.6f} sec"
        if avg_step_time
        else "No 1-step forecast time recorded"
    )

    return forecasts


# Example usage:
forecasts = test_arima(
    df, target_col="Speed", interval=200, p_range=(0, 5), d_range=(0, 2), q_range=(0, 5)
)
