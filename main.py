import warnings
import time
import pandas as pd
import numpy as np
import statsmodels.api as sm
from statsmodels.tsa.arima.model import ARIMA
from pmdarima import auto_arima
from sklearn.metrics import mean_squared_error, mean_absolute_error
import seaborn as sns
import matplotlib.pyplot as plt

df=pd.read_csv("T1_06_30_2021_2 (1).csv")
df['datetime'] = pd.to_datetime(df['Date'] + ' ' +df['Time'])
plt.figure(figsize=(15, 8))
plt.plot(df['datetime'], df['Speed'], label='Speed (Km/h)', color='b')
plt.xlabel('Time')
plt.ylabel('Speed (Km/h)')
plt.title('Speed over Time(COMPLETE)')
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
    """
    Runs ARIMA on rolling intervals ensuring stationarity.
    Uses AutoARIMA to select the best (p, d, q) values.
    Evaluates forecasts using RMSE, MAE, MAPE, and SMAPE.
    Tracks execution time for model fitting and forecasting.
    """
    forecast_horizon = 10  # Explicitly set forecast horizon to 10
    total_start_time = time.time()

    if "datetime" not in df.columns:
        raise ValueError("Column 'datetime' is missing from the dataframe.")

    df["datetime"] = pd.to_datetime(df["datetime"], errors="coerce")
    df = df.dropna(subset=["datetime"]).sort_values(by="datetime").set_index("datetime")

    detected_freq = pd.infer_freq(df.index) or "s"
    series = df[target_col].astype(float)
    forecasts = []

    # Measure time complexity for 1-step ahead prediction
    single_step_times = []

    for start in range(0, len(series) - interval, forecast_horizon):
        end = start + interval
        if end + forecast_horizon >= len(series):
            break

        train_data, test_data = (
            series.iloc[start:end],
            series.iloc[end : end + forecast_horizon],
        )

        try:
            # Select best ARIMA order using AutoARIMA
            p, d, q = auto_select_arima(train_data, p_range, d_range, q_range)

            # Fit the best ARIMA model
            model_fit_start = time.time()
            model = ARIMA(train_data, order=(p, d, q))
            fitted_model = model.fit()
            model_fit_end = time.time()

            # Forecast for the defined horizon
            forecast_start = time.time()
            forecast = fitted_model.forecast(steps=forecast_horizon)
            forecast_end = time.time()

            # Measure time complexity for a single-step prediction
            single_step_start = time.time()
            _ = fitted_model.forecast(steps=1)
            single_step_end = time.time()
            single_step_times.append(single_step_end - single_step_start)

            # Store forecast results
            forecast_index = pd.date_range(
                start=series.index[end], periods=forecast_horizon, freq=detected_freq
            )
            forecast_series = pd.Series(forecast.values, index=forecast_index)

            # Compute error metrics
            errors = evaluate_forecast(test_data.values, forecast_series.values)

            forecasts.append((start, forecast_series, errors))

            print(
                f"Interval {start}-{end}: Best ARIMA({p},{d},{q}) Forecast -> {forecast_series.values}"
            )
            print(
                f"Evaluation -> RMSE: {errors['RMSE']:.4f}, MAE: {errors['MAE']:.4f}, MAPE: {errors['MAPE']:.2f}%, SMAPE: {errors['SMAPE']:.2f}%"
            )
            print(
                f"Model Fit Time: {model_fit_end - model_fit_start:.4f} sec | Forecast Time: {forecast_end - forecast_start:.4f} sec\n"
            )

        except Exception as e:
            print(f"ARIMA model failed at interval {start}-{end} due to: {str(e)}")
            continue

    total_end_time = time.time()
    avg_single_step_time = np.mean(single_step_times) if single_step_times else None

    print(
        f"Total execution time for test_arima: {total_end_time - total_start_time:.4f} sec"
    )
    print(
        f"Average time complexity for 1-step ahead forecast: {avg_single_step_time:.6f} sec"
        if avg_single_step_time
        else "No 1-step forecast timings recorded"
    )

    return forecasts


# Example usage:
forecasts = test_arima(
    df, target_col="Speed", interval=200, p_range=(0, 5), d_range=(0, 2), q_range=(0, 5)
)
