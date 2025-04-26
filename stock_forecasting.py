# ================================================================
# Project Title: Forecasting Stock Prices with ARIMA and Prophet Models
# Description: This project applies time series analysis techniques
# (ARIMA and Prophet) to predict future stock prices based on historical data.
# Dataset: Microsoft Stock Data (Open, High, Low, Close, Volume)

# College Mini Project Report Form
# Project Title: Forecasting Stock Prices with ARIMA and Prophet Models
# Student Name: Orozaliev Adilet
# Course: Computer Science
# Group: CS-21
# Advanced Python Project - Stock Price Prediction
# ================================================================

# ----------------------------
# Install dependencies
# ----------------------------
# pip install -r requirements.txt

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
from statsmodels.tsa.stattools import adfuller
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.tsa.arima.model import ARIMA
from prophet import Prophet
from sklearn.metrics import mean_squared_error, mean_absolute_error
import warnings

warnings.filterwarnings("ignore")
plt.style.use('seaborn-v0_8')

# Create necessary directories
os.makedirs("plots", exist_ok=True)
os.makedirs("outputs", exist_ok=True)

# ----------------------------
# 1. Load the data
# ----------------------------

file_path = 'MSFT_stock.csv'

# Чтение файла без пропуска строк
df = pd.read_csv(file_path)

# Убираем лишние строки с "Ticker" и "Date"
df = df[~df['Price'].isin(['Ticker', 'Date'])]

# Преобразуем колонку 'Price' в datetime
df['Price'] = pd.to_datetime(df['Price'])

# Ставим дату как индекс
df.set_index('Price', inplace=True)
df.sort_index(inplace=True)

# Преобразуем столбцы к числовому типу, если нужно
df[['Close', 'High', 'Low', 'Open', 'Volume']] = df[['Close', 'High', 'Low', 'Open', 'Volume']].astype(float)

df.fillna(method='ffill', inplace=True)

# ----------------------------
# 2. Visualize Data
# ----------------------------

plt.figure(figsize=(14,7))
plt.plot(df['Close'])
plt.title('Stock Closing Price')
plt.xlabel('Date')
plt.ylabel('Price')
plt.savefig("plots/closing_price.png")
plt.show()

# ----------------------------
# 3. Stationarity Check
# ----------------------------

result = adfuller(df['Close'])
print('ADF Statistic:', result[0])
print('p-value:', result[1])

if result[1] > 0.05:
    print("Time Series is NOT stationary. Applying differencing.")
    df['Close_diff'] = df['Close'].diff()
    df.dropna(inplace=True)
else:
    print("Time Series is Stationary.")
    df['Close_diff'] = df['Close']

# ----------------------------
# 4. ACF and PACF Plots
# ----------------------------

fig, axes = plt.subplots(1,2, figsize=(18,5))
plot_acf(df['Close_diff'], lags=40, ax=axes[0])
plot_pacf(df['Close_diff'], lags=40, ax=axes[1])
plt.savefig("plots/acf_pacf_plots.png")
plt.show()

# ----------------------------
# 5. Train-Test Split
# ----------------------------

train_size = int(len(df) * 0.8)
train, test = df.iloc[:train_size], df.iloc[train_size:]

# ----------------------------
# 6. ARIMA Model
# ----------------------------

model_arima = ARIMA(train['Close'], order=(5,1,0))
model_arima_fit = model_arima.fit()
forecast_arima = model_arima_fit.forecast(steps=len(test))

# ----------------------------
# 7. Prophet Model
# ----------------------------

train_prophet = train.reset_index()[['Price', 'Close']].rename(columns={'Price': 'ds', 'Close': 'y'})
test_prophet = test.reset_index()[['Price', 'Close']].rename(columns={'Price': 'ds', 'Close': 'y'})

model_prophet = Prophet()
model_prophet.fit(train_prophet)

future = model_prophet.make_future_dataframe(periods=len(test))
forecast_prophet = model_prophet.predict(future)
forecast_prophet = forecast_prophet.set_index('ds')
prophet_forecast = pd.merge_asof(test.reset_index(), forecast_prophet[['yhat']], left_on='Price', right_index=True)['yhat'].values

# ----------------------------
# 8. Evaluation
# ----------------------------

def evaluate(y_true, y_pred, model_name):
    mse = mean_squared_error(y_true, y_pred)
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    print(f"{model_name} - MSE: {mse:.2f}, MAE: {mae:.2f}, RMSE: {rmse:.2f}")
    return mse, mae, rmse

metrics = {}

metrics['ARIMA'] = evaluate(test['Close'], forecast_arima, "ARIMA")
metrics['Prophet'] = evaluate(test['Close'].values, prophet_forecast, "Prophet")

# Save metrics
with open("outputs/metrics.txt", "w") as f:
    for model, (mse, mae, rmse) in metrics.items():
        f.write(f"{model} - MSE: {mse:.2f}, MAE: {mae:.2f}, RMSE: {rmse:.2f}\n")

# Save forecasts
pd.DataFrame({"Actual": test['Close'], "ARIMA Forecast": forecast_arima}).to_csv("outputs/forecast_arima.csv")
pd.DataFrame({"Actual": test['Close'].values, "Prophet Forecast": prophet_forecast}).to_csv("outputs/forecast_prophet.csv")

# ----------------------------
# 9. Plot Results
# ----------------------------

plt.figure(figsize=(14,7))
plt.plot(test.index, test['Close'], label='Actual')
plt.plot(test.index, forecast_arima, label='ARIMA Predictions')
plt.plot(test.index, prophet_forecast, label='Prophet Predictions')
plt.legend()
plt.title('Forecast vs Actuals')
plt.savefig("plots/forecast_vs_actuals.png")
plt.show()

# ----------------------------
# 10. Bar Chart of Errors
# ----------------------------

metrics_df = pd.DataFrame(metrics, index=['MSE', 'MAE', 'RMSE']).T

metrics_df[['RMSE']].plot(kind='bar', figsize=(10,6))
plt.title('Model Comparison (Lower RMSE Better)')
plt.ylabel('RMSE')
plt.xticks(rotation=0)
plt.savefig("plots/model_errors.png")
plt.show()

# ----------------------------
# 11. Final Full Forecast with Prophet
# ----------------------------

full_data = df.reset_index()[['Price', 'Close']].rename(columns={'Price': 'ds', 'Close': 'y'})
full_model = Prophet()
full_model.fit(full_data)

future_full = full_model.make_future_dataframe(periods=90)
forecast_full = full_model.predict(future_full)

full_model.plot(forecast_full)
plt.title('90 Days Future Forecast')
plt.savefig("plots/forecast_90_days.png")
plt.show()

# ----------------------------
# Done!
# ----------------------------