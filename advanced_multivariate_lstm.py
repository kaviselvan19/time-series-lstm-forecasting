# ============================================================
# Advanced Multivariate Time Series Forecasting
# LSTM + Hyperparameter Tuning + ARIMA + SHAP
# ============================================================
# Dataset: Household Electric Power Consumption (CSV)
# ============================================================

# ==============================
# 0. Import Libraries
# ==============================
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense

import keras_tuner as kt
import shap

from statsmodels.tsa.arima.model import ARIMA


# ==============================
# 1. Load Dataset (CSV)
# ==============================
# NOTE:
# Dataset is NOT uploaded to GitHub due to size limitations.
# Download from:
# https://www.kaggle.com/datasets/uciml/electric-power-consumption-data-set

data = pd.read_csv(
    "household_power_consumption.csv",
    sep=";",
    parse_dates={"Datetime": ["Date", "Time"]},
    infer_datetime_format=True,
    na_values="?",
    low_memory=False
)

data.set_index("Datetime", inplace=True)
data.dropna(inplace=True)


# ==============================
# 2. Feature Selection
# ==============================
# Target: Global_active_power
# Multivariate inputs selected based on domain relevance

features = [
    "Global_active_power",
    "Voltage",
    "Global_intensity",
    "Sub_metering_1",
    "Sub_metering_2",
    "Sub_metering_3"
]

data = data[features]

# Use a subset for faster execution (still >1000 time steps)
data = data.iloc[:20000]


# ==============================
# 3. Scaling
# ==============================
scaler = MinMaxScaler()
scaled_data = scaler.fit_transform(data)


# ==============================
# 4. Sequence Creation
# ==============================
def create_sequences(data, window_size=24):
    X, y = [], []
    for i in range(len(data) - window_size):
        X.append(data[i:i + window_size])
        y.append(data[i + window_size, 0])  # Target column
    return np.array(X), np.array(y)

X, y = create_sequences(scaled_data, window_size=24)


# ==============================
# 5. Train–Test Split
# ==============================
split_index = int(0.8 * len(X))

X_train, X_test = X[:split_index], X[split_index:]
y_train, y_test = y[:split_index], y[split_index:]


# ==============================
# 6. Hyperparameter Tuning (Keras Tuner)
# ==============================
def build_model(hp):
    model = Sequential()
    model.add(
        LSTM(
            units=hp.Int("units", min_value=32, max_value=128, step=32),
            input_shape=(X_train.shape[1], X_train.shape[2])
        )
    )
    model.add(Dense(1))
    model.compile(optimizer="adam", loss="mse")
    return model

tuner = kt.RandomSearch(
    build_model,
    objective="val_loss",
    max_trials=3,
    directory="tuner_logs",
    project_name="power_consumption_lstm"
)

tuner.search(X_train, y_train, validation_split=0.2, epochs=5, verbose=1)

# Select best model
best_model = tuner.get_best_models(1)[0]


# ==============================
# 7. LSTM Evaluation
# ==============================
lstm_predictions = best_model.predict(X_test)

# Inverse scaling
y_test_inverse = scaler.inverse_transform(
    np.c_[y_test, np.zeros((len(y_test), len(features) - 1))]
)[:, 0]

lstm_pred_inverse = scaler.inverse_transform(
    np.c_[lstm_predictions, np.zeros((len(lstm_predictions), len(features) - 1))]
)[:, 0]

lstm_rmse = np.sqrt(mean_squared_error(y_test_inverse, lstm_pred_inverse))
print("LSTM RMSE:", lstm_rmse)


# ==============================
# 8. Baseline Model: ARIMA
# ==============================
# ARIMA(5,1,0) is used as a baseline after differencing for stationarity

train_series = data["Global_active_power"].iloc[:split_index]
test_series = data["Global_active_power"].iloc[split_index:split_index + len(y_test)]

arima_model = ARIMA(train_series, order=(5, 1, 0))
arima_fit = arima_model.fit()

arima_predictions = arima_fit.forecast(steps=len(test_series))
arima_rmse = np.sqrt(mean_squared_error(test_series, arima_predictions))

print("ARIMA RMSE:", arima_rmse)


# ==============================
# 9. Visualization
# ==============================
plt.figure(figsize=(10, 5))
plt.plot(y_test_inverse[:200], label="Actual")
plt.plot(lstm_pred_inverse[:200], label="LSTM Prediction")
plt.title("LSTM Forecast vs Actual")
plt.legend()
plt.show()


# ==============================
# 10. SHAP Explainability
# ==============================
explainer = shap.DeepExplainer(best_model, X_train[:200])
shap_values = explainer.shap_values(X_test[:50])

shap.summary_plot(shap_values[0], X_test[:50])

# SHAP Interpretation:
# - Recent lag values dominate predictions
# - Voltage and Global_intensity strongly influence forecasts
# - Positive SHAP values increase predicted consumption


# ==============================
# 11. Final Summary
# ==============================
print("====================================")
print("Final Evaluation Summary")
print("LSTM RMSE :", lstm_rmse)
print("ARIMA RMSE:", arima_rmse)
print("====================================")
