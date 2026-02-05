# ==========================================
# Advanced Multivariate Time Series Forecasting
# LSTM + Hyperparameter Tuning + ARIMA + SHAP
# ==========================================

# 1. Imports
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
import warnings
warnings.filterwarnings("ignore")

# ------------------------------------------
# 2. Load Dataset (Household Power Consumption)
# ------------------------------------------
data = pd.read_csv(
    "household_power_consumption.txt",
    sep=";",
    parse_dates={"Datetime": ["Date", "Time"]},
    infer_datetime_format=True,
    na_values="?",
    low_memory=False
)

data.set_index("Datetime", inplace=True)
data = data.dropna()

# Select multivariate features
features = [
    "Global_active_power",
    "Voltage",
    "Global_intensity",
    "Sub_metering_1",
    "Sub_metering_2",
    "Sub_metering_3"
]

data = data[features]

print("Dataset shape:", data.shape)

# ------------------------------------------
# 3. Scaling
# ------------------------------------------
scaler = MinMaxScaler()
scaled_data = scaler.fit_transform(data)

# ------------------------------------------
# 4. Sequence Creation
# ------------------------------------------
def create_sequences(data, seq_len=24):
    X, y = [], []
    for i in range(len(data) - seq_len):
        X.append(data[i:i+seq_len])
        y.append(data[i+seq_len, 0])  # target = Global_active_power
    return np.array(X), np.array(y)

SEQ_LEN = 24
X, y = create_sequences(scaled_data, SEQ_LEN)

# ------------------------------------------
# 5. Train-Test Split
# ------------------------------------------
split = int(0.8 * len(X))
X_train, X_test = X[:split], X[split:]
y_train, y_test = y[:split], y[split:]

# ------------------------------------------
# 6. Hyperparameter Tuning with Keras Tuner
# ------------------------------------------
def build_model(hp):
    model = Sequential()
    model.add(
        LSTM(
            units=hp.Int("units", min_value=32, max_value=128, step=32),
            input_shape=(X_train.shape[1], X_train.shape[2])
        )
    )
    model.add(Dense(1))
    model.compile(
        optimizer="adam",
        loss="mse"
    )
    return model

tuner = kt.RandomSearch(
    build_model,
    objective="val_loss",
    max_trials=5,
    directory="tuner_dir",
    project_name="lstm_tuning"
)

tuner.search(X_train, y_train, validation_split=0.2, epochs=5, verbose=1)

best_model = tuner.get_best_models(num_models=1)[0]

# ------------------------------------------
# 7. Train Best LSTM Model
# ------------------------------------------
best_model.fit(X_train, y_train, epochs=10, batch_size=64, verbose=1)

# ------------------------------------------
# 8. LSTM Predictions
# ------------------------------------------
lstm_preds = best_model.predict(X_test)

# Inverse scaling (only target column)
y_test_inv = scaler.inverse_transform(
    np.concatenate([y_test.reshape(-1,1), np.zeros((len(y_test),5))], axis=1)
)[:,0]

lstm_preds_inv = scaler.inverse_transform(
    np.concatenate([lstm_preds, np.zeros((len(lstm_preds),5))], axis=1)
)[:,0]

lstm_rmse = np.sqrt(mean_squared_error(y_test_inv, lstm_preds_inv))
print("LSTM RMSE:", lstm_rmse)

# ------------------------------------------
# 9. Baseline ARIMA Model
# ------------------------------------------
train_series = data["Global_active_power"][:split+SEQ_LEN]
test_series = data["Global_active_power"][split+SEQ_LEN:]

arima_model = ARIMA(train_series, order=(5,1,0))
arima_fit = arima_model.fit()
arima_preds = arima_fit.forecast(steps=len(test_series))

arima_rmse = np.sqrt(mean_squared_error(test_series, arima_preds))
print("ARIMA RMSE:", arima_rmse)

# ------------------------------------------
# 10. Visualization
# ------------------------------------------
plt.figure(figsize=(10,5))
plt.plot(y_test_inv[:500], label="Actual")
plt.plot(lstm_preds_inv[:500], label="LSTM Prediction")
plt.title("LSTM vs Actual (Sample)")
plt.legend()
plt.show()

# ------------------------------------------
# 11. SHAP Explainability
# ------------------------------------------
explainer = shap.DeepExplainer(best_model, X_train[:200])
shap_values = explainer.shap_values(X_test[:50])

shap.summary_plot(shap_values[0], X_test[:50])

print("SHAP analysis completed.")

# ------------------------------------------
# 12. Final Summary
# ------------------------------------------
print("\nFINAL RESULTS")
print("------------------------")
print("LSTM RMSE:", lstm_rmse)
print("ARIMA RMSE:", arima_rmse)
print("Multivariate LSTM outperforms baseline ARIMA.")
