# ===============================
# Advanced Time Series Forecasting
# Using LSTM + Explainability
# ===============================

# 1. Import Libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense

import shap

# 2. Load Dataset
# Make sure AirPassengers.csv is in the same folder
data = pd.read_csv("AirPassengers.csv")

data['Month'] = pd.to_datetime(data['Month'])
data.set_index('Month', inplace=True)

print("Dataset Loaded Successfully")
print(data.head())

# 3. Scale the Data
scaler = MinMaxScaler(feature_range=(0, 1))
scaled_data = scaler.fit_transform(data)

# 4. Create Sequences
def create_sequences(data, seq_len=12):
    X, y = [], []
    for i in range(len(data) - seq_len):
        X.append(data[i:i + seq_len])
        y.append(data[i + seq_len])
    return np.array(X), np.array(y)

X, y = create_sequences(scaled_data, seq_len=12)

# 5. Train-Test Split (80-20)
split = int(0.8 * len(X))

X_train, X_test = X[:split], X[split:]
y_train, y_test = y[:split], y[split:]

# 6. Build LSTM Model
model = Sequential()
model.add(LSTM(50, activation='relu', input_shape=(X_train.shape[1], 1)))
model.add(Dense(1))

model.compile(optimizer='adam', loss='mse')

print("\nTraining Model...")
model.fit(X_train, y_train, epochs=50, batch_size=16, verbose=1)

# 7. Predictions
predictions = model.predict(X_test)

# Inverse scaling
predictions = scaler.inverse_transform(predictions)
y_test_inv = scaler.inverse_transform(y_test)

# 8. Model Evaluation
rmse = np.sqrt(mean_squared_error(y_test_inv, predictions))
print("\nRMSE:", rmse)

# 9. Visualization
plt.figure(figsize=(10, 5))
plt.plot(y_test_inv, label="Actual Values")
plt.plot(predictions, label="Predicted Values")
plt.xlabel("Time")
plt.ylabel("Passengers")
plt.title("LSTM Time Series Forecasting")
plt.legend()
plt.show()

# 10. Explainability using SHAP
print("\nGenerating SHAP Explainability...")

explainer = shap.DeepExplainer(model, X_train[:50])
shap_values = explainer.shap_values(X_test[:10])

shap.summary_plot(shap_values[0], X_test[:10])

print("\nProject Execution Completed Successfully")
