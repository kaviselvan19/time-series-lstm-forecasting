## Dataset

This project uses the **Household Electric Power Consumption Dataset**.

Due to GitHub file size limitations, the dataset is not included in this repository.

Dataset source (Kaggle):
https://www.kaggle.com/datasets/uciml/electric-power-consumption-data-set

Please download the dataset and place `household_power_consumption.csv`
in the project root directory before running the code.


# Advanced Multivariate Time Series Forecasting

## Dataset
This project uses the Household Electric Power Consumption dataset from Kaggle:
https://www.kaggle.com/datasets/uciml/electric-power-consumption-data-set

The dataset exceeds GitHub’s file size limits and must be downloaded manually.
Place `household_power_consumption.csv` in the project root directory.

---

## Data Characteristics
The dataset contains over two million time-indexed observations with multiple
electrical measurements. Missing values are removed, and all features are
normalized using MinMax scaling.

---

## Model Architecture
A stacked multivariate LSTM architecture is used with two recurrent layers.
This design enables learning both short-term and long-term temporal
dependencies across correlated electrical features.

---

## Hyperparameter Optimization
Keras Tuner RandomSearch is used to optimize LSTM units. Multiple trials and
epochs are executed, and the best configuration is explicitly selected and
used for final evaluation.

---

## Baseline Model
A traditional ARIMA(5,1,0) model is implemented as a statistical baseline.
First-order differencing ensures stationarity, while autoregressive terms
capture short-term dependencies.

---

## Explainability
SHAP DeepExplainer is applied using a large background dataset.
Feature importance rankings are printed numerically and visualized,
demonstrating that recent lagged power values dominate predictions, with
voltage and intensity providing significant corrective influence.

---

## Execution
Run the project using:

```bash
python advanced_multivariate_lstm.py

