#!/usr/bin/env python3
"""
plot_model_comparisons.py

Loads the trained LSTM, FNN, and TRN models and compares their predictions visually
on the same DS18B20 temperature dataset.

Outputs:
 - combined_predictions.png (overlay plot)
 - comparison_metrics.png (bar chart of MAE, RMSE, MAPE)
"""

import os
import numpy as np
import pandas as pd
import joblib
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error, mean_squared_error
import tensorflow as tf

# ---------- CONFIG ----------
MODEL_DIR = 'models'
DATA_FILE = 'ds18b20_data.csv'
SEQ_LEN = 60
TRAIN_RATIO = 0.8
COMBINED_PLOT = os.path.join(MODEL_DIR, 'combined_predictions.png')
METRICS_PLOT = os.path.join(MODEL_DIR, 'comparison_metrics.png')
# ----------------------------

def find_temp_column(df):
    for col in df.columns:
        if 'temp' in col.lower():
            return col
    numcols = [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])]
    return numcols[1] if len(numcols) >= 2 else (numcols[0] if numcols else None)

def create_sequences(scaled, seq_len):
    X, y = [], []
    for i in range(seq_len, len(scaled)):
        X.append(scaled[i-seq_len:i])
        y.append(scaled[i])
    return np.array(X), np.array(y)

def create_sequences_with_split(scaled, seq_len, train_ratio=0.8):
    X, y = create_sequences(scaled, seq_len)
    split_idx = int(len(X) * train_ratio)
    X_train, X_val = X[:split_idx], X[split_idx:]
    y_train, y_val = y[:split_idx], y[split_idx:]
    return X_train, X_val, y_train, y_val

print("Loading data and models...")
df = pd.read_csv(DATA_FILE)
temp_col = find_temp_column(df)
df = df[[temp_col]].dropna().reset_index(drop=True)

# Load scaler
scaler = joblib.load(os.path.join(MODEL_DIR, 'scaler.save'))
scaled = scaler.transform(df[[temp_col]].values.astype(float))
_, X_val, _, y_val = create_sequences_with_split(scaled, SEQ_LEN, TRAIN_RATIO)

# Load models
models = {
    'LSTM': tf.keras.models.load_model(os.path.join(MODEL_DIR, 'LSTM.keras')),
    'FNN': tf.keras.models.load_model(os.path.join(MODEL_DIR, 'FNN.keras')),
    'TRN': tf.keras.models.load_model(os.path.join(MODEL_DIR, 'TRN.keras')),
}

results = {}
predictions_all = {}
y_true_orig = scaler.inverse_transform(y_val.reshape(-1,1)).flatten()

for name, model in models.items():
    print(f"Predicting with {name}...")
    y_pred = model.predict(X_val, verbose=0)
    y_pred_orig = scaler.inverse_transform(y_pred).flatten()
    predictions_all[name] = y_pred_orig

    mae = mean_absolute_error(y_true_orig, y_pred_orig)
    rmse = np.sqrt(mean_squared_error(y_true_orig, y_pred_orig))
    mape = np.mean(np.abs((y_true_orig - y_pred_orig) / np.maximum(np.abs(y_true_orig), 1e-8))) * 100
    results[name] = (mae, rmse, mape)
    print(f"{name}: MAE={mae:.4f} °C, RMSE={rmse:.4f} °C, MAPE={mape:.2f}%")

# --- Combined line plot ---
print("Generating combined comparison plot...")
plt.figure(figsize=(14,7))
plot_points = min(300, len(y_true_orig))
plt.plot(range(plot_points), y_true_orig[:plot_points], label='Actual', linewidth=2)
for name, preds in predictions_all.items():
    plt.plot(range(plot_points), preds[:plot_points], linestyle='--', linewidth=1.8, label=f'{name} Predicted')
plt.title('DS18B20 Temperature Prediction Comparison (LSTM vs FNN vs TRN)')
plt.xlabel('Time Steps')
plt.ylabel('Temperature (°C)')
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(COMBINED_PLOT, dpi=150, bbox_inches='tight')
plt.close()
print(f"Saved combined prediction plot: {COMBINED_PLOT}")

# --- Metrics bar chart ---
print("Generating metrics bar chart...")
models_list = list(results.keys())
mae_vals = [results[m][0] for m in models_list]
rmse_vals = [results[m][1] for m in models_list]
mape_vals = [results[m][2] for m in models_list]

x = np.arange(len(models_list))
width = 0.25

plt.figure(figsize=(10,6))
plt.bar(x - width, mae_vals, width, label='MAE (°C)')
plt.bar(x, rmse_vals, width, label='RMSE (°C)')
plt.bar(x + width, mape_vals, width, label='MAPE (%)')
plt.xticks(x, models_list)
plt.title('Performance Comparison: LSTM vs FNN vs TRN')
plt.ylabel('Error Metrics')
plt.legend()
plt.grid(axis='y', alpha=0.3)
plt.tight_layout()
plt.savefig(METRICS_PLOT, dpi=150, bbox_inches='tight')
plt.close()
print(f"Saved comparison metrics plot: {METRICS_PLOT}")

print("\n=== Final Comparison ===")
print("Model\tMAE(°C)\tRMSE(°C)\tMAPE(%)")
for name, (mae, rmse, mape) in results.items():
    print(f"{name}\t{mae:.4f}\t{rmse:.4f}\t{mape:.2f}")
