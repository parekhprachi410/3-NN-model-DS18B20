#!/usr/bin/env python3
"""
train_ds18b20_multiple_models_full.py

Train and compare three models (LSTM, FNN, TRN) for DS18B20 temperature prediction.
Generates:
 - Per-model training plots (loss & MAE)
 - Per-model prediction plots
 - Per-model training results text files
 - Per-model TFLite models
 - Combined predictions plot (all three on same axes)
 - A final comparison_results.txt

Usage:
    pip install tensorflow scikit-learn joblib pandas numpy matplotlib
    python train_ds18b20_multiple_models_full.py
"""

import os
import numpy as np
import pandas as pd
import joblib
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, Flatten, Conv1D
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
import traceback

# ---------- CONFIG ----------
DATA_FILE = 'onsiteLSTM200.xlsx'
SEQ_LEN = 60
EPOCHS = 200
BATCH_SIZE = 32
TRAIN_RATIO = 0.8
MODEL_DIR = 'models'
os.makedirs(MODEL_DIR, exist_ok=True)
SCALER_FILE = os.path.join(MODEL_DIR, 'scaler.save')
COMBINED_PLOT = os.path.join(MODEL_DIR, 'combined_predictions.png')
COMPARISON_RESULTS = os.path.join(MODEL_DIR, 'comparison_results.txt')
# ----------------------------

def find_temp_column(df):
    for col in df.columns:
        if 'temp' in col.lower():
            return col
    numcols = [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])]
    return numcols[1] if len(numcols) >= 2 else (numcols[0] if numcols else None)

def enhanced_data_validation(df, temp_col):
    """Remove extreme outliers (beyond 4 stddev)."""
    mean = df[temp_col].mean()
    std = df[temp_col].std()
    df_clean = df[(df[temp_col] >= mean - 4*std) & (df[temp_col] <= mean + 4*std)].copy()
    return df_clean

def load_and_prepare():
    if not os.path.exists(DATA_FILE):
        raise FileNotFoundError(f"{DATA_FILE} not found.")
    df = pd.read_csv(DATA_FILE)
    temp_col = find_temp_column(df)
    if temp_col is None:
        raise ValueError("No temperature column found.")
    df = df[[temp_col]].dropna().reset_index(drop=True)
    df_clean = enhanced_data_validation(df, temp_col)
    return df_clean, temp_col

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

# ------------------- MODEL BUILDERS -------------------

def build_lstm_model(seq_len):
    model = Sequential([
        LSTM(64, return_sequences=True, input_shape=(seq_len, 1)),
        Dropout(0.2),
        LSTM(32, return_sequences=False),
        Dropout(0.2),
        Dense(16, activation='relu'),
        Dropout(0.1),
        Dense(8, activation='relu'),
        Dense(1)
    ])
    model.compile(optimizer=tf.keras.optimizers.Adam(1e-3), loss='mse', metrics=['mae'])
    return model

def build_fnn_model(seq_len):
    model = Sequential([
        Flatten(input_shape=(seq_len,1)),
        Dense(128, activation='relu'),
        Dropout(0.2),
        Dense(64, activation='relu'),
        Dropout(0.1),
        Dense(32, activation='relu'),
        Dense(1)
    ])
    model.compile(optimizer=tf.keras.optimizers.Adam(1e-3), loss='mse', metrics=['mae'])
    return model

def build_trn_model(seq_len):
    model = Sequential([
        Conv1D(32, kernel_size=2, activation='relu', input_shape=(seq_len,1)),
        Conv1D(64, kernel_size=2, activation='relu'),
        Flatten(),
        Dense(64, activation='relu'),
        Dense(1)
    ])
    model.compile(optimizer=tf.keras.optimizers.Adam(1e-3), loss='mse', metrics=['mae'])
    return model

# ------------------- UTILITIES -------------------

def plot_history(history, out_file):
    hist = history.history
    epochs = range(1, len(hist['loss']) + 1)
    plt.figure(figsize=(12,5))
    # Loss
    plt.subplot(1,2,1)
    plt.plot(epochs, hist['loss'], label='Train Loss')
    if 'val_loss' in hist:
        plt.plot(epochs, hist['val_loss'], label='Val Loss')
    plt.title('MSE Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True, alpha=0.3)
    # MAE
    plt.subplot(1,2,2)
    if 'mae' in hist:
        plt.plot(epochs, hist['mae'], label='Train MAE')
    if 'val_mae' in hist:
        plt.plot(epochs, hist.get('val_mae',[]), label='Val MAE')
    plt.title('MAE')
    plt.xlabel('Epoch')
    plt.ylabel('MAE')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_file, dpi=150, bbox_inches='tight')
    plt.close()

def save_training_results_file(model_name, history, metrics, out_file):
    mae, rmse, mape = metrics
    hist = history.history
    final_epoch = len(hist['loss'])
    with open(out_file, 'w') as f:
        f.write(f"{model_name} Training Results\n")
        f.write("="*40 + "\n")
        f.write(f"Epochs trained: {final_epoch}\n")
        f.write(f"Final Train Loss: {hist['loss'][-1]:.6f}\n")
        if 'val_loss' in hist:
            f.write(f"Final Val Loss: {hist['val_loss'][-1]:.6f}\n")
        if 'mae' in hist:
            f.write(f"Final Train MAE: {hist['mae'][-1]:.6f}\n")
        if 'val_mae' in hist:
            f.write(f"Final Val MAE: {hist['val_mae'][-1]:.6f}\n")
        f.write("\nValidation Metrics (original scale):\n")
        f.write(f"MAE: {mae:.4f} °C\n")
        f.write(f"RMSE: {rmse:.4f} °C\n")
        f.write(f"MAPE: {mape:.2f} %\n")

def safe_tflite_conversion_from_model(model, out_path_f16):
    try:
        converter = tf.lite.TFLiteConverter.from_keras_model(model)
        converter.optimizations = [tf.lite.Optimize.DEFAULT]
        converter.target_spec.supported_types = [tf.float16]
        tflite_model = converter.convert()
        with open(out_path_f16, 'wb') as f:
            f.write(tflite_model)
        print(f"Saved TFLite (float16) to {out_path_f16}")
    except Exception as e:
        print(f"TFLite conversion to float16 failed for {out_path_f16}: {e}")
        # fallback to simple convert
        try:
            converter = tf.lite.TFLiteConverter.from_keras_model(model)
            tflite_model = converter.convert()
            fallback = out_path_f16.replace('.tflite', '_simple.tflite')
            with open(fallback, 'wb') as f:
                f.write(tflite_model)
            print(f"Saved simple TFLite fallback to {fallback}")
        except Exception as e2:
            print(f"All TFLite conversion attempts failed: {e2}")

# ------------------- TRAIN/VALIDATE -------------------

def train_model(model, X_train, y_train, X_val, y_val, scaler, model_name):
    model_file = os.path.join(MODEL_DIR, f'{model_name}.keras')
    train_plot = os.path.join(MODEL_DIR, f'{model_name}_training.png')
    pred_plot = os.path.join(MODEL_DIR, f'{model_name}_predictions.png')
    results_file = os.path.join(MODEL_DIR, f'{model_name}_results.txt')
    tflite_out = os.path.join(MODEL_DIR, f'{model_name}.tflite')

    callbacks = [
        EarlyStopping(monitor='val_loss', patience=20, restore_best_weights=True, verbose=1),
        ModelCheckpoint(model_file, save_best_only=True, monitor='val_loss', verbose=1),
        ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=10, min_lr=1e-6, verbose=1)
    ]

    print(f"\n--- Training {model_name} ---")
    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        callbacks=callbacks,
        shuffle=False,
        verbose=1
    )

    # Save model (best saved by checkpoint)
    try:
        model.save(model_file)
    except Exception as e:
        print(f"Warning: could not save model in Keras format: {e}")

    # Plot and save training curves
    try:
        plot_history(history, train_plot)
        print(f"Saved training plot: {train_plot}")
    except Exception as e:
        print(f"Failed to save training plot: {e}")

    # Predict on validation set and inverse-transform
    try:
        predictions = model.predict(X_val, verbose=0)
        y_true_orig = scaler.inverse_transform(y_val.reshape(-1,1)).flatten()
        y_pred_orig = scaler.inverse_transform(predictions).flatten()
    except Exception as e:
        print("Prediction/inverse-transform failed:", e)
        traceback.print_exc()
        raise

    # Metrics
    mae = mean_absolute_error(y_true_orig, y_pred_orig)
    rmse = np.sqrt(mean_squared_error(y_true_orig, y_pred_orig))
    mape = np.mean(np.abs((y_true_orig - y_pred_orig) / np.maximum(np.abs(y_true_orig), 1e-8))) * 100
    print(f"{model_name} - MAE: {mae:.4f} °C, RMSE: {rmse:.4f} °C, MAPE: {mape:.2f} %")

    # Plot predictions (first N points)
    try:
        plt.figure(figsize=(12,6))
        plot_points = min(200, len(y_true_orig))
        plt.plot(range(plot_points), y_true_orig[:plot_points], label='Actual', linewidth=2)
        plt.plot(range(plot_points), y_pred_orig[:plot_points], label=f'{model_name} Predicted', linewidth=2, linestyle='--')
        plt.title(f'{model_name} - Actual vs Predicted (first {plot_points} points)')
        plt.xlabel('Time Steps')
        plt.ylabel('Temperature (°C)')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(pred_plot, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"Saved prediction plot: {pred_plot}")
    except Exception as e:
        print(f"Failed to save prediction plot: {e}")

    # Save results file
    try:
        save_training_results_file(model_name, history, (mae, rmse, mape), results_file)
        print(f"Saved training results to: {results_file}")
    except Exception as e:
        print(f"Failed to write results file: {e}")

    # TFLite conversion
    try:
        safe_tflite_conversion_from_model(model, tflite_out)
    except Exception as e:
        print(f"TFLite conversion failed for {model_name}: {e}")

    return history, (mae, rmse, mape), y_true_orig, y_pred_orig

# ------------------- MAIN -------------------

if __name__ == '__main__':
    print("Loading and preparing data...")
    try:
        df, temp_col = load_and_prepare()
    except Exception as e:
        print("Failed to load/prepare data:", e)
        raise

    print(f"Samples after cleaning: {len(df)} (using column '{temp_col}')")

    # Scale
    scaler = MinMaxScaler()
    scaled = scaler.fit_transform(df[[temp_col]].values.astype(float))
    joblib.dump(scaler, SCALER_FILE)
    print(f"Saved scaler to {SCALER_FILE}")

    # Create sequences and split
    X_train, X_val, y_train, y_val = create_sequences_with_split(scaled, SEQ_LEN, TRAIN_RATIO)
    print(f"X_train: {X_train.shape}, X_val: {X_val.shape}")

    # Instantiate models
    models_info = [
        ('LSTM', build_lstm_model(SEQ_LEN)),
        ('FNN', build_fnn_model(SEQ_LEN)),
        ('TRN', build_trn_model(SEQ_LEN))
    ]

    results = {}
    preds_for_combined = {}
    y_true_for_combined = None

    # Train each model
    for name, model in models_info:
        try:
            history, metrics, y_true_orig, y_pred_orig = train_model(
                model, X_train, y_train, X_val, y_val, scaler, name
            )
            results[name] = metrics
            preds_for_combined[name] = y_pred_orig
            # store y_true once
            if y_true_for_combined is None:
                y_true_for_combined = y_true_orig
        except Exception as e:
            print(f"Training/eval failed for {name}: {e}")
            traceback.print_exc()

    # Create combined predictions plot (overlay models)
    try:
        if y_true_for_combined is None:
            raise RuntimeError("No predictions available for combined plot.")
        plt.figure(figsize=(14,7))
        plot_points = min(200, len(y_true_for_combined))
        plt.plot(range(plot_points), y_true_for_combined[:plot_points], label='Actual', linewidth=2)
        for name in preds_for_combined:
            preds = preds_for_combined[name][:plot_points]
            plt.plot(range(plot_points), preds, linestyle='--', linewidth=1.8, label=f'{name} Pred')
        plt.title(f'Combined Actual vs Predicted (first {plot_points} points)')
        plt.xlabel('Time Steps')
        plt.ylabel('Temperature (°C)')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(COMBINED_PLOT, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"Saved combined predictions plot: {COMBINED_PLOT}")
    except Exception as e:
        print("Failed to create combined plot:", e)
        traceback.print_exc()

    # Write comparison results file
    try:
        with open(COMPARISON_RESULTS, 'w') as f:
            f.write("Model\tMAE(°C)\tRMSE(°C)\tMAPE(%)\n")
            for name, (mae, rmse, mape) in results.items():
                f.write(f"{name}\t{mae:.4f}\t{rmse:.4f}\t{mape:.2f}\n")
        print(f"Saved comparison results: {COMPARISON_RESULTS}")
    except Exception as e:
        print("Failed to save comparison results:", e)

    # Print final comparison
    print("\n=== MODEL COMPARISON ===")
    print("Model\tMAE(°C)\tRMSE(°C)\tMAPE(%)")
    for name, (mae, rmse, mape) in results.items():
        print(f"{name}\t{mae:.4f}\t{rmse:.4f}\t{mape:.2f}")
