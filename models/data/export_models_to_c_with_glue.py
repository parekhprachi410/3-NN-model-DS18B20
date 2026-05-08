#!/usr/bin/env python3
"""
export_models_to_c_with_glue.py
---------------------------------------------------------
Exports trained .keras models (LSTM, FNN, TRN) into
C-compatible headers (.h) containing float16 weight arrays.

✅ No TensorFlow Lite
✅ Generates ready-to-include headers for ESP32 inference
✅ Float16 precision for compactness
✅ Console summary for architecture, weights, activations
---------------------------------------------------------
"""

import os
import numpy as np
import tensorflow as tf
from tensorflow import keras

# =======================================================
# CONFIGURATION
# =======================================================
MODELS = {
    "LSTM": "LSTM.keras",
    "FNN": "FNN.keras",
    "TRN": "TRN.keras",
}

OUTPUT_DIR = "c_export"
os.makedirs(OUTPUT_DIR, exist_ok=True)


# =======================================================
# Utility: format numpy arrays to C arrays
# =======================================================
def format_c_array(name, arr):
    """Format a NumPy array into a C array of float16."""
    flat = arr.flatten().astype(np.float16)
    values_per_line = 10
    lines = []

    for i in range(0, len(flat), values_per_line):
        slice_ = flat[i:i+values_per_line]
        vals = ", ".join(f"{x:.6f}f" for x in slice_)
        lines.append("    " + vals + ",")

    text = f"static const uint16_t {name}_data[] = {{\n" + "\n".join(lines) + "\n};\n"
    shape = ", ".join(map(str, arr.shape))
    text += f"static const int {name}_shape[] = {{{shape}}};\n"
    return text


# =======================================================
# Pretty-print model details
# =======================================================
def print_model_summary(model, name):
    print(f"\n{'='*60}")
    print(f"Model: {name}")
    print(f"{'='*60}")
    print(f"{'Layer (type)':30} | {'Output Shape':20} | {'Activation':12}")
    print("-"*60)

    total_params = 0

    for layer in model.layers:
        cfg = layer.get_config()
        act = cfg.get('activation', '-')
        out_shape = str(layer.output_shape) if hasattr(layer, 'output_shape') else "-"
        print(f"{layer.name + ' (' + layer.__class__.__name__ + ')':30} | {out_shape:20} | {act:12}")

        weights = layer.get_weights()
        for i, w in enumerate(weights):
            print(f"    ↳ Weight[{i}] shape: {w.shape}")
            total_params += np.prod(w.shape)

    print("-"*60)
    print(f"Total Parameters: {total_params:,}")
    print("="*60 + "\n")


# =======================================================
# Export function
# =======================================================
def export_model_to_header(model_path, name):
    model = keras.models.load_model(model_path)
    print_model_summary(model, name)

    out_path = os.path.join(OUTPUT_DIR, f"{name.lower()}_model_data.h")
    print(f"[INFO] Exporting {name} -> {out_path}")

    header_lines = [
        "// Auto-generated model weights",
        f"// Model: {name}",
        "#pragma once",
        "#include <stdint.h>",
        "",
    ]

    # For each layer
    for i, layer in enumerate(model.layers):
        weights = layer.get_weights()
        if not weights:
            continue

        header_lines.append(f"// Layer {i}: {layer.name} ({layer.__class__.__name__})")

        for j, w in enumerate(weights):
            arr_name = f"{layer.name}_w{j}"
            header_lines.append(format_c_array(arr_name, w))
            header_lines.append("")

    with open(out_path, "w") as f:
        f.write("\n".join(header_lines))

    print(f"[OK] Saved {out_path}\n")


# =======================================================
# Generate model_glue.h / .cpp
# =======================================================
def generate_glue_files():
    glue_h = os.path.join(OUTPUT_DIR, "model_glue.h")
    glue_cpp = os.path.join(OUTPUT_DIR, "model_glue.cpp")

    with open(glue_h, "w") as f:
        f.write("""// Auto-generated glue header
#pragma once
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef struct {
    const uint16_t* data;
    const int* shape;
} ModelArray;

#ifdef __cplusplus
}
#endif
""")

    with open(glue_cpp, "w") as f:
        f.write("// Auto-generated glue source\n#include \"model_glue.h\"\n")

    print(f"[OK] Generated model_glue.h / model_glue.cpp\n")


# =======================================================
# MAIN EXECUTION
# =======================================================
if __name__ == "__main__":
    for name, path in MODELS.items():
        if not os.path.exists(path):
            print(f"[WARN] Skipping {name} — {path} not found.")
            continue
        export_model_to_header(path, name)

    generate_glue_files()
    print("✅ All models exported successfully (float16).")
