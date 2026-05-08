#!/usr/bin/env python3
"""
install_fp16_headers.py
Copies/renames FP16 headers from c_models/ -> <sketch>/c_export/ with expected names.

Run from the parent directory that contains:
 - c_models/  (exporter output)
 - 3m_temp_predict/  (your Arduino sketch folder)
"""

import shutil, os, sys
from pathlib import Path

C_MODELS = Path("c_models")
SKETCH = Path("3m_temp_predict")
C_EXPORT = SKETCH / "c_export"

if not C_MODELS.exists():
    print("c_models/ not found in current folder.")
    sys.exit(1)

if not SKETCH.exists():
    print("Sketch folder '3m_temp_predict' not found in current folder.")
    sys.exit(1)

C_EXPORT.mkdir(parents=True, exist_ok=True)

# Map exporter filenames to the names we expect in the sketch
mapping = {
    "FNN": "fnn_model_data.h",
    "LSTM": "lstm_model_data.h",
    "TRN": "trn_model_data.h"
}

# If exporter used .h extension, accept that too
for src_basename, dest_name in mapping.items():
    candidates = [C_MODELS / src_basename, C_MODELS / (src_basename + ".h")]
    found = None
    for c in candidates:
        if c.exists():
            found = c
            break
    if not found:
        print(f"Warning: {src_basename} not found in c_models/. Skip.")
        continue
    dest = C_EXPORT / dest_name
    shutil.copy2(found, dest)
    print(f"Copied {found} -> {dest}")

print("Done. Check 3m_temp_predict/c_export/ for the three headers.")
