#!/usr/bin/env python3
"""
fix_inference_names.py

Place this in your Arduino sketch/project folder (where your *.cpp inference files are),
and ensure the folder 'c_export' (or the folder containing exported headers) is present.

It will:
 - scan c_export/*.h for exported symbol names (arrays)
 - try to map those to expected logical names used by inference cpp files
 - replace placeholders like /*MISSING_dense_3_w0_data*/ in your inference .cpp files
   with the detected actual symbol names
 - write a mapping JSON and backup originals.

If mapping fails for some names, it will leave the placeholder and print warnings.
"""

import re, os, json, shutil, sys
from pathlib import Path

# --- CONFIG: adjust if your headers live somewhere else ---
EXPORT_DIRS = ["c_export", "c_models", "c_models_fp16", "c_models"]  # tried order
INFERENCE_FILES = ["fnn_inference.cpp", "lstm_inference.cpp", "trn_inference.cpp"]
BACKUP_DIR = "backups_inference_namefix"

# Logical names we expect to replace in the inference .cpp (the names your inference uses)
# Add or remove expected keys if your inference files use slightly different names.
EXPECTED_KEYS = [
    # FNN dense chain
    "dense_3_w0_data", "dense_3_w0_shape", "dense_3_w1_data", "dense_3_w1_shape",
    "dense_4_w0_data", "dense_4_w0_shape", "dense_4_w1_data", "dense_4_w1_shape",
    "dense_5_w0_data", "dense_5_w0_shape", "dense_5_w1_data", "dense_5_w1_shape",
    "dense_6_w0_data", "dense_6_w0_shape", "dense_6_w1_data", "dense_6_w1_shape",
    # TRN conv/dense
    "conv1d_w0_data", "conv1d_w0_shape", "conv1d_w1_data", "conv1d_w1_shape",
    "conv1d_1_w0_data", "conv1d_1_w0_shape", "conv1d_1_w1_data", "conv1d_1_w1_shape",
    "dense_7_w0_data", "dense_7_w0_shape", "dense_7_w1_data", "dense_7_w1_shape",
    "dense_8_w0_data", "dense_8_w0_shape", "dense_8_w1_data", "dense_8_w1_shape",
    # LSTM kernels/recurrent/bias names (generic)
    "lstm_w0_data", "lstm_w0_shape", "lstm_w1_data", "lstm_w1_shape", "lstm_w2_data", "lstm_w2_shape",
    "lstm_1_w0_data", "lstm_1_w0_shape", "lstm_1_w1_data", "lstm_1_w1_shape", "lstm_1_w2_data", "lstm_1_w2_shape",
    # Dense heads for LSTM
    "dense_w0_data", "dense_w0_shape", "dense_w1_data", "dense_w1_shape",
    "dense_1_w0_data", "dense_1_w0_shape", "dense_1_w1_data", "dense_1_w1_shape",
    "dense_2_w0_data", "dense_2_w0_shape", "dense_2_w1_data", "dense_2_w1_shape"
]

# heuristics: keywords to match expected logical names to exported symbols
HEURISTICS = {
    "lstm_w0_data": ["lstm_kernel", "_lstm_kernel", "lstm_kernel_data"],
    "lstm_w1_data": ["lstm_recurrent", "_lstm_recurrent", "lstm_recurrent_data"],
    "lstm_w2_data": ["lstm_bias", "_lstm_bias", "lstm_bias_data"],
    "lstm_1_w0_data": ["lstm_1_kernel", "lstm_1_kernel_data"],
    "lstm_1_w1_data": ["lstm_1_recurrent", "lstm_1_recurrent_data"],
    "lstm_1_w2_data": ["lstm_1_bias", "lstm_1_bias_data"],
    # dense layers: try to match dense_<n> or group by model prefix (FNN_, TRN_, LSTM_)
    "dense_3_w0_data": ["dense_3_kernel", "dense_3_kernel_data", "dense3_kernel", "dense_3_w0"],
    "dense_3_w1_data": ["dense_3_bias", "dense_3_bias_data", "dense3_bias", "dense_3_w1"],
    "dense_4_w0_data": ["dense_4_kernel", "dense_4_kernel_data", "dense4_kernel"],
    "dense_4_w1_data": ["dense_4_bias", "dense_4_bias_data", "dense4_bias"],
    "dense_5_w0_data": ["dense_5_kernel", "dense_5_kernel_data", "dense5_kernel"],
    "dense_5_w1_data": ["dense_5_bias", "dense_5_bias_data", "dense5_bias"],
    "dense_6_w0_data": ["dense_6_kernel", "dense_6_kernel_data", "dense6_kernel"],
    "dense_6_w1_data": ["dense_6_bias", "dense_6_bias_data", "dense6_bias"],
    "dense_7_w0_data": ["dense_7_kernel", "dense_7_kernel_data", "dense7_kernel"],
    "dense_7_w1_data": ["dense_7_bias", "dense_7_bias_data", "dense7_bias"],
    "dense_8_w0_data": ["dense_8_kernel", "dense_8_kernel_data", "dense8_kernel"],
    "dense_8_w1_data": ["dense_8_bias", "dense_8_bias_data", "dense8_bias"],
    # conv layers
    "conv1d_w0_data": ["conv1d_kernel", "conv1d_kernel_data", "conv1d_w0", "conv1d0_kernel"],
    "conv1d_w1_data": ["conv1d_bias", "conv1d_bias_data", "conv1d_w1", "conv1d0_bias"],
    "conv1d_1_w0_data": ["conv1d_1_kernel", "conv1d_1_kernel_data", "conv1d1_kernel"],
    "conv1d_1_w1_data": ["conv1d_1_bias", "conv1d_1_bias_data", "conv1d1_bias"],
    # LSTM dense heads - generic matches as fallback
    "dense_w0_data": ["dense_kernel", "dense_kernel_data"],
    "dense_w1_data": ["dense_bias", "dense_bias_data"],
    "dense_1_w0_data": ["dense_1_kernel", "dense_1_kernel_data"],
    "dense_1_w1_data": ["dense_1_bias", "dense_1_bias_data"],
    "dense_2_w0_data": ["dense_2_kernel", "dense_2_kernel_data"],
    "dense_2_w1_data": ["dense_2_bias", "dense_2_bias_data"],
}

# helper: read all exported names from header files in export_dir
def collect_exported_symbols(export_dir):
    exported = set()
    shapes = set()
    path = Path(export_dir)
    if not path.exists():
        return exported, shapes
    for h in path.glob("*.h"):
        txt = h.read_text(encoding="utf-8", errors="ignore")
        # catch symbols like: const unsigned short LSTM_lstm_kernel_data[] = {
        for m in re.finditer(r'const\s+(?:unsigned\s+short|unsigned\s+int|unsigned\s+char|uint16_t|float|double)\s+(\w+)\s*(?:\[\])?', txt):
            exported.add(m.group(1))
        # also capture *_len and *_shape
        for m in re.finditer(r'const\s+unsigned\s+int\s+(\w+_len)\s*=', txt):
            exported.add(m.group(1))
        for m in re.finditer(r'const\s+int\s+(\w+_shape)\s*\[', txt):
            exported.add(m.group(1))
    return sorted(exported), sorted(shapes)

# heuristic mapper: try to find best match for each expected key
def map_expected_to_exported(expected_keys, exported_names):
    mapping = {}
    lowered = [ (n, n.lower()) for n in exported_names ]
    for key in expected_keys:
        mapped = None
        # 1) direct name present?
        if key in exported_names:
            mapped = key
        else:
            # 2) try heuristics list
            patt_list = HEURISTICS.get(key, [])
            for patt in patt_list:
                for name, lname in lowered:
                    if patt in lname:
                        mapped = name
                        break
                if mapped:
                    break
            # 3) fallback: try to find name that has the layer index (e.g., "dense_3" in it)
            if mapped is None:
                num_p = re.search(r'dense_(\d+)|conv1d_?(\d+)|lstm_?(\d+)', key)
                if num_p:
                    num = next(group for group in num_p.groups() if group)
                    for name, lname in lowered:
                        if f"_{num}_" in lname or f"_{num}" in lname or f"{num}" in lname:
                            # ensure 'dense' or 'conv1d' present
                            if ('dense' in key and 'dense' in lname) or ('conv' in key and 'conv' in lname) or ('lstm' in key and 'lstm' in lname):
                                mapped = name
                                break
        mapping[key] = mapped
    return mapping

# replace placeholders in files
def patch_inference_files(mapping):
    os.makedirs(BACKUP_DIR, exist_ok=True)
    changed_files = []
    for fname in INFERENCE_FILES:
        p = Path(fname)
        if not p.exists():
            print(f" WARN: inference file {fname} not found in cwd; skipping.")
            continue
        txt = p.read_text(encoding="utf-8", errors="ignore")
        # backup
        shutil.copy2(p, Path(BACKUP_DIR) / (fname + ".bak"))
        original = txt
        # replace placeholders like /*MISSING_dense_3_w0_data*/ or /*MISSING_dense_3_w0_data*/
        for k, v in mapping.items():
            placeholder = r'/\*MISSING_' + re.escape(k) + r'\*/'
            if v:
                txt_new = re.sub(placeholder, v, txt)
            else:
                # also try replacing `/*MISSING_name*/` without MISSING_ prefix (some variants)
                txt_new = txt
            if txt_new != txt:
                txt = txt_new
        if txt != original:
            p.write_text(txt, encoding="utf-8")
            changed_files.append(fname)
            print(f" Patched {fname} (backup -> {BACKUP_DIR}/{fname}.bak)")
        else:
            print(f" No placeholders patched in {fname} (placeholders may be absent or mapping missing).")
    return changed_files

def main():
    # find export dir
    export_dir = None
    for d in EXPORT_DIRS:
        if Path(d).exists():
            export_dir = d
            break
    if export_dir is None:
        print("ERROR: Could not find c_export folder. Make sure your exported headers are in one of:", EXPORT_DIRS)
        sys.exit(1)
    print("Using export dir:", export_dir)

    exported, shapes = collect_exported_symbols(export_dir)
    print("Found exported symbols (sample):")
    for s in exported[:80]:
        print(" ", s)
    mapping = map_expected_to_exported(EXPECTED_KEYS, exported)
    print("\nProposed mapping (partial):")
    for k in EXPECTED_KEYS:
        print(f" {k} -> {mapping.get(k)}")
    # write mapping json for review
    with open("name_mapping.json", "w", encoding="utf-8") as f:
        json.dump(mapping, f, indent=2)
    print("Wrote name_mapping.json")

    # patch inference files
    patched = patch_inference_files(mapping)
    if patched:
        print("\nPatched files:", patched)
    else:
        print("\nNo files were patched. If you still see /*MISSING_*/ placeholders in the .cpp files, open name_mapping.json and adjust heuristics or the EXPECTED_KEYS list in this script.")
    print("\nDone. Re-open Arduino IDE and verify. If compilation fails, copy the FIRST compiler error here and paste it.")

if __name__ == "__main__":
    main()
