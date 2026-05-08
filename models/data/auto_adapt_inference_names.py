#!/usr/bin/env python3
"""
auto_adapt_inference_names.py

Scans c_export/*.h (FP16 headers) to detect variable names and generates
adapted fnn_inference.cpp, lstm_inference.cpp, trn_inference.cpp
that use the actual names exported by the model exporter.

Usage:
    python auto_adapt_inference_names.py

It will:
 - backup existing *.cpp to backups/
 - write adapted cpp files into 3m_temp_predict/
"""

import re, shutil, os
from pathlib import Path

SKETCH = Path("3m_temp_predict")
C_EXPORT = SKETCH / "c_export"
BACKUP = SKETCH / "backups_inference"
BACKUP.mkdir(parents=True, exist_ok=True)

if not C_EXPORT.exists():
    print("Error: c_export/ not found under sketch folder 3m_temp_predict/")
    raise SystemExit(1)

# Files to inspect
headers = {
    "fnn": C_EXPORT / "fnn_model_data.h",
    "lstm": C_EXPORT / "lstm_model_data.h",
    "trn": C_EXPORT / "trn_model_data.h"
}

for k,h in headers.items():
    if not h.exists():
        print(f"Warning: {h} not found. Ensure you copied FP16 headers into c_export/")
        # continue; we'll still try to parse what we have

# regex to find variable names
# looks for declarations like: const unsigned short dense_3_w0_data[] = { ...
re_data = re.compile(r'\b(?:const\s+unsigned\s+short|const\s+uint16_t)\s+([A-Za-z0-9_]+_w[0-9]_data)\s*\[')
re_shape = re.compile(r'\b(?:const\s+int|static\s+const\s+int)\s+([A-Za-z0-9_]+_w[0-9]_shape)\s*\[')
re_other = re.compile(r'\b(?:const\s+unsigned\s+short|const\s+uint16_t)\s+([A-Za-z0-9_]+_w[0-9]_1_data)\s*\[')  # fallback

# helper to extract all names from a header
def extract_names(path):
    if not path.exists():
        return {}
    text = path.read_text(encoding='utf-8', errors='ignore')
    data_matches = re_data.findall(text)
    shape_matches = re_shape.findall(text)
    names = {"data": data_matches, "shape": shape_matches, "raw": text}
    return names

fnn_names = extract_names(headers["fnn"])
lstm_names = extract_names(headers["lstm"])
trn_names  = extract_names(headers["trn"])

print("Found names (samples):")
print(" FNN data vars:", fnn_names.get("data")[:8])
print(" LSTM data vars:", lstm_names.get("data")[:8])
print(" TRN  data vars:", trn_names.get("data")[:8])

# We'll create a simple mapping from *expected* names used in templates to actual found names.
# Expected template names we used earlier:
expected = {
    "fnn": [
        "dense_3_w0_data","dense_3_w0_shape","dense_3_w1_data","dense_3_w1_shape",
        "dense_4_w0_data","dense_4_w0_shape","dense_4_w1_data","dense_4_w1_shape",
        "dense_5_w0_data","dense_5_w0_shape","dense_5_w1_data","dense_5_w1_shape",
        "dense_6_w0_data","dense_6_w0_shape","dense_6_w1_data","dense_6_w1_shape"
    ],
    "trn": [
        "conv1d_w0_data","conv1d_w0_shape","conv1d_w1_data","conv1d_w1_shape",
        "conv1d_1_w0_data","conv1d_1_w0_shape","conv1d_1_w1_data","conv1d_1_w1_shape",
        "dense_7_w0_data","dense_7_w0_shape","dense_7_w1_data","dense_7_w1_shape",
        "dense_8_w0_data","dense_8_w0_shape","dense_8_w1_data","dense_8_w1_shape"
    ],
    "lstm": [
        "lstm_w0_data","lstm_w0_shape","lstm_w1_data","lstm_w1_shape","lstm_w2_data","lstm_w2_shape",
        "lstm_1_w0_data","lstm_1_w0_shape","lstm_1_w1_data","lstm_1_w1_shape","lstm_1_w2_data","lstm_1_w2_shape",
        "dense_w0_data","dense_w0_shape","dense_w1_data","dense_w1_shape",
        "dense_1_w0_data","dense_1_w0_shape","dense_1_w1_data","dense_1_w1_shape",
        "dense_2_w0_data","dense_2_w0_shape","dense_2_w1_data","dense_2_w1_shape"
    ]
}

# Find best matches: For each expected name, look for an actual name that contains the same layer number or keywords.
def match_expected(expected_list, extracted):
    actual = extracted.get("data", []) + extracted.get("shape", [])
    mapping = {}
    for exp in expected_list:
        base = exp.rsplit('_',2)[0]  # e.g. dense_3
        # find actual that startswith base or contains base
        found = None
        for cand in actual:
            if cand.startswith(base) or base in cand:
                found = cand
                break
        if found is None:
            # try any candidate with same suffix (w0_data -> endswith _w0_data)
            suffix = exp.split('_')[-2] + "_" + exp.split('_')[-1]
            for cand in actual:
                if cand.endswith(suffix):
                    found = cand
                    break
        if found is None and actual:
            found = actual[0]  # fallback to first (will likely break but we'll detect later)
        mapping[exp] = found
    return mapping

fnn_map = match_expected(expected["fnn"], fnn_names)
trn_map = match_expected(expected["trn"], trn_names)
lstm_map= match_expected(expected["lstm"], lstm_names)

# show mapping summary
print("\nMappings (sample):")
print(" FNN mapping:", {k:fnn_map[k] for k in list(fnn_map)[:6]})
print(" TRN mapping:", {k:trn_map[k] for k in list(trn_map)[:6]})
print(" LSTM mapping:", {k:lstm_map[k] for k in list(lstm_map)[:6]})

# templates (shortened versions of the FP16 .cpp earlier). We'll load full templates from strings below.
# For brevity in this script we store the templates as multi-line strings.
# NOTE: If you customized earlier .cpp, edit the templates here accordingly.

TEMPLATE_FNN = r'''
// fnn_inference.cpp (auto-adapted)
#include "fnn_inference.h"
#include "model_runtime_utils.h"
#include "dense_layer_fp16.h"
#include "c_export/fnn_model_data.h"

float predict_fnn(const float input_seq[SEQ_LEN]) {{
    float scaled_in[SEQ_LEN];
    for (int i = 0; i < SEQ_LEN; ++i) {{
        scaled_in[i] = MIN_PARAM + ((input_seq[i] - DATA_MIN) * (SCALE_PARAM / DATA_RANGE));
    }}

    static float buf1[128];
    dense_forward_from_table_fp16(scaled_in, {dense_3_w0_data}, {dense_3_w0_shape}, {dense_3_w1_data}, {dense_3_w1_shape}, buf1);
    for (int i = 0; i < 128; ++i) buf1[i] = relu_f(buf1[i]);

    static float buf2[64];
    dense_forward_from_table_fp16(buf1, {dense_4_w0_data}, {dense_4_w0_shape}, {dense_4_w1_data}, {dense_4_w1_shape}, buf2);
    for (int i = 0; i < 64; ++i) buf2[i] = relu_f(buf2[i]);

    static float buf3[32];
    dense_forward_from_table_fp16(buf2, {dense_5_w0_data}, {dense_5_w0_shape}, {dense_5_w1_data}, {dense_5_w1_shape}, buf3);
    for (int i = 0; i < 32; ++i) buf3[i] = relu_f(buf3[i]);

    float out_final[1];
    dense_forward_from_table_fp16(buf3, {dense_6_w0_data}, {dense_6_w0_shape}, {dense_6_w1_data}, {dense_6_w1_shape}, out_final);

    float scaled_pred = out_final[0];
    float value = (scaled_pred - MIN_PARAM) / SCALE_PARAM;
    float temp_c = value * DATA_RANGE + DATA_MIN;
    return temp_c;
}}
'''

TEMPLATE_TRN = r'''
// trn_inference.cpp (auto-adapted)
#include "trn_inference.h"
#include "model_runtime_utils.h"
#include "conv1d_layer_fp16.h"
#include "dense_layer_fp16.h"
#include "c_export/trn_model_data.h"

float predict_trn(const float input_seq[SEQ_LEN]) {{
    float scaled_seq[SEQ_LEN];
    for (int i = 0; i < SEQ_LEN; ++i) {{
        scaled_seq[i] = MIN_PARAM + ((input_seq[i] - DATA_MIN) * (SCALE_PARAM / DATA_RANGE));
    }}

    int out_len1 = 0;
    int in_ch1 = {conv1d_w0_shape}[1];
    static float conv1_out[SEQ_LEN * 32 + 8];
    conv1d_forward_from_table_fp16(scaled_seq, SEQ_LEN, in_ch1,
                                  {conv1d_w0_data}, {conv1d_w0_shape},
                                  {conv1d_w1_data}, {conv1d_w1_shape},
                                  conv1_out, out_len1);

    int out_len2 = 0;
    int in_ch2 = {conv1d_1_w0_shape}[1];
    static float conv2_out[SEQ_LEN * 64 + 16];
    conv1d_forward_from_table_fp16(conv1_out, out_len1, in_ch2,
                                  {conv1d_1_w0_data}, {conv1d_1_w0_shape},
                                  {conv1d_1_w1_data}, {conv1d_1_w1_shape},
                                  conv2_out, out_len2);

    static float dense7_out[64];
    dense_forward_from_table_fp16(conv2_out, {dense_7_w0_data}, {dense_7_w0_shape}, {dense_7_w1_data}, {dense_7_w1_shape}, dense7_out);
    for (int i = 0; i < 64; ++i) dense7_out[i] = relu_f(dense7_out[i]);

    float out_final[1];
    dense_forward_from_table_fp16(dense7_out, {dense_8_w0_data}, {dense_8_w0_shape}, {dense_8_w1_data}, {dense_8_w1_shape}, out_final);

    float scaled_pred = out_final[0];
    float value = (scaled_pred - MIN_PARAM) / SCALE_PARAM;
    float temp_c = value * DATA_RANGE + DATA_MIN;
    return temp_c;
}}
'''

TEMPLATE_LSTM = r'''
// lstm_inference.cpp (auto-adapted)
#include "lstm_inference.h"
#include "model_runtime_utils.h"
#include "lstm_layer_fp16.h"
#include "dense_layer_fp16.h"
#include "c_export/lstm_model_data.h"
#include <string.h>

static void dense_act_fp16(const float *in_vec, const uint16_t *kern, const int *kern_shape,
                           const uint16_t *bias, const int *bias_shape,
                           float *out_vec, const char *activation)
{{
    dense_forward_from_table_fp16(in_vec, kern, kern_shape, bias, bias_shape, out_vec);
    int out_dim = kern_shape[1];
    if (!activation) return;
    if (strcmp(activation, "relu") == 0) {{
        for (int i = 0; i < out_dim; ++i) out_vec[i] = relu_f(out_vec[i]);
    }} else if (strcmp(activation, "tanh") == 0) {{
        for (int i = 0; i < out_dim; ++i) out_vec[i] = tanh_f(out_vec[i]);
    }} else if (strcmp(activation, "sigmoid") == 0) {{
        for (int i = 0; i < out_dim; ++i) out_vec[i] = sigmoid_f(out_vec[i]);
    }}
}}

float predict_lstm(const float input_seq[SEQ_LEN]) {{
    float scaled_in[SEQ_LEN];
    for (int i = 0; i < SEQ_LEN; ++i)
        scaled_in[i] = MIN_PARAM + ((input_seq[i] - DATA_MIN) * (SCALE_PARAM / DATA_RANGE));

    int k0_out4 = {lstm_w0_shape}[1];
    int units0 = k0_out4 / 4;
    float *lstm1_out_last = (float*)malloc(sizeof(float) * units0);
    if (!lstm1_out_last) return DATA_MIN;
    lstm_forward_from_table_fp16(scaled_in, SEQ_LEN, 1,
                                {lstm_w0_data}, {lstm_w0_shape},
                                {lstm_w1_data}, {lstm_w1_shape},
                                {lstm_w2_data}, {lstm_w2_shape},
                                units0, lstm1_out_last);

    int units1 = {lstm_1_w0_shape}[1] / 4;
    float *lstm2_out_last = (float*)malloc(sizeof(float) * units1);
    if (!lstm2_out_last) {{ free(lstm1_out_last); return DATA_MIN; }}

    lstm_forward_from_table_fp16(lstm1_out_last, 1, units0,
                                {lstm_1_w0_data}, {lstm_1_w0_shape},
                                {lstm_1_w1_data}, {lstm_1_w1_shape},
                                {lstm_1_w2_data}, {lstm_1_w2_shape},
                                units1, lstm2_out_last);

    float dense_out1[16];
    dense_act_fp16(lstm2_out_last, {dense_w0_data}, {dense_w0_shape}, {dense_w1_data}, {dense_w1_shape}, dense_out1, "relu");

    float dense_out2[8];
    dense_act_fp16(dense_out1, {dense_1_w0_data}, {dense_1_w0_shape}, {dense_1_w1_data}, {dense_1_w1_shape}, dense_out2, "relu");

    float dense_out3[1];
    dense_act_fp16(dense_out2, {dense_2_w0_data}, {dense_2_w0_shape}, {dense_2_w1_data}, {dense_2_w1_shape}, dense_out3, "linear");

    float scaled_prediction = dense_out3[0];

    free(lstm1_out_last);
    free(lstm2_out_last);

    float value = (scaled_prediction - MIN_PARAM) / SCALE_PARAM;
    float temp_c = value * DATA_RANGE + DATA_MIN;
    return temp_c;
}}
'''

# Build mapping dict for all expected -> actual
full_map = {}
full_map.update({k: v for k, v in fnn_map.items()})
full_map.update({k: v for k, v in trn_map.items()})
full_map.update({k: v for k, v in lstm_map.items()})

# Simple function to substitute template placeholders
def render_template(tmpl, mapping):
    out = tmpl
    for k,v in mapping.items():
        out = out.replace("{" + k + "}", v if v else "/*MISSING_"+k+"*/")
    return out

# Back up existing cpp files
for name in ["fnn_inference.cpp","trn_inference.cpp","lstm_inference.cpp"]:
    fpath = SKETCH / name
    if fpath.exists():
        shutil.copy2(fpath, BACKUP / (name + ".bak"))
        print("Backed up", fpath, "->", BACKUP / (name + ".bak"))

# Render templates with mapping
fnn_src = render_template(TEMPLATE_FNN, full_map)
trn_src = render_template(TEMPLATE_TRN, full_map)
lstm_src= render_template(TEMPLATE_LSTM, full_map)

# Write out
(SKETCH / "fnn_inference.cpp").write_text(fnn_src, encoding='utf-8')
(SKETCH / "trn_inference.cpp").write_text(trn_src, encoding='utf-8')
(SKETCH / "lstm_inference.cpp").write_text(lstm_src, encoding='utf-8')

print("Wrote adapted inference cpp files to", SKETCH)
print("Please re-open Arduino IDE and Verify. If there's an error, copy the first error and paste here.")
