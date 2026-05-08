#!/usr/bin/env python3
"""
auto_adapt_inference_names_v2.py

More robust adaptor: reads c_export/*.h, finds kernel/recurrent/bias names and their _len/_shape metadata,
maps them to expected template placeholders and writes adapted inference .cpp files.

Place next to the sketch folder (3m_temp_predict/) and run:
    python auto_adapt_inference_names_v2.py
"""
import re, shutil, os
from pathlib import Path
SKETCH = Path("3m_temp_predict")
C_EXPORT = SKETCH / "c_export"
BACKUP = SKETCH / "backups_inference_v2"
BACKUP.mkdir(parents=True, exist_ok=True)

if not C_EXPORT.exists():
    print("Error: c_export/ not found under sketch folder 3m_temp_predict/")
    raise SystemExit(1)

# read header text
def read_header(p):
    if not p.exists(): return ""
    return p.read_text(encoding='utf-8', errors='ignore')

files = {
    "fnn": C_EXPORT / "fnn_model_data.h",
    "lstm": C_EXPORT / "lstm_model_data.h",
    "trn": C_EXPORT / "trn_model_data.h"
}
texts = {k: read_header(v) for k,v in files.items()}

# regex patterns for common declarations
re_uint16_arr = re.compile(r'\b(?:const\s+(?:unsigned\s+short|uint16_t))\s+([A-Za-z0-9_]+)\s*\[\s*\]\s*=')
re_uint16_arr_alt = re.compile(r'\b(?:const\s+(?:unsigned\s+short|uint16_t))\s+([A-Za-z0-9_]+)\s*\[\s*\d*\s*\]\s*=')
re_int_shape = re.compile(r'\b(?:const\s+(?:int|unsigned\s+int))\s+([A-Za-z0-9_]+_shape)\s*\[\s*\]\s*=')
re_len = re.compile(r'\b(?:const\s+(?:unsigned\s+int|int))\s+([A-Za-z0-9_]+_len)\s*=')

def find_symbols(text):
    names = set(re_uint16_arr.findall(text) + re_uint16_arr_alt.findall(text))
    shapes = set(re_int_shape.findall(text))
    lens = set(re_len.findall(text))
    return {"names": sorted(names), "shapes": sorted(shapes), "lens": sorted(lens), "text": text}

fnn_syms = find_symbols(texts["fnn"])
lstm_syms = find_symbols(texts["lstm"])
trn_syms  = find_symbols(texts["trn"])

print("Detected symbols (counts):")
print(" FNN:", len(fnn_syms["names"]), "names,", len(fnn_syms["lens"]), "lens,", len(fnn_syms["shapes"]), "shapes")
print(" LSTM:", len(lstm_syms["names"]), "names,", len(lstm_syms["lens"]), "lens,", len(lstm_syms["shapes"]), "shapes")
print(" TRN:", len(trn_syms["names"]), "names,", len(trn_syms["lens"]), "lens,", len(trn_syms["shapes"]), "shapes")

# helper to find a candidate that contains tokens
def find_candidate(candidates, tokens):
    for c in candidates:
        ok = True
        for t in tokens:
            if t and t not in c:
                ok = False; break
        if ok:
            return c
    return None

# mapping heuristics
def build_dense_map(names, lens, prefix_guess):
    # expects dense layers like dense_3, dense_4, dense_5, dense_6
    mapping = {}
    for i in range(3,7):
        layer = f"dense_{i}"
        # kernel name likely contains layer and 'kernel' or 'weight'
        kern = find_candidate(names, [layer, "kernel"]) or find_candidate(names, [layer, "weight"]) or find_candidate(names, [layer])
        bias = find_candidate(names, [layer, "bias"]) or find_candidate(names, [layer, "bias"]) or None
        # lengths
        kern_len = find_candidate(lens, [layer, "kernel_len"]) or find_candidate(lens, [layer, "len"]) or None
        bias_len = find_candidate(lens, [layer, "bias_len"]) or find_candidate(lens, [layer, "len"]) or None
        mapping[f"{layer}_w0_data"] = kern
        mapping[f"{layer}_w0_shape"] = kern_len or "/*NO_SHAPE*/"
        mapping[f"{layer}_w1_data"] = bias
        mapping[f"{layer}_w1_shape"] = bias_len or "/*NO_SHAPE*/"
    return mapping

# Try to construct maps for each model using better heuristics
def construct_mapping_for(model_syms, model_name):
    names = model_syms["names"]
    lens = model_syms["lens"]
    shapes = model_syms["shapes"]
    map_out = {}

    # dense layers (generic)
    for token in ["dense", "dense_1", "dense_2", "dense_3", "dense_4", "dense_5", "dense_6", "dense_7", "dense_8"]:
        # find kernel and bias
        kern = find_candidate(names, [token, "kernel"]) or find_candidate(names, [token, "weight"]) or find_candidate(names, [token])
        bias = find_candidate(names, [token, "bias"])
        if kern:
            map_out[f"{token}_kernel"] = kern
        if bias:
            map_out[f"{token}_bias"] = bias
    # specific LSTM mapping
    for layer in ["lstm", "lstm_1", "lstm_2"]:
        kern = find_candidate(names, [layer, "kernel"]) or find_candidate(names, [layer, "kernel_data"]) or find_candidate(names, [layer])
        rec  = find_candidate(names, [layer, "recurrent"]) or find_candidate(names, [layer, "recurrent_kernel"]) 
        bias = find_candidate(names, [layer, "bias"])
        if kern: map_out[f"{layer}_kernel"] = kern
        if rec: map_out[f"{layer}_recurrent"] = rec
        if bias: map_out[f"{layer}_bias"] = bias

    # conv mapping: conv, conv1d, conv1d_1 etc.
    for token in ["conv1d", "conv1d_1", "conv", "conv_1"]:
        kern = find_candidate(names, [token, "kernel"]) or find_candidate(names, [token])
        bias = find_candidate(names, [token, "bias"])
        if kern: map_out[f"{token}_kernel"] = kern
        if bias: map_out[f"{token}_bias"] = bias

    # also capture any *_len tokens into map for quick use
    for l in lens:
        map_out[l] = l

    # capture shapes if present
    for s in shapes:
        map_out[s] = s

    return map_out

fnn_map = construct_mapping_for(fnn_syms, "fnn")
lstm_map = construct_mapping_for(lstm_syms, "lstm")
trn_map  = construct_mapping_for(trn_syms, "trn")

print("\nSample maps (first 20 keys):")
print(" FNN map keys:", list(fnn_map.keys())[:20])
print(" LSTM map keys:", list(lstm_map.keys())[:20])
print(" TRN map keys: ", list(trn_map.keys())[:20])

# Templates: simpler templates but we will replace using the discovered actual names.
TEMPLATE_FNN = r'''
// fnn_inference.cpp (auto-adapted v2)
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
    dense_forward_from_table_fp16(scaled_in, {d3_k}, NULL, {d3_b}, NULL, buf1);
    for (int i = 0; i < 128; ++i) buf1[i] = relu_f(buf1[i]);

    static float buf2[64];
    dense_forward_from_table_fp16(buf1, {d4_k}, NULL, {d4_b}, NULL, buf2);
    for (int i = 0; i < 64; ++i) buf2[i] = relu_f(buf2[i]);

    static float buf3[32];
    dense_forward_from_table_fp16(buf2, {d5_k}, NULL, {d5_b}, NULL, buf3);
    for (int i = 0; i < 32; ++i) buf3[i] = relu_f(buf3[i]);

    float out_final[1];
    dense_forward_from_table_fp16(buf3, {d6_k}, NULL, {d6_b}, NULL, out_final);

    float scaled_pred = out_final[0];
    float value = (scaled_pred - MIN_PARAM) / SCALE_PARAM;
    float temp_c = value * DATA_RANGE + DATA_MIN;
    return temp_c;
}}
'''
TEMPLATE_TRN = r'''
// trn_inference.cpp (auto-adapted v2)
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
    static float conv1_out[SEQ_LEN * 32 + 8];
    conv1d_forward_from_table_fp16(scaled_seq, SEQ_LEN, 1,
                                  {c1_k}, NULL,
                                  {c1_b}, NULL,
                                  conv1_out, out_len1);

    int out_len2 = 0;
    static float conv2_out[SEQ_LEN * 64 + 16];
    conv1d_forward_from_table_fp16(conv1_out, out_len1, 32,
                                  {c2_k}, NULL,
                                  {c2_b}, NULL,
                                  conv2_out, out_len2);

    static float dense7_out[64];
    dense_forward_from_table_fp16(conv2_out, {d7_k}, NULL, {d7_b}, NULL, dense7_out);
    for (int i = 0; i < 64; ++i) dense7_out[i] = relu_f(dense7_out[i]);

    float out_final[1];
    dense_forward_from_table_fp16(dense7_out, {d8_k}, NULL, {d8_b}, NULL, out_final);

    float scaled_pred = out_final[0];
    float value = (scaled_pred - MIN_PARAM) / SCALE_PARAM;
    float temp_c = value * DATA_RANGE + DATA_MIN;
    return temp_c;
}}
'''
TEMPLATE_LSTM = r'''
// lstm_inference.cpp (auto-adapted v2)
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
    int out_dim = 16;
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

    float *lstm1_out_last = (float*)malloc(sizeof(float) * 64);
    if (!lstm1_out_last) return DATA_MIN;
    lstm_forward_from_table_fp16(scaled_in, SEQ_LEN, 1,
                                {l_k}, NULL,
                                {l_r}, NULL,
                                {l_b}, NULL,
                                64, lstm1_out_last);

    float *lstm2_out_last = (float*)malloc(sizeof(float) * 32);
    if (!lstm2_out_last) {{ free(lstm1_out_last); return DATA_MIN; }}

    lstm_forward_from_table_fp16(lstm1_out_last, 1, 64,
                                {l1_k}, NULL,
                                {l1_r}, NULL,
                                {l1_b}, NULL,
                                32, lstm2_out_last);

    float dense_out1[16];
    dense_act_fp16(lstm2_out_last, {d_k}, NULL, {d_b}, NULL, dense_out1, "relu");

    float dense_out2[8];
    dense_act_fp16(dense_out1, {d1_k}, NULL, {d1_b}, NULL, dense_out2, "relu");

    float dense_out3[1];
    dense_act_fp16(dense_out2, {d2_k}, NULL, {d2_b}, NULL, dense_out3, "linear");

    float scaled_prediction = dense_out3[0];

    free(lstm1_out_last);
    free(lstm2_out_last);

    float value = (scaled_prediction - MIN_PARAM) / SCALE_PARAM;
    float temp_c = value * DATA_RANGE + DATA_MIN;
    return temp_c;
}}
'''

# Fill placeholders with discovered names (best-effort)
def get_or(name, dmap):
    return dmap.get(name, "/*"+name+"*/")

fnn_fill = {
    "d3_k": get_or("dense_3_kernel", fnn_map),
    "d3_b": get_or("dense_3_bias", fnn_map),
    "d4_k": get_or("dense_4_kernel", fnn_map),
    "d4_b": get_or("dense_4_bias", fnn_map),
    "d5_k": get_or("dense_5_kernel", fnn_map),
    "d5_b": get_or("dense_5_bias", fnn_map),
    "d6_k": get_or("dense_6_kernel", fnn_map),
    "d6_b": get_or("dense_6_bias", fnn_map),
}

trn_fill = {
    "c1_k": get_or("conv1d_kernel", trn_map),
    "c1_b": get_or("conv1d_bias", trn_map),
    "c2_k": get_or("conv1d_1_kernel", trn_map),
    "c2_b": get_or("conv1d_1_bias", trn_map),
    "d7_k": get_or("dense_7_kernel", trn_map),
    "d7_b": get_or("dense_7_bias", trn_map),
    "d8_k": get_or("dense_8_kernel", trn_map),
    "d8_b": get_or("dense_8_bias", trn_map),
}

lstm_fill = {
    "l_k": get_or("lstm_kernel", lstm_map) or get_or("lstm_lstm_kernel", lstm_map) or get_or("LSTM_lstm_kernel_data", lstm_map),
    "l_r": get_or("lstm_recurrent", lstm_map) or get_or("LSTM_lstm_recurrent_data", lstm_map),
    "l_b": get_or("lstm_bias", lstm_map) or get_or("LSTM_lstm_bias_data", lstm_map),
    "l1_k": get_or("lstm_1_kernel", lstm_map) or get_or("LSTM_lstm_1_kernel_data", lstm_map),
    "l1_r": get_or("lstm_1_recurrent", lstm_map) or get_or("LSTM_lstm_1_recurrent_data", lstm_map),
    "l1_b": get_or("lstm_1_bias", lstm_map) or get_or("LSTM_lstm_1_bias_data", lstm_map),
    "d_k": get_or("dense_kernel", lstm_map) or get_or("LSTM_dense_kernel_data", lstm_map),
    "d_b": get_or("dense_bias", lstm_map) or get_or("LSTM_dense_bias_data", lstm_map),
    "d1_k": get_or("dense_1_kernel", lstm_map) or get_or("LSTM_dense_1_kernel_data", lstm_map),
    "d1_b": get_or("dense_1_bias", lstm_map) or get_or("LSTM_dense_1_bias_data", lstm_map),
    "d2_k": get_or("dense_2_kernel", lstm_map) or get_or("LSTM_dense_2_kernel_data", lstm_map),
    "d2_b": get_or("dense_2_bias", lstm_map) or get_or("LSTM_dense_2_bias_data", lstm_map),
}

# Render and write files
def write_with_backup(name, content):
    path = SKETCH / name
    if path.exists():
        shutil.copy2(path, BACKUP / (name + ".bak"))
    path.write_text(content, encoding='utf-8')
    print("Wrote", path)

fnn_code = TEMPLATE_FNN.format(**fnn_fill)
trn_code = TEMPLATE_TRN.format(**trn_fill)
lstm_code= TEMPLATE_LSTM.format(**lstm_fill)

write_with_backup("fnn_inference.cpp", fnn_code)
write_with_backup("trn_inference.cpp", trn_code)
write_with_backup("lstm_inference.cpp", lstm_code)

print("\nWrote adapted files (v2). If compilation still errors, copy the FIRST compiler error line and paste it here.")
