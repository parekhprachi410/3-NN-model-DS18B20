// lstm_inference.cpp
// Final corrected lstm inference file.
// Exports: extern "C" float predict_lstm(const float *input_seq)
// Also contains an internal predict_lstm_with_len(...)

#include <Arduino.h>
#include <stdint.h>
#include <string.h>
#include <math.h>

#include "scaler.h"         // optional: if you provide invert_scale() or other scaler helpers
#include "c_models/LSTM.h"  // ensure this path/name matches your c_models export

// Tunables / safety caps
#ifndef LSTM_EXPECTED_SEQ_LEN
#define LSTM_EXPECTED_SEQ_LEN 60
#endif
#ifndef LSTM_MAX_UNITS
#define LSTM_MAX_UNITS 256
#endif
#ifndef LSTM_MAX_IN_DIM
#define LSTM_MAX_IN_DIM 4
#endif

#ifdef __cplusplus
extern "C" {
#endif

// --- FP16 -> float conversion helper (bit-manipulation)
static float fp16_to_float(uint16_t h) {
    uint32_t sign = (h & 0x8000u) << 16;
    uint32_t exp  = (h & 0x7C00u) >> 10;
    uint32_t mant = (h & 0x03FFu);

    uint32_t f;
    if (exp == 0x1F) {
        // Inf or NaN
        f = sign | 0x7F800000u | (mant << 13);
    } else if (exp == 0) {
        if (mant == 0) {
            // zero
            f = sign;
        } else {
            // subnormal
            // normalize
            while ((mant & 0x0400u) == 0) {
                mant <<= 1;
                exp--;
            }
            mant &= 0x03FFu;
            exp = (int)exp + 1;
            uint32_t e = ((int)exp + (127 - 15)) << 23;
            f = sign | e | (mant << 13);
        }
    } else {
        uint32_t e = ((int)exp + (127 - 15)) << 23;
        f = sign | e | (mant << 13);
    }

    union { uint32_t u; float f; } u;
    u.u = f;
    return u.f;
}

// --- small helpers used by the LSTM runtime ---
static inline float sigmoid_f(float x) {
    if (x < -20.0f) return 2.06e-9f;
    if (x > 20.0f) return 0.999999998f;
    return 1.0f / (1.0f + expf(-x));
}
static inline float tanh_f_local(float x) { return tanhf(x); }
static inline float relu_f_local(float x) { return x > 0.0f ? x : 0.0f; }

// dense fallback (in case project does not provide a dense helper)
static bool dense_forward_from_table_fp16_fallback(const float *in_vec, int in_dim,
                                                  const uint16_t *kern_fp16, unsigned int kern_len,
                                                  const uint16_t *bias_fp16, unsigned int bias_len,
                                                  float *out_vec)
{
    if (!in_vec || !kern_fp16 || !out_vec || in_dim <= 0) return false;
    if (kern_len % (unsigned int)in_dim != 0) return false;
    unsigned int out_dim = kern_len / (unsigned int)in_dim;

    for (unsigned int r = 0; r < out_dim; ++r) {
        double acc = 0.0;
        unsigned int base = r * in_dim;
        for (int c = 0; c < in_dim; ++c) {
            float w = fp16_to_float(kern_fp16[base + c]);
            acc += (double)w * (double)in_vec[c];
        }
        out_vec[r] = (float)acc;
    }
    if (bias_fp16 && bias_len >= out_dim) {
        for (unsigned int i = 0; i < out_dim; ++i) out_vec[i] += fp16_to_float(bias_fp16[i]);
    }
    return true;
}

// --- LSTM forward that accepts FP16 tables (kernels, recurrent, bias) ---
static bool lstm_forward_from_table_fp16(const float *in_seq, int seq_len, int in_ch,
                                         const uint16_t *k0, unsigned int k0_len,
                                         const uint16_t *k1, unsigned int k1_len,
                                         const uint16_t *k2, unsigned int k2_len,
                                         int units_capacity, float *out_hidden)
{
    // Preconditions
    if (!in_seq || seq_len <= 0 || in_ch <= 0 || !k0 || !k1 || !k2 || !out_hidden) return false;

    // infer fourU = 4*units from k0_len / in_ch
    if ((unsigned int)in_ch == 0) return false;
    if (k0_len % (unsigned int)in_ch != 0) return false;
    unsigned int fourU = k0_len / (unsigned int)in_ch;
    if (fourU % 4u != 0) return false;
    unsigned int units = fourU / 4u;
    if (units == 0 || units > (unsigned int)units_capacity) return false;

    // check recurrent shape: expected units * (4*units)
    if (k1_len != units * 4u * units) return false;
    // check bias shape
    if (k2_len < 4u * units) return false;

    // static working buffers (avoid stack)
    static float h_prev[LSTM_MAX_UNITS];
    static float c_prev[LSTM_MAX_UNITS];
    static float gates[LSTM_MAX_UNITS * 4u];

    // initialize
    for (unsigned int i = 0; i < units; ++i) { h_prev[i] = 0.0f; c_prev[i] = 0.0f; }

    // For each timestep
    for (int t = 0; t < seq_len; ++t) {
        const float *x_t = &in_seq[t * in_ch]; // assume input_seq is [t0,ch0, t1,ch0, ...] (for scalar ch==1 it's okay)

        // Compute W_x * x_t into gates[0 .. 4*units-1]
        // k0 layout assumed: for in=0..in_ch-1, for out=0..4*units-1, index = in * (4*units) + out
        for (unsigned int g = 0; g < 4u * units; ++g) {
            double acc = 0.0;
            for (int ic = 0; ic < in_ch; ++ic) {
                uint16_t hf = k0[(unsigned int)ic * (4u * units) + g];
                float w = fp16_to_float(hf);
                acc += (double)w * (double)x_t[ic];
            }
            gates[g] = (float)acc;
        }

        // Add recurrent contribution R * h_prev; R layout: row r has 4*units entries => index = r*(4*units) + col
        for (unsigned int g = 0; g < 4u * units; ++g) {
            double acc = 0.0;
            unsigned int col = g;
            for (unsigned int r = 0; r < units; ++r) {
                uint16_t hf = k1[r * (4u * units) + col];
                float wr = fp16_to_float(hf);
                acc += (double)wr * (double)h_prev[r];
            }
            gates[g] += (float)acc;
        }

        // add bias
        for (unsigned int g = 0; g < 4u * units; ++g) gates[g] += fp16_to_float(k2[g]);

        // compute gate outputs per unit (assuming order i, f, c, o)
        for (unsigned int u = 0; u < units; ++u) {
            float i_gate = sigmoid_f(gates[u + 0u * units]);
            float f_gate = sigmoid_f(gates[u + 1u * units]);
            float c_tilde = tanh_f_local(gates[u + 2u * units]);
            float o_gate = sigmoid_f(gates[u + 3u * units]);

            float c_new = f_gate * c_prev[u] + i_gate * c_tilde;
            float h_new = o_gate * tanh_f_local(c_new);

            c_prev[u] = c_new;
            h_prev[u] = h_new;
        }
    }

    // copy final hidden to out_hidden
    for (unsigned int i = 0; i < units; ++i) out_hidden[i] = h_prev[i];
    return true;
}

// internal function with explicit seq length
static float predict_lstm_with_len(const float *input_seq, unsigned int seq_len) {
    Serial.println("predict_lstm_with_len: enter");
    Serial.print("predict_lstm_with_len: freeHeap at entry = ");
    Serial.println(ESP.getFreeHeap());

    if (!input_seq || seq_len == 0) {
        Serial.println("predict_lstm_with_len: invalid input");
        return 0.0f;
    }

    // sanity: check presence of exported tables from c_models/LSTM.h
    bool ok_tables = true;
    if (LSTM_lstm_kernel_len == 0 || LSTM_lstm_recurrent_len == 0 || LSTM_lstm_bias_len == 0) ok_tables = false;
    if (LSTM_lstm_1_kernel_len == 0 || LSTM_lstm_1_recurrent_len == 0 || LSTM_lstm_1_bias_len == 0) ok_tables = false;
    if (LSTM_dense_kernel_len == 0 || LSTM_dense_bias_len == 0) ok_tables = false;

    if (!ok_tables) {
        Serial.println("predict_lstm_with_len: WARNING: missing essential LSTM tables in header!");
        // fallback: return last input value (safe default)
        return input_seq[(seq_len > 0) ? (seq_len - 1) : 0];
    }

    // Prepare scaled / clipped input: assume scalar time series (in_ch=1).
    int in_ch = 1;
    int use_len = (int)seq_len;
    int cap_len = (use_len > LSTM_EXPECTED_SEQ_LEN) ? LSTM_EXPECTED_SEQ_LEN : use_len;
    static float scaled_in[LSTM_EXPECTED_SEQ_LEN * LSTM_MAX_IN_DIM];

    // If you have a scaler.h with a function scale_input() / invert_scale() — adapt here.
    // For now copy raw values into buffer.
    for (int i = 0; i < cap_len; ++i) scaled_in[i] = input_seq[i];

    Serial.println("predict_lstm_with_len: -> calling LSTM layer 0");
    Serial.print("predict_lstm_with_len: freeHeap before lstm0: "); Serial.println(ESP.getFreeHeap());

    static float lstm1_out[LSTM_MAX_UNITS];
    static float lstm2_out[LSTM_MAX_UNITS];
    memset(lstm1_out, 0, sizeof(lstm1_out));
    memset(lstm2_out, 0, sizeof(lstm2_out));

    // run first LSTM: input sequence, in_ch=1
    bool ok = lstm_forward_from_table_fp16(
        scaled_in, cap_len, in_ch,
        LSTM_lstm_kernel_data, LSTM_lstm_kernel_len,
        LSTM_lstm_recurrent_data, LSTM_lstm_recurrent_len,
        LSTM_lstm_bias_data, LSTM_lstm_bias_len,
        LSTM_MAX_UNITS,
        lstm1_out
    );

    if (!ok) {
        Serial.println("predict_lstm_with_len: ERROR: first LSTM failed");
        return input_seq[cap_len - 1];
    }
    Serial.print("predict_lstm_with_len: freeHeap after lstm0: "); Serial.println(ESP.getFreeHeap());

    // run second LSTM: treat lstm1_out as a single-step sequence (seq_len=1)
    ok = lstm_forward_from_table_fp16(
        lstm1_out, 1, /*in_ch=*/ (int)LSTM_MAX_UNITS, // in_ch hint; function infers shapes
        LSTM_lstm_1_kernel_data, LSTM_lstm_1_kernel_len,
        LSTM_lstm_1_recurrent_data, LSTM_lstm_1_recurrent_len,
        LSTM_lstm_1_bias_data, LSTM_lstm_1_bias_len,
        LSTM_MAX_UNITS,
        lstm2_out
    );

    if (!ok) {
        Serial.println("predict_lstm_with_len: ERROR: second LSTM failed");
        return input_seq[cap_len - 1];
    }
    Serial.print("predict_lstm_with_len: freeHeap after lstm1: "); Serial.println(ESP.getFreeHeap());

    // Post-LSTM dense sequence: dense -> relu -> dense -> relu -> dense -> linear
    static float dense1_out[256];
    static float dense2_out[256];
    static float dense3_out[8];

    // dense 0: LSTM_dense_kernel, LSTM_dense_bias
    if (!dense_forward_from_table_fp16_fallback(lstm2_out, LSTM_MAX_UNITS,
                                               LSTM_dense_kernel_data, LSTM_dense_kernel_len,
                                               LSTM_dense_bias_data, LSTM_dense_bias_len,
                                               dense1_out)) {
        Serial.println("predict_lstm_with_len: ERROR dense0");
        return input_seq[cap_len - 1];
    }
    // apply relu to first dense outputs (limit by bias_len as out dim hint)
    for (unsigned int i = 0; i < LSTM_dense_bias_len && i < sizeof(dense1_out)/sizeof(dense1_out[0]); ++i)
        dense1_out[i] = relu_f_local(dense1_out[i]);

    // dense1
    if (!dense_forward_from_table_fp16_fallback(dense1_out, (int)LSTM_dense_bias_len,
                                               LSTM_dense_1_kernel_data, LSTM_dense_1_kernel_len,
                                               LSTM_dense_1_bias_data, LSTM_dense_1_bias_len,
                                               dense2_out)) {
        Serial.println("predict_lstm_with_len: ERROR dense1");
        return input_seq[cap_len - 1];
    }
    for (unsigned int i = 0; i < LSTM_dense_1_bias_len && i < sizeof(dense2_out)/sizeof(dense2_out[0]); ++i)
        dense2_out[i] = relu_f_local(dense2_out[i]);

    // final dense
    if (!dense_forward_from_table_fp16_fallback(dense2_out, (int)LSTM_dense_1_bias_len,
                                               LSTM_dense_2_kernel_data, LSTM_dense_2_kernel_len,
                                               LSTM_dense_2_bias_data, LSTM_dense_2_bias_len,
                                               dense3_out)) {
        Serial.println("predict_lstm_with_len: ERROR dense2");
        return input_seq[cap_len - 1];
    }

    float raw_out = dense3_out[0];

    // If you provide an invert_scale(raw) function in scaler.h, use it. Otherwise assume output is final.
#ifdef HAVE_INVERT_SCALE_FN
    float pred = invert_scale(raw_out);
#else
    float pred = raw_out;
#endif

    Serial.print("predict_lstm_with_len: LSTM returned: ");
    Serial.println(pred);
    Serial.print("predict_lstm_with_len: freeHeap final: ");
    Serial.println(ESP.getFreeHeap());
    return pred;
}

// Single-argument C-linkage wrapper your .ino expects.
float predict_lstm(const float *input_seq) {
    // call with default expected sequence length
    return predict_lstm_with_len(input_seq, LSTM_EXPECTED_SEQ_LEN);
}

// Provide an explicit two-argument (C++) symbol as well so other compilation units that expect it can use it.
// Note: C++ overloaded symbols are mangled; we export only the single-arg C-linkage symbol above for the Arduino sketch.
float predict_lstm_with_length(const float *input_seq, unsigned int seq_len) {
    return predict_lstm_with_len(input_seq, seq_len);
}

#ifdef __cplusplus
} // extern "C"
#endif

