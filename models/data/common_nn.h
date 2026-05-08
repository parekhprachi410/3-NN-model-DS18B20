// common_nn.h
#ifndef COMMON_NN_H
#define COMMON_NN_H

#include <stdint.h>
#include <math.h>
#include <Arduino.h>

// ---------- FP16 (IEEE 754 half) -> float conversion ----------
static inline float half_to_float(uint16_t h) {
    // Fast-ish half -> float conversion (works on little/big)
    uint32_t sign = (h >> 15) & 0x00000001;
    uint32_t exp  = (h >> 10) & 0x0000001f;
    uint32_t mant = h & 0x03ff;

    uint32_t f_sign = sign << 31;
    uint32_t f_exp;
    uint32_t f_mant;

    if (exp == 0) {
        if (mant == 0) {
            // zero
            f_exp = 0;
            f_mant = 0;
        } else {
            // subnormal -> normalize
            exp = 1;
            while ((mant & 0x0400) == 0) {
                mant <<= 1;
                exp -= 1;
            }
            mant &= 0x03ff;
            exp = exp + (127 - 15);
            f_exp = exp << 23;
            f_mant = (mant << 13);
        }
    } else if (exp == 31) {
        // Inf or NaN
        f_exp = 0xff << 23;
        f_mant = (mant ? (mant << 13) : 0);
    } else {
        // normal
        exp = exp + (127 - 15);
        f_exp = exp << 23;
        f_mant = mant << 13;
    }

    uint32_t bits = f_sign | f_exp | f_mant;
    float result;
    memcpy(&result, &bits, sizeof(result));
    return result;
}

// activations
static inline float relu_f(float x) { return x > 0.0f ? x : 0.0f; }
static inline float tanh_f(float x) { return tanhf(x); }
static inline float sigmoid_f(float x) { return 1.0f / (1.0f + expf(-x)); }

// Dense forward for fp16 weights/bias
// - kern points to uint16_t FP16 array with length kern_len
// - bias points to uint16_t FP16 array with length bias_len
// We assume exported kernel is flatten(in_dim * out_dim) in row-major
// format: for i in [0..in_dim), for j in [0..out_dim): kern[i*out_dim + j]
static inline bool dense_forward_from_table_fp16(
    const float *in_vec, unsigned int in_dim,
    const uint16_t *kern, unsigned int kern_len,
    const uint16_t *bias, unsigned int bias_len,
    float *out_vec)
{
    if (!kern || !bias || !in_vec || !out_vec) return false;
    if (bias_len == 0) return false;
    unsigned int out_dim = bias_len;
    if (kern_len == 0) return false;
    unsigned int inferred_in = kern_len / out_dim;
    if (inferred_in != in_dim) {
        // shape mismatch - attempt to proceed with inferred_in
        in_dim = inferred_in;
    }

    for (unsigned int j = 0; j < out_dim; ++j) {
        float s = half_to_float(bias[j]);
        // accumulate sum over inputs
        const uint16_t *kcol = kern + j; // start at column j with stride out_dim
        for (unsigned int i = 0; i < in_dim; ++i) {
            float w = half_to_float(kcol[i * out_dim]);
            s += in_vec[i] * w;
        }
        out_vec[j] = s;
    }
    return true;
}

// Small 1D conv (single-channel input assumed)
static inline bool conv1d_forward_from_table_fp16(
    const float *in_seq, unsigned int seq_len, unsigned int in_ch,
    const uint16_t *kern, unsigned int kern_len, // kernel flattened: kernel_width * in_ch * out_ch
    const uint16_t *bias, unsigned int bias_len, // out_ch
    float *out_seq, unsigned int &out_len)
{
    if (!kern || !bias || !in_seq || !out_seq) return false;
    // We attempt to infer kernel sizes from lengths:
    // Assume kernel_len = kernel_width * in_ch * out_ch
    if (bias_len == 0) return false;
    unsigned int out_ch = bias_len;
    if (in_ch == 0) return false;
    // try find kernel_width
    unsigned int kernel_width = kern_len / (in_ch * out_ch);
    if (kernel_width == 0) return false;

    // Valid simple SAME-less convolution (valid conv): out_len = seq_len - kernel_width + 1
    if (seq_len < kernel_width) {
        out_len = 0;
        return false;
    }
    out_len = seq_len - kernel_width + 1;

    // For each output position
    for (unsigned int pos = 0; pos < out_len; ++pos) {
        for (unsigned int oc = 0; oc < out_ch; ++oc) {
            float acc = half_to_float(bias[oc]);
            // kernel layout: kw, in_ch, out_ch => index = ((kw * in_ch) + ic) * out_ch + oc
            for (unsigned int kw = 0; kw < kernel_width; ++kw) {
                for (unsigned int ic = 0; ic < in_ch; ++ic) {
                    unsigned int in_idx = (pos + kw) * in_ch + ic;
                    float x = in_seq[in_idx];
                    unsigned int kidx = ((kw * in_ch) + ic) * out_ch + oc;
                    float w = half_to_float(kern[kidx]);
                    acc += x * w;
                }
            }
            out_seq[pos * out_ch + oc] = acc;
        }
    }
    return true;
}

#endif // COMMON_NN_H
