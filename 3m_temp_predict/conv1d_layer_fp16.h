#ifndef CONV1D_LAYER_FP16_H
#define CONV1D_LAYER_FP16_H

#include "model_runtime_utils.h"
#include <stddef.h>

// Conv1D forward reading FP16 kernel table.
// kernel_shape: [kernel_size, in_ch, filters]
// input layout: input[seq_len * in_ch] (time-major)
// output buffer must be (out_len * filters)
static inline void conv1d_forward_from_table_fp16(
    const float *input, int seq_len, int in_ch,
    const uint16_t *kernel_table, const int *kernel_shape,
    const uint16_t *bias_table, const int *bias_shape,
    float *output, int &out_len)
{
    int kernel_size = kernel_shape[0];
    int k_in_ch = kernel_shape[1];
    int filters = kernel_shape[2];
    int stride = 1;
    int pad = 0;
    out_len = (seq_len + 2*pad - kernel_size) / stride + 1;
    for (int t = 0; t < out_len; ++t) {
        for (int f = 0; f < filters; ++f) {
            float acc = 0.0f;
            for (int k = 0; k < kernel_size; ++k) {
                int inpos = t*stride + k - pad;
                if (inpos < 0 || inpos >= seq_len) continue;
                for (int c = 0; c < in_ch; ++c) {
                    int kidx = (k * k_in_ch + c) * filters + f;
                    float w = get_float_from_table_fp16(kernel_table, (unsigned)kidx);
                    float v = input[inpos * in_ch + c];
                    acc += v * w;
                }
            }
            if (bias_table) acc += get_float_from_table_fp16(bias_table, (unsigned)f);
            output[t * filters + f] = acc;
        }
    }
}

#endif // CONV1D_LAYER_FP16_H
