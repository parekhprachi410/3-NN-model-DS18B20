#ifndef DENSE_LAYER_FP16_H
#define DENSE_LAYER_FP16_H

#include "model_runtime_utils.h"
#include <stddef.h>

// Dense forward reading weights from FP16 table
// kernel_table: FP16 table row-major [in_dim, out_dim]
// bias_table: FP16 table length out_dim (or NULL)
static inline void dense_forward_from_table_fp16(
    const float *x, const uint16_t *kernel_table, const int *kernel_shape,
    const uint16_t *bias_table, const int *bias_shape,
    float *out)
{
    int in_dim = kernel_shape[0];
    int out_dim = kernel_shape[1];
    for (int j = 0; j < out_dim; ++j) {
        float acc = 0.0f;
        for (int i = 0; i < in_dim; ++i) {
            int kidx = i * out_dim + j;
            float w = get_float_from_table_fp16(kernel_table, (unsigned)kidx);
            acc += x[i] * w;
        }
        if (bias_table != NULL && bias_shape != NULL && bias_shape[0] >= out_dim) {
            acc += get_float_from_table_fp16(bias_table, (unsigned)j);
        }
        out[j] = acc;
    }
}

#endif // DENSE_LAYER_FP16_H
