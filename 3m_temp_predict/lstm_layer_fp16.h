#ifndef LSTM_LAYER_FP16_H
#define LSTM_LAYER_FP16_H

#include "model_runtime_utils.h"
#include <stdlib.h>
#include <string.h>

// LSTM forward reading FP16 weight tables
// kernel_table shape = [input_dim, 4*units]
// recurrent_table shape = [units, 4*units]
// bias_table shape = [4*units]
static inline void lstm_forward_from_table_fp16(
    const float *input_seq, int seq_len, int input_dim,
    const uint16_t *kernel_table, const int *kernel_shape,
    const uint16_t *recurrent_table, const int *recurrent_shape,
    const uint16_t *bias_table, const int *bias_shape,
    int units,
    float *output_last)
{
    float *h = (float*)malloc(sizeof(float)*units);
    float *c = (float*)malloc(sizeof(float)*units);
    if (!h || !c) { if (h) free(h); if (c) free(c); return; }
    memset(h,0,sizeof(float)*units);
    memset(c,0,sizeof(float)*units);

    int out4 = 4 * units;

    auto Kget = [&](int i, int g)->float {
        int idx = i * out4 + g;
        return get_float_from_table_fp16(kernel_table, (unsigned)idx);
    };
    auto Rget = [&](int i, int g)->float {
        int idx = i * out4 + g;
        return get_float_from_table_fp16(recurrent_table, (unsigned)idx);
    };
    auto Bget = [&](int g)->float {
        return get_float_from_table_fp16(bias_table, (unsigned)g);
    };

    for (int t = 0; t < seq_len; ++t) {
        const float *x_t = input_seq + t * input_dim;
        for (int u = 0; u < units; ++u) {
            float zi = Bget(u + 0*units);
            float zf = Bget(u + 1*units);
            float zc = Bget(u + 2*units);
            float zo = Bget(u + 3*units);

            for (int i = 0; i < input_dim; ++i) {
                float xv = x_t[i];
                zi += xv * Kget(i, u + 0*units);
                zf += xv * Kget(i, u + 1*units);
                zc += xv * Kget(i, u + 2*units);
                zo += xv * Kget(i, u + 3*units);
            }
            for (int j = 0; j < units; ++j) {
                float hj = h[j];
                zi += hj * Rget(j, u + 0*units);
                zf += hj * Rget(j, u + 1*units);
                zc += hj * Rget(j, u + 2*units);
                zo += hj * Rget(j, u + 3*units);
            }

            float i_gate = sigmoid_f(zi);
            float f_gate = sigmoid_f(zf);
            float c_tilde = tanh_f(zc);
            float o_gate = sigmoid_f(zo);

            float new_c = f_gate * c[u] + i_gate * c_tilde;
            float new_h = o_gate * tanh_f(new_c);

            c[u] = new_c;
            h[u] = new_h;
        }
    }

    for (int u = 0; u < units; ++u) output_last[u] = h[u];

    free(h);
    free(c);
}

#endif // LSTM_LAYER_FP16_H
