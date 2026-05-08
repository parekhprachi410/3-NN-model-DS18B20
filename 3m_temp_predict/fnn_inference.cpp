// fnn_inference.cpp
#include "fnn_inference.h"
#include "scaler.h"
#include "common_nn.h"
#include "c_models/FNN.h"
#include <Arduino.h>

extern "C" float predict_fnn(const float *input_seq, unsigned int input_len) {
    Serial.println("predict_fnn: enter");
    if (!input_seq) return 0.0f;

    unsigned int in_len = input_len;
    if (in_len > 60) in_len = 60;
    float scaled_in[60];
    for (unsigned int i=0;i<in_len;i++) scaled_in[i] = (input_seq[i] - SCALER_MEAN) * SCALER_SCALE;

    static float buf1[128], buf2[64], buf3[32], out_final[1];

    if (FNN_dense_3_kernel_len == 0 || FNN_dense_3_bias_len == 0) {
        Serial.println("predict_fnn: missing tables -> fallback");
        return input_seq[in_len-1];
    }

    bool ok = dense_forward_from_table_fp16(scaled_in, in_len,
                                           FNN_dense_3_kernel_data, FNN_dense_3_kernel_len,
                                           FNN_dense_3_bias_data, FNN_dense_3_bias_len,
                                           buf1);
    if (!ok) return input_seq[in_len-1];
    for (unsigned int i=0;i<FNN_dense_3_bias_len && i<128;i++) buf1[i] = relu_f(buf1[i]);

    ok = dense_forward_from_table_fp16(buf1, FNN_dense_3_bias_len,
                                       FNN_dense_4_kernel_data, FNN_dense_4_kernel_len,
                                       FNN_dense_4_bias_data, FNN_dense_4_bias_len,
                                       buf2);
    if (!ok) return input_seq[in_len-1];
    for (unsigned int i=0;i<FNN_dense_4_bias_len && i<64;i++) buf2[i] = relu_f(buf2[i]);

    ok = dense_forward_from_table_fp16(buf2, FNN_dense_4_bias_len,
                                       FNN_dense_5_kernel_data, FNN_dense_5_kernel_len,
                                       FNN_dense_5_bias_data, FNN_dense_5_bias_len,
                                       buf3);
    if (!ok) return input_seq[in_len-1];
    for (unsigned int i=0;i<FNN_dense_5_bias_len && i<32;i++) buf3[i] = relu_f(buf3[i]);

    ok = dense_forward_from_table_fp16(buf3, FNN_dense_5_bias_len,
                                       FNN_dense_6_kernel_data, FNN_dense_6_kernel_len,
                                       FNN_dense_6_bias_data, FNN_dense_6_bias_len,
                                       out_final);
    if (!ok) return input_seq[in_len-1];

    return invert_scale(out_final[0]);
}

