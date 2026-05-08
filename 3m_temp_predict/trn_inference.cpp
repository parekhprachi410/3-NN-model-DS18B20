

// trn_inference.cpp
#include "trn_inference.h"
#include "scaler.h"
#include "common_nn.h"
#include "c_models/TRN.h"
#include <Arduino.h>

extern "C" float predict_trn(const float *input_seq, unsigned int seq_len) {
    Serial.println("predict_trn: enter");
    if (!input_seq) return 0.0f;

    // quick defensive checks
    if (TRN_conv1d_kernel_len == 0 || TRN_conv1d_bias_len == 0) {
        Serial.println("predict_trn: missing conv1 tables -> fallback");
        return input_seq[seq_len-1];
    }

    unsigned int in_ch = 1;
    if (seq_len > 1024) seq_len = 1024;
    static float seq_buf[1024];
    for (unsigned int i=0;i<seq_len;i++) seq_buf[i*in_ch] = (input_seq[i] - SCALER_MEAN) * SCALER_SCALE;

    static float conv1_out[1024];
    unsigned int conv1_out_len = 0;
    bool ok = conv1d_forward_from_table_fp16(seq_buf, seq_len, in_ch,
                                             TRN_conv1d_kernel_data, TRN_conv1d_kernel_len,
                                             TRN_conv1d_bias_data, TRN_conv1d_bias_len,
                                             conv1_out, conv1_out_len);
    if (!ok) return input_seq[seq_len-1];
    for (unsigned int i=0;i<conv1_out_len * TRN_conv1d_bias_len;i++) conv1_out[i] = relu_f(conv1_out[i]);

    static float conv2_out[1024];
    unsigned int conv2_out_len = 0;
    ok = conv1d_forward_from_table_fp16(conv1_out, conv1_out_len, TRN_conv1d_bias_len,
                                        TRN_conv1d_1_kernel_data, TRN_conv1d_1_kernel_len,
                                        TRN_conv1d_1_bias_data, TRN_conv1d_1_bias_len,
                                        conv2_out, conv2_out_len);
    if (!ok) return input_seq[seq_len-1];
    for (unsigned int i=0;i<conv2_out_len * TRN_conv1d_1_bias_len;i++) conv2_out[i] = relu_f(conv2_out[i]);

    unsigned int flat_len = conv2_out_len * TRN_conv1d_1_bias_len;
    static float dense7_out[128];
    ok = dense_forward_from_table_fp16(conv2_out, flat_len,
                                       TRN_dense_7_kernel_data, TRN_dense_7_kernel_len,
                                       TRN_dense_7_bias_data, TRN_dense_7_bias_len,
                                       dense7_out);
    if (!ok) return input_seq[seq_len-1];
    for (unsigned int i=0;i<TRN_dense_7_bias_len && i<128;i++) dense7_out[i] = relu_f(dense7_out[i]);

    static float out_final[1];
    ok = dense_forward_from_table_fp16(dense7_out, TRN_dense_7_bias_len,
                                       TRN_dense_8_kernel_data, TRN_dense_8_kernel_len,
                                       TRN_dense_8_bias_data, TRN_dense_8_bias_len,
                                       out_final);
    if (!ok) return input_seq[seq_len-1];

    return invert_scale(out_final[0]);
}
