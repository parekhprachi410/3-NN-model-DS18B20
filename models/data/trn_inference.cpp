// trn_inference.cpp (FP16)
#include "trn_inference.h"
#include "model_runtime_utils.h"
#include "conv1d_layer_fp16.h"
#include "dense_layer_fp16.h"
#include "c_export/trn_model_data.h"

float predict_trn(const float input_seq[SEQ_LEN]) {
    float scaled_seq[SEQ_LEN];
    for (int i = 0; i < SEQ_LEN; ++i) {
        scaled_seq[i] = MIN_PARAM + ((input_seq[i] - DATA_MIN) * (SCALE_PARAM / DATA_RANGE));
    }

    int out_len1 = 0;
    int in_ch1 = conv1d_w0_shape[1];
    static float conv1_out[SEQ_LEN * 32 + 8];
    conv1d_forward_from_table_fp16(scaled_seq, SEQ_LEN, in_ch1,
                                  conv1d_w0_data, conv1d_w0_shape,
                                  conv1d_w1_data, conv1d_w1_shape,
                                  conv1_out, out_len1);

    int out_len2 = 0;
    int in_ch2 = conv1d_1_w0_shape[1];
    static float conv2_out[SEQ_LEN * 64 + 16];
    conv1d_forward_from_table_fp16(conv1_out, out_len1, in_ch2,
                                  conv1d_1_w0_data, conv1d_1_w0_shape,
                                  conv1d_1_w1_data, conv1d_1_w1_shape,
                                  conv2_out, out_len2);

    static float dense7_out[64];
    dense_forward_from_table_fp16(conv2_out, dense_7_w0_data, dense_7_w0_shape, dense_7_w1_data, dense_7_w1_shape, dense7_out);
    for (int i = 0; i < 64; ++i) dense7_out[i] = relu_f(dense7_out[i]);

    float out_final[1];
    dense_forward_from_table_fp16(dense7_out, dense_8_w0_data, dense_8_w0_shape, dense_8_w1_data, dense_8_w1_shape, out_final);

    float scaled_pred = out_final[0];
    float value = (scaled_pred - MIN_PARAM) / SCALE_PARAM;
    float temp_c = value * DATA_RANGE + DATA_MIN;
    return temp_c;
}
