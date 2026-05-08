// fnn_inference.cpp (FP16)
#include "fnn_inference.h"
#include "model_runtime_utils.h"
#include "dense_layer_fp16.h"
#include "c_export/fnn_model_data.h"

// predict_fnn: returns temperature in °C (inverse scaled)
float predict_fnn(const float input_seq[SEQ_LEN]) {
    float scaled_in[SEQ_LEN];
    for (int i = 0; i < SEQ_LEN; ++i) {
        scaled_in[i] = MIN_PARAM + ((input_seq[i] - DATA_MIN) * (SCALE_PARAM / DATA_RANGE));
    }

    static float buf1[128];
    dense_forward_from_table_fp16(scaled_in, dense_3_w0_data, dense_3_w0_shape, dense_3_w1_data, dense_3_w1_shape, buf1);
    for (int i = 0; i < 128; ++i) buf1[i] = relu_f(buf1[i]);

    static float buf2[64];
    dense_forward_from_table_fp16(buf1, dense_4_w0_data, dense_4_w0_shape, dense_4_w1_data, dense_4_w1_shape, buf2);
    for (int i = 0; i < 64; ++i) buf2[i] = relu_f(buf2[i]);

    static float buf3[32];
    dense_forward_from_table_fp16(buf2, dense_5_w0_data, dense_5_w0_shape, dense_5_w1_data, dense_5_w1_shape, buf3);
    for (int i = 0; i < 32; ++i) buf3[i] = relu_f(buf3[i]);

    float out_final[1];
    dense_forward_from_table_fp16(buf3, dense_6_w0_data, dense_6_w0_shape, dense_6_w1_data, dense_6_w1_shape, out_final);

    float scaled_pred = out_final[0];
    float value = (scaled_pred - MIN_PARAM) / SCALE_PARAM;
    float temp_c = value * DATA_RANGE + DATA_MIN;
    return temp_c;
}
