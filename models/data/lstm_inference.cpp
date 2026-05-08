// lstm_inference.cpp (FP16)
#include "lstm_inference.h"
#include "model_runtime_utils.h"
#include "lstm_layer_fp16.h"
#include "dense_layer_fp16.h"
#include "c_export/lstm_model_data.h"
#include <string.h>

static void dense_act_fp16(const float *in_vec, const uint16_t *kern, const int *kern_shape,
                           const uint16_t *bias, const int *bias_shape,
                           float *out_vec, const char *activation)
{
    dense_forward_from_table_fp16(in_vec, kern, kern_shape, bias, bias_shape, out_vec);
    int out_dim = kern_shape[1];
    if (!activation) return;
    if (strcmp(activation, "relu") == 0) {
        for (int i = 0; i < out_dim; ++i) out_vec[i] = relu_f(out_vec[i]);
    } else if (strcmp(activation, "tanh") == 0) {
        for (int i = 0; i < out_dim; ++i) out_vec[i] = tanh_f(out_vec[i]);
    } else if (strcmp(activation, "sigmoid") == 0) {
        for (int i = 0; i < out_dim; ++i) out_vec[i] = sigmoid_f(out_vec[i]);
    }
}

float predict_lstm(const float input_seq[SEQ_LEN]) {
    float scaled_in[SEQ_LEN];
    for (int i = 0; i < SEQ_LEN; ++i)
        scaled_in[i] = MIN_PARAM + ((input_seq[i] - DATA_MIN) * (SCALE_PARAM / DATA_RANGE));

    // LSTM 1
    int k0_out4 = lstm_w0_shape[1];
    int units0 = k0_out4 / 4;
    float *lstm1_out_last = (float*)malloc(sizeof(float) * units0);
    if (!lstm1_out_last) return DATA_MIN;
    lstm_forward_from_table_fp16(scaled_in, SEQ_LEN, 1,
                                lstm_w0_data, lstm_w0_shape,
                                lstm_w1_data, lstm_w1_shape,
                                lstm_w2_data, lstm_w2_shape,
                                units0, lstm1_out_last);

    // LSTM 2 (1-step sequence)
    int units1 = lstm_1_w0_shape[1] / 4;
    float *lstm2_out_last = (float*)malloc(sizeof(float) * units1);
    if (!lstm2_out_last) { free(lstm1_out_last); return DATA_MIN; }

    lstm_forward_from_table_fp16(lstm1_out_last, 1, units0,
                                lstm_1_w0_data, lstm_1_w0_shape,
                                lstm_1_w1_data, lstm_1_w1_shape,
                                lstm_1_w2_data, lstm_1_w2_shape,
                                units1, lstm2_out_last);

    // Dense chain
    float dense_out1[16];
    dense_act_fp16(lstm2_out_last, dense_w0_data, dense_w0_shape, dense_w1_data, dense_w1_shape, dense_out1, "relu");

    float dense_out2[8];
    dense_act_fp16(dense_out1, dense_1_w0_data, dense_1_w0_shape, dense_1_w1_data, dense_1_w1_shape, dense_out2, "relu");

    float dense_out3[1];
    dense_act_fp16(dense_out2, dense_2_w0_data, dense_2_w0_shape, dense_2_w1_data, dense_2_w1_shape, dense_out3, "linear");

    float scaled_prediction = dense_out3[0];

    free(lstm1_out_last);
    free(lstm2_out_last);

    float value = (scaled_prediction - MIN_PARAM) / SCALE_PARAM;
    float temp_c = value * DATA_RANGE + DATA_MIN;
    return temp_c;
}
