// lstm_inference.h
#ifndef LSTM_INFERENCE_H
#define LSTM_INFERENCE_H

#ifdef __cplusplus
extern "C" {
#endif

float predict_lstm(const float *input_seq, unsigned int seq_len);

#ifdef __cplusplus
}
#endif

#endif // LSTM_INFERENCE_H
