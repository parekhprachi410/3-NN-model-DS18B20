#ifndef LSTM_INFERENCE_H
#define LSTM_INFERENCE_H

#ifdef __cplusplus
extern "C" {
#endif

void lstm_init();
float predict_temperature(float* input_sequence);

#ifdef __cplusplus
}
#endif

#endif