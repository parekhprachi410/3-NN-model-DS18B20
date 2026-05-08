#ifndef FNN_INFERENCE_H
#define FNN_INFERENCE_H

#ifdef __cplusplus
extern "C" {
#endif

void fnn_init();
float fnn_predict(float* input_sequence);

#ifdef __cplusplus
}
#endif

#endif