// fnn_inference.h
#ifndef FNN_INFERENCE_H
#define FNN_INFERENCE_H

#ifdef __cplusplus
extern "C" {
#endif

float predict_fnn(const float *input_seq, unsigned int input_len);

#ifdef __cplusplus
}
#endif

#endif // FNN_INFERENCE_H
