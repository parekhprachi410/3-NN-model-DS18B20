// trn_inference.h
#ifndef TRN_INFERENCE_H
#define TRN_INFERENCE_H

#ifdef __cplusplus
extern "C" {
#endif

float predict_trn(const float *input_seq, unsigned int seq_len);

#ifdef __cplusplus
}
#endif

#endif // TRN_INFERENCE_H
