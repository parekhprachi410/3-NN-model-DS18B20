#ifndef TRN_INFERENCE_H
#define TRN_INFERENCE_H

#ifdef __cplusplus
extern "C" {
#endif

void trn_init();
float trn_predict(float* input_sequence);

#ifdef __cplusplus
}
#endif

#endif