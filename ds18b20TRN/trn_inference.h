#ifndef TRN_INFERENCE_H
#define TRN_INFERENCE_H

#ifdef __cplusplus
extern "C" {
#endif

void trn_init();
float predict_temperature(const float *scaled_input);

#ifdef __cplusplus
}
#endif

#endif
