

#ifndef FNN_INFERENCE_H
#define FNN_INFERENCE_H

#ifdef __cplusplus
extern "C" {
#endif

// Initialize FNN (optional for float models)
void fnn_init();

// Run prediction
// Input: scaled_input[length = LAYER_IN[0]]
// Output: predicted temperature (still scaled)
float predict_temperature(const float *scaled_input);

#ifdef __cplusplus
}
#endif

#endif
