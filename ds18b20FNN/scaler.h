#ifndef SCALER_H
#define SCALER_H

// Initialize scaler (not required for simple arrays but kept for API symmetry)
void scaler_init();

// Scale an input value (per index)
float scale_input_val(int idx, float x);

// Undo output scaling
float scale_output_val(float y);

#endif
