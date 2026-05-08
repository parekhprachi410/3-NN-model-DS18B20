#include "scaler.h"

// MinMaxScaler parameters from your data
#define DATA_MIN 7.312f
#define DATA_MAX 38.938f
#define DATA_RANGE 31.626f
#define SCALER_MIN -0.23120218f
#define SCALER_SCALE 0.03161955f

float scale_input(float value) {
    // Scale from original range to normalized range
    return (value - DATA_MIN) / DATA_RANGE;
}

float scale_output(float value) {
    // Scale from normalized range back to original range
    return (value * DATA_RANGE) + DATA_MIN;
}