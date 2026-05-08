#include "scaler.h"
#include "trn_weights.h"

// Using min–max for input, z-score for output

float scale_input(float x)
{
    float mn = INPUT_MIN;
    float mx = INPUT_MAX;

    if (mx - mn < 1e-9f)
        return 0.0f;

    return (x - mn) / (mx - mn);   // → 0..1 range
}

float scale_output(float y)
{
    // OUTPUT_MEAN = 0, OUTPUT_STD = 1 in your model → identity
    return y * OUTPUT_STD + OUTPUT_MEAN;
}
