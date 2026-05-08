#include "scaler.h"
#include "fnn_weights.h"
#include <Arduino.h>

// optional init
void scaler_init() {}

// ----- INPUT SCALING -----
// Supports both forms:
//   INPUT_MIN_ARR[idx] / INPUT_MAX_ARR[idx]   (per timestep)
//   INPUT_MIN / INPUT_MAX                     (global)
float scale_input_val(int idx, float x)
{
#ifdef INPUT_MIN_ARR
    // Array version: If length == 1, use index 0 for all features
    #if defined(INPUT_MIN_ARR_LENGTH) && INPUT_MIN_ARR_LENGTH > 1
        float mn = INPUT_MIN_ARR[idx];
        float mx = INPUT_MAX_ARR[idx];
    #else
        float mn = INPUT_MIN_ARR[0];
        float mx = INPUT_MAX_ARR[0];
    #endif
    return (x - mn) / (mx - mn + 1e-9f);

#elif defined(INPUT_MIN) && defined(INPUT_MAX)
    return (x - INPUT_MIN) / (INPUT_MAX - INPUT_MIN + 1e-9f);

#elif defined(INPUT_MEAN_ARR)
    // Standard scaler array
    return (x - INPUT_MEAN_ARR[idx]) / (INPUT_STD_ARR[idx] + 1e-9f);

#elif defined(INPUT_MEAN)
    // Standard scaler global
    return (x - INPUT_MEAN) / (INPUT_STD + 1e-9f);

#else
    return x; // fallback: no scaling
#endif
}

// ----- OUTPUT DE-SCALING -----
float scale_output_val(float y)
{
#ifdef OUTPUT_STD
    return y * OUTPUT_STD + OUTPUT_MEAN;
#else
    return y;
#endif
}
