// scaler.h
#ifndef SCALER_H
#define SCALER_H

// Scaler constants (exported from training). Replace these with your
// actual scaler values if they differ.
static const float SCALER_MEAN = 25.0f;   // example mean in °C
static const float SCALER_SCALE = 0.1f;   // example scale (std or scale factor)

// invert_scale: converts model output (normalized) back to °C
static inline float invert_scale(float normalized) {
    return normalized * (1.0f / SCALER_SCALE) + SCALER_MEAN;
}

#endif // SCALER_H

