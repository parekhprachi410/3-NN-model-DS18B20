#ifndef MODEL_RUNTIME_UTILS_H
#define MODEL_RUNTIME_UTILS_H

#include <stdint.h>
#include <string.h>
#include <math.h>

// Convert IEEE 754 binary16 (float16) bit pattern to float32
static inline float half_to_float(uint16_t h) {
    uint32_t sign = (h & 0x8000u) << 16;
    uint32_t exp  = (h & 0x7C00u) >> 10;
    uint32_t mant = (h & 0x03FFu);

    uint32_t f;
    if (exp == 0) {
        if (mant == 0) {
            f = sign;
        } else {
            // subnormal
            exp = 1;
            while ((mant & 0x0400u) == 0) {
                mant <<= 1;
                exp--;
            }
            mant &= 0x03FFu;
            uint32_t fexp = (exp - 15 + 127) << 23;
            uint32_t fmant = mant << 13;
            f = sign | fexp | fmant;
        }
    } else if (exp == 0x1Fu) {
        // Inf or NaN
        uint32_t fexp = 0xFFu << 23;
        uint32_t fmant = mant << 13;
        f = sign | fexp | fmant;
    } else {
        uint32_t fexp = (exp - 15 + 127) << 23;
        uint32_t fmant = mant << 13;
        f = sign | fexp | fmant;
    }

    float out;
    memcpy(&out, &f, sizeof(f));
    return out;
}

// Read float value from FP16 table (uint16_t array) and return float32
static inline float get_float_from_table_fp16(const uint16_t *table, unsigned int idx) {
    uint16_t h = table[idx];
    return half_to_float(h);
}

// activations (float32 math)
static inline float relu_f(float x) { return x > 0.0f ? x : 0.0f; }
static inline float tanh_f(float x) { return tanhf(x); }
static inline float sigmoid_f(float x) { return 1.0f / (1.0f + expf(-x)); }
static inline float linear_f(float x) { return x; }

#endif // MODEL_RUNTIME_UTILS_H
