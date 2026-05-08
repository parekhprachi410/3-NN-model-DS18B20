#include "fnn_inference.h"
#include "fnn_weights.h"
#include <Arduino.h>
#include <math.h>

// ---- Buffer sizes ----
// Largest layer output = max(128, 64, 32, 1) = 128
static float bufA[128];
static float bufB[128];

// -------- ACTIVATIONS ----------
static inline float relu(float x) { return x > 0 ? x : 0; }
static inline float sigmoid(float x) { return 1.0f / (1.0f + expf(-x)); }

static void apply_activation(float *v, int n, const char *act)
{
    if (!act) return;

    if (strcmp(act, "relu") == 0)
    {
        for (int i = 0; i < n; i++)
            v[i] = relu(v[i]);
    }
    else if (strcmp(act, "tanh") == 0)
    {
        for (int i = 0; i < n; i++)
            v[i] = tanhf(v[i]);
    }
    else if (strcmp(act, "sigmoid") == 0)
    {
        for (int i = 0; i < n; i++)
            v[i] = sigmoid(v[i]);
    }
    // linear → do nothing
}

// -------- DENSE LAYER ----------
static void dense(
    const float *W, const float *B,
    const float *inp, float *out,
    int out_dim, int in_dim)
{
    for (int o = 0; o < out_dim; o++)
    {
        const float *row = W + (size_t)o * in_dim;
        float acc = 0.0f;

        for (int i = 0; i < in_dim; i++)
            acc += row[i] * inp[i];

        out[o] = acc + B[o];
    }
}

void fnn_init()
{
    // nothing needed for float model
}

float predict_temperature(const float *scaled_input)
{
    float *in = bufA;
    float *out = bufB;

    // ---- Copy first input ----
    int first_in = LAYER_IN[0];   // 60
    for (int i = 0; i < first_in; i++)
        in[i] = scaled_input[i];

    // ---- LAYER 0 ----
    dense(W0, Bias0, in, out, LAYER_OUT[0], LAYER_IN[0]);   // 128×60
    apply_activation(out, LAYER_OUT[0], LAYER_ACT[0]);   // relu

    // swap
    { float *tmp = in; in = out; out = tmp; }

    // ---- LAYER 1 ----
    dense(W1, Bias1, in, out, LAYER_OUT[1], LAYER_IN[1]);   // 64×128
    apply_activation(out, LAYER_OUT[1], LAYER_ACT[1]);   // relu

    { float *tmp = in; in = out; out = tmp; }

    // ---- LAYER 2 ----
    dense(W2, Bias2, in, out, LAYER_OUT[2], LAYER_IN[2]);   // 32×64
    apply_activation(out, LAYER_OUT[2], LAYER_ACT[2]);   // relu

    { float *tmp = in; in = out; out = tmp; }

    // ---- LAYER 3 (final) ----
    dense(W3, Bias3, in, out, LAYER_OUT[3], LAYER_IN[3]);   // 1×32
    apply_activation(out, LAYER_OUT[3], LAYER_ACT[3]);   // linear

    // Final output is single neuron
    return out[0];
}
