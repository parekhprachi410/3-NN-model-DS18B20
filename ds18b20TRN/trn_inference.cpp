#include "trn_inference.h"
#include "trn_weights.h"
#include <Arduino.h>
#include <math.h>
#include <string.h>

// ---------------- ACTIVATIONS ----------------
static inline float relu(float x) { return x > 0 ? x : 0; }

// ---------------- BUFFERS ----------------
static float bufA[5000];
static float bufB[5000];

// ------------------------------------------------------------
// Conv1D: VALID padding, stride = 1
// W layout (exported): [out_ch][in_ch][kernel]
// Input layout:  in[pos * in_ch + ch]
// Output layout: out[pos * out_ch + oc]
// ------------------------------------------------------------
static void conv1d_valid(
    const float *W,
    const float *B,
    const float *in,
    float *out,
    int L_in,
    int in_ch,
    int out_ch,
    int kernel)
{
    int L_out = L_in - kernel + 1;

    for (int pos = 0; pos < L_out; pos++)
    {
        for (int oc = 0; oc < out_ch; oc++)
        {
            float acc = B[oc];
            const float *Wrow = W + (size_t)oc * (in_ch * kernel);

            for (int k = 0; k < kernel; k++)
            {
                const float *inptr = in + (pos + k) * in_ch;
                const float *wptr = Wrow + k * in_ch;
                for (int ic = 0; ic < in_ch; ic++)
                {
                    acc += inptr[ic] * wptr[ic];
                }
            }

            out[pos * out_ch + oc] = acc;
        }
    }
}

// ---------------- DENSE ----------------
static void dense_layer(
    const float *W,
    const float *B,
    const float *in,
    float *out,
    int in_dim,
    int out_dim)
{
    for (int o = 0; o < out_dim; o++)
    {
        const float *row = W + (size_t)o * in_dim;
        float acc = B[o];
        for (int i = 0; i < in_dim; i++)
            acc += row[i] * in[i];
        out[o] = acc;
    }
}

void trn_init()
{
    // nothing needed
}

// ------------------------------------------------------------
// MAIN INFERENCE
// scaled_input must be length SEQ_LEN (your project uses 60)
// ------------------------------------------------------------
float predict_temperature(const float *scaled_input)
{
    float *cur = bufA;
    float *nxt = bufB;

    // your sequence length is 60; single-channel
    const int INPUT_LEN = 60;
    int cur_len = INPUT_LEN;
    int cur_ch  = 1;

    // load input
    for (int i = 0; i < INPUT_LEN; i++)
        cur[i] = scaled_input[i];

    // ------------------------------------------------------------
    // LAYER LOOP
    // ------------------------------------------------------------
    for (int L = 0; L < NUM_LAYERS; L++)
    {
        int kind = LAYER_KIND[L];
        int p0 = LAYER_PARAMS[L][0];
        int p1 = LAYER_PARAMS[L][1];
        int p2 = LAYER_PARAMS[L][2];

        // ---------------- CONV ----------------
        if (kind == 0)
        {
            int kernel = p0;
            int in_ch  = p1;
            int out_ch = p2;

            int L_out = cur_len - kernel + 1;

            // pick weight pointers
            const float *W = nullptr;
            const float *B = nullptr;

            switch (L)
            {
                case 0: W = W0; B = Bias0; break;
                case 1: W = W1; B = Bias1; break;
                default: return 0.0f;
            }

            conv1d_valid(W, B, cur, nxt, cur_len, in_ch, out_ch, kernel);

            // update shape
            cur_len = L_out;
            cur_ch = out_ch;

            // swap buffers
            float *tmp = cur; cur = nxt; nxt = tmp;

            // activation
            for (int i=0;i<cur_len*cur_ch;i++)
                cur[i] = relu(cur[i]);
        }

        // ---------------- DENSE ----------------
        else if (kind == 1)
        {
            int in_dim  = p0;
            int out_dim = p1;

            // flatten if needed
            int total = cur_len * cur_ch;

            float *flat = nxt;
            for (int i = 0; i < total; i++)
                flat[i] = cur[i];

            // pick weights
            const float *W = nullptr;
            const float *B = nullptr;

            switch (L)
            {
                case 2: W = W2; B = Bias2; break;
                case 3: W = W3; B = Bias3; break;
                default: return 0.0f;
            }

            dense_layer(W, B, flat, cur, in_dim, out_dim);

            cur_len = 1;
            cur_ch = out_dim;

            // activation except last layer
            if (L != NUM_LAYERS - 1)
            {
                for (int i=0;i<cur_ch;i++)
                    cur[i] = relu(cur[i]);
            }
        }
    }

    // final output scalar
    return cur[0];
}
