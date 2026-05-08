// model_glue.h
// Single place to map exported model symbols to the generic names
// expected by the inference code. Auto-created by hand from your
// c_models contents / name_mapping.json.
//
// Place this file in the same folder as your .ino and inference .cpp files.
// Then #include "model_glue.h" early in compilation (in your .ino or in each inference .cpp).

#ifndef MODEL_GLUE_H
#define MODEL_GLUE_H

// include the exported model headers (adjust names if your headers have other filenames)
#include "c_models/FNN.h"
#include "c_models/LSTM.h"
#include "c_models/TRN.h"

// -----------------------
// FNN mappings (dense_3 .. dense_6)
// -----------------------
#define dense_3_w0_data  FNN_dense_3_kernel_data
#define dense_3_w1_data  FNN_dense_3_bias_data
#define dense_4_w0_data  FNN_dense_4_kernel_data
#define dense_4_w1_data  FNN_dense_4_bias_data
#define dense_5_w0_data  FNN_dense_5_kernel_data
#define dense_5_w1_data  FNN_dense_5_bias_data
#define dense_6_w0_data  FNN_dense_6_kernel_data
#define dense_6_w1_data  FNN_dense_6_bias_data

// create shape arrays for the FNN dense layers using kernel_len and bias_len
// kernel_len = input_dim * units ; bias_len = units
static const int dense_3_w0_shape[2] = {
    (int)(FNN_dense_3_kernel_len / ( (FNN_dense_3_bias_len>0) ? FNN_dense_3_bias_len : 1 )), // input_dim
    (int)FNN_dense_3_bias_len // units/out_dim
};
static const int dense_3_w1_shape[1] = { (int)FNN_dense_3_bias_len };

static const int dense_4_w0_shape[2] = {
    (int)(FNN_dense_4_kernel_len / ( (FNN_dense_4_bias_len>0) ? FNN_dense_4_bias_len : 1 )),
    (int)FNN_dense_4_bias_len
};
static const int dense_4_w1_shape[1] = { (int)FNN_dense_4_bias_len };

static const int dense_5_w0_shape[2] = {
    (int)(FNN_dense_5_kernel_len / ( (FNN_dense_5_bias_len>0) ? FNN_dense_5_bias_len : 1 )),
    (int)FNN_dense_5_bias_len
};
static const int dense_5_w1_shape[1] = { (int)FNN_dense_5_bias_len };

static const int dense_6_w0_shape[2] = {
    (int)(FNN_dense_6_kernel_len / ( (FNN_dense_6_bias_len>0) ? FNN_dense_6_bias_len : 1 )),
    (int)FNN_dense_6_bias_len
};
static const int dense_6_w1_shape[1] = { (int)FNN_dense_6_bias_len };

// -----------------------
// TRN mappings (conv1d, conv1d_1, dense_7/8)
// -----------------------
#define conv1d_w0_data         TRN_conv1d_kernel_data
#define conv1d_w1_data         TRN_conv1d_bias_data
#define conv1d_1_w0_data       TRN_conv1d_1_kernel_data
#define conv1d_1_w1_data       TRN_conv1d_1_bias_data

#define dense_7_w0_data  TRN_dense_7_kernel_data
#define dense_7_w1_data  TRN_dense_7_bias_data
#define dense_8_w0_data  TRN_dense_8_kernel_data
#define dense_8_w1_data  TRN_dense_8_bias_data

// For TRN dense shapes (7 and 8) create shape arrays like FNN above:
static const int dense_7_w0_shape[2] = {
    (int)(TRN_dense_7_kernel_len / ( (TRN_dense_7_bias_len>0) ? TRN_dense_7_bias_len : 1 )),
    (int)TRN_dense_7_bias_len
};
static const int dense_7_w1_shape[1] = { (int)TRN_dense_7_bias_len };

static const int dense_8_w0_shape[2] = {
    (int)(TRN_dense_8_kernel_len / ( (TRN_dense_8_bias_len>0) ? TRN_dense_8_bias_len : 1 )),
    (int)TRN_dense_8_bias_len
};
static const int dense_8_w1_shape[1] = { (int)TRN_dense_8_bias_len }

// NOTE: conv1d shapes are more complex (kernel_size,in_ch,out_ch). The exporter
// did not give explicit shape arrays. If your conv1d forward function expects a
// shape array, we will need to compute/insert it when we know kernel_size & channels.
// For now, if your trn_inference.cpp only uses conv kernel pointer + kernel_len/bias_len,
// it may still work. If a compile-time error about conv1d_*_shape appears, paste the first
// error here and I'll compute the correct values from the header constants.

// -----------------------
// LSTM mappings
// -----------------------
#define lstm_w0_data    LSTM_lstm_kernel_data
#define lstm_w1_data    LSTM_lstm_recurrent_data
#define lstm_w2_data    LSTM_lstm_bias_data

#define lstm_1_w0_data  LSTM_lstm_1_kernel_data
#define lstm_1_w1_data  LSTM_lstm_1_recurrent_data
#define lstm_1_w2_data  LSTM_lstm_1_bias_data

// Dense head for LSTM
#define dense_w0_data   LSTM_dense_kernel_data
#define dense_w1_data   LSTM_dense_bias_data

#define dense_1_w0_data LSTM_dense_1_kernel_data
#define dense_1_w1_data LSTM_dense_1_bias_data

#define dense_2_w0_data LSTM_dense_2_kernel_data
#define dense_2_w1_data LSTM_dense_2_bias_data

// Provide shapes for LSTM dense heads (constructed same as prior dense)
static const int dense_w0_shape[2] = {
    (int)(LSTM_dense_kernel_len / ( (LSTM_dense_bias_len>0) ? LSTM_dense_bias_len : 1 )),
    (int)LSTM_dense_bias_len
};
static const int dense_w1_shape[1] = { (int)LSTM_dense_bias_len };

static const int dense_1_w0_shape[2] = {
    (int)(LSTM_dense_1_kernel_len / ( (LSTM_dense_1_bias_len>0) ? LSTM_dense_1_bias_len : 1 )),
    (int)LSTM_dense_1_bias_len
};
static const int dense_1_w1_shape[1] = { (int)LSTM_dense_1_bias_len };

static const int dense_2_w0_shape[2] = {
    (int)(LSTM_dense_2_kernel_len / ( (LSTM_dense_2_bias_len>0) ? LSTM_dense_2_bias_len : 1 )),
    (int)LSTM_dense_2_bias_len
};
static const int dense_2_w1_shape[1] = { (int)LSTM_dense_2_bias_len };

#endif // MODEL_GLUE_H
