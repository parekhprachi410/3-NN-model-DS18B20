// ============================================================
// ESP32 + DS18B20 + TRN (1D-CNN) Temperature Prediction System
// CLEAN VERSION (NO SD) + FIXED RECONNECTION
// ============================================================

#include <Arduino.h>
#include <OneWire.h>
#include <DallasTemperature.h>
#include <BluetoothSerial.h>

#include "trn_inference.h"
#include "scaler.h"

// -------------------------------------------
// USER SETTINGS
// -------------------------------------------
#ifndef SEQ_LEN
#define SEQ_LEN 30
#endif

#define ONE_WIRE_BUS 4

const unsigned long PREDICTION_INTERVAL = 60000UL;

// Linear correction params
const float BLEND_BETA = 0.65f;
const float BIAS_ALPHA = 0.02f;
const float PARAM_ALPHA = 0.3f;
const float SLOPE_MIN = 0.7f;
const float SLOPE_MAX = 1.3f;
const float EPS_NUM = 1e-6f;

// -------------------------------------------
// GLOBALS
// -------------------------------------------
OneWire oneWire(ONE_WIRE_BUS);
DallasTemperature sensors(&oneWire);
BluetoothSerial BT;

bool bt_connected = false;
bool sensor_connected = true;

float temperature_history[SEQ_LEN];
int history_index = 0;
int history_count = 0;

float last_temperature = 22.0f;
unsigned long last_prediction = 0;

// Linear correction
bool linear_initialized = false;
float ema_M = 0, ema_S = 0, ema_MS = 0, ema_M2 = 0;
float learned_slope = 1.0f;
float learned_intercept = 0.0f;

bool csv_header_sent = false;

// -------------------------------------------
// BLUETOOTH CALLBACK
// -------------------------------------------
void onBTEvent(esp_spp_cb_event_t event, esp_spp_cb_param_t *param)
{
    if (event == ESP_SPP_SRV_OPEN_EVT)
        bt_connected = true;
    else if (event == ESP_SPP_CLOSE_EVT)
        bt_connected = false;
}

// -------------------------------------------
// LINEAR CORRECTION
// -------------------------------------------
void reset_linear_params()
{
    linear_initialized = false;
    learned_slope = 1.0f;
    learned_intercept = 0.0f;
}

void update_linear_correction(float model_pred, float measured)
{
    if (!linear_initialized)
    {
        ema_M = model_pred;
        ema_S = measured;
        ema_MS = model_pred * measured;
        ema_M2 = model_pred * model_pred;
        linear_initialized = true;
        return;
    }

    ema_M = (1 - BIAS_ALPHA) * ema_M + BIAS_ALPHA * model_pred;
    ema_S = (1 - BIAS_ALPHA) * ema_S + BIAS_ALPHA * measured;
    ema_MS = (1 - BIAS_ALPHA) * ema_MS + BIAS_ALPHA * (model_pred * measured);
    ema_M2 = (1 - BIAS_ALPHA) * ema_M2 + BIAS_ALPHA * (model_pred * model_pred);

    float denom = ema_M2 - ema_M * ema_M;
    float slope = (fabs(denom) > EPS_NUM) ?
        (ema_MS - ema_M * ema_S) / denom : 1.0f;

    slope = constrain(slope, SLOPE_MIN, SLOPE_MAX);

    learned_slope = (1 - PARAM_ALPHA) * learned_slope + PARAM_ALPHA * slope;
    learned_intercept = ema_S - learned_slope * ema_M;
}

float apply_linear_correction(float x)
{
    return learned_slope * x + learned_intercept;
}

// -------------------------------------------
// SENSOR READ (FIXED)
// -------------------------------------------
float read_temperature()
{
    sensors.requestTemperatures();
    float t = sensors.getTempCByIndex(0);

    // DISCONNECTED
    if (t == DEVICE_DISCONNECTED_C)
    {
        sensor_connected = false;
        return last_temperature;
    }

    // RECONNECTED
    if (!sensor_connected)
    {
        sensor_connected = true;

        for (int i = 0; i < SEQ_LEN; i++)
            temperature_history[i] = t;

        history_index = 0;
        history_count = SEQ_LEN;

        reset_linear_params();
    }

    // NORMAL FLOW
    temperature_history[history_index] = t;
    history_index = (history_index + 1) % SEQ_LEN;

    if (history_count < SEQ_LEN)
        history_count++;

    last_temperature = t;
    return t;
}

// -------------------------------------------
// PREDICTION PIPE
// -------------------------------------------
float make_prediction()
{
    if (history_count < SEQ_LEN)
        return last_temperature;

    float raw[SEQ_LEN], scaled[SEQ_LEN];

    for (int i = 0; i < SEQ_LEN; i++)
    {
        int idx = (history_index + i) % SEQ_LEN;
        raw[i] = temperature_history[idx];
        scaled[i] = scale_input(raw[i]);
    }

    float y = predict_temperature(scaled);
    return scale_output(y);
}

// -------------------------------------------
// SETUP
// -------------------------------------------
void setup()
{
    Serial.begin(115200);

    BT.begin("ESP32_TRN");
    BT.register_callback(onBTEvent);

    sensors.begin();
    sensors.setResolution(12);

    trn_init();

    float init_temp = read_temperature();

    for (int i = 0; i < SEQ_LEN; i++)
        temperature_history[i] = init_temp;

    history_index = 0;
    history_count = SEQ_LEN;
    last_temperature = init_temp;

    reset_linear_params();
}

// -------------------------------------------
// LOOP
// -------------------------------------------
void loop()
{
    unsigned long now = millis();

    if (now - last_prediction < PREDICTION_INTERVAL)
        return;

    last_prediction = now;

    float current = read_temperature();

    float pred_raw = make_prediction();

    if (sensor_connected)
        update_linear_correction(pred_raw, current);

    float pred_corrected = apply_linear_correction(pred_raw);

    float final_pred =
        BLEND_BETA * pred_corrected +
        (1 - BLEND_BETA) * current;

    // CSV OUTPUT
    String csv;

    if (!csv_header_sent)
    {
        csv = "timestamp,real_temperature,prediction,status\n";
        csv_header_sent = true;
    }

    csv += String(now) + ",";
    csv += String(current, 2) + ",";
    csv += String(final_pred, 2) + ",";
    csv += (sensor_connected ? "CONNECTED" : "DISCONNECTED");
    csv += "\n";

    if (bt_connected)
        BT.print(csv);
    else
        Serial.print(csv);
}