// ======================================================
// ESP32 + DS18B20 + Bluetooth + SD Logging
// FNN Temperature Predictor
// ======================================================

#include "fnn_inference.h"
#include "scaler.h"

#include <OneWire.h>
#include <DallasTemperature.h>
#include <Arduino.h>
#include <BluetoothSerial.h>
#include <SD.h>
#include <SPI.h>

#ifndef SEQ_LEN
#define SEQ_LEN 60
#endif

// ======================================================
// DS18B20 Sensor
// ======================================================
#define ONE_WIRE_BUS 4

OneWire oneWire(ONE_WIRE_BUS);
DallasTemperature sensors(&oneWire);

// ======================================================
// Bluetooth
// ======================================================
BluetoothSerial BT;
bool bt_connected = false;

// ======================================================
// SD Card
// ======================================================
#define SD_CS_PIN 5
File logFile;

// ======================================================
// Prediction Parameters
// ======================================================
const unsigned long PREDICTION_INTERVAL = 600000UL;

const float SPIKE_THRESHOLD = 0.6f;
const int FAST_ADOPT_STEPS = 5;

const float BLEND_BETA = 0.65f;

const bool ENABLE_LINEAR_CORRECTION = true;

const float BIAS_ALPHA = 0.02f;
const float PARAM_ALPHA = 0.3f;

const float STABLE_DELTA_THRESH = 0.15f;

const float SLOPE_MIN = 0.7f;
const float SLOPE_MAX = 1.3f;

const float EPS_NUM = 1e-6f;

// ======================================================
// History Buffer
// ======================================================
float temperature_history[SEQ_LEN];

int history_index = 0;
int history_count = 0;

bool sensor_connected = true;

float last_temperature = 22.0f;

unsigned long last_prediction = 0;

int fast_adopt_counter = 0;

// ======================================================
// Linear Correction
// ======================================================
float ema_M = 0;
float ema_S = 0;
float ema_MS = 0;
float ema_M2 = 0;

bool linear_initialized = false;

float learned_slope = 1.0f;
float learned_intercept = 0.0f;

// ======================================================
// Bluetooth Callbacks
// ======================================================
void onBTConnect(esp_spp_cb_event_t event, esp_spp_cb_param_t *param)
{
  bt_connected = true;
}

void onBTDisconnect(esp_spp_cb_event_t event, esp_spp_cb_param_t *param)
{
  bt_connected = false;
}

// ======================================================
// Function Declarations
// ======================================================
float make_prediction();
float read_temperature();
void simulate_temperature_variation();

void update_linear_correction(float model_pred, float measured);
float apply_linear_correction(float model_pred);
void reset_linear_params();

void log_to_sd(String data);

void checkSensorStatus();

// ======================================================
// SETUP
// ======================================================
void setup()
{
  Serial.begin(115200);

  BT.begin("ESP32_BT");

  BT.register_callback([](esp_spp_cb_event_t event, esp_spp_cb_param_t *param)
  {
    if (event == ESP_SPP_SRV_OPEN_EVT) onBTConnect(event, param);
    if (event == ESP_SPP_CLOSE_EVT) onBTDisconnect(event, param);
  });

  SPI.begin(18, 19, 23, SD_CS_PIN);

  if (!SD.begin(SD_CS_PIN))
    Serial.println("SD card initialization failed!");
  else
    Serial.println("SD card initialized.");

  sensors.begin();
  sensors.setResolution(12);

  sensor_connected = sensors.getDeviceCount() > 0;

  fnn_init();
  scaler_init();

  float initTemp = read_temperature();

  for (int i = 0; i < SEQ_LEN; i++)
    temperature_history[i] = initTemp;

  history_count = SEQ_LEN;

  last_temperature = initTemp;

  reset_linear_params();

  // CSV header
  Serial.println("timestamp_ms,real_temperature_C,predicted_temperature_C,sensor_status");
}

// ======================================================
// LOOP
// ======================================================
void loop()
{
  checkSensorStatus();

  unsigned long now = millis();

  if (now - last_prediction < PREDICTION_INTERVAL)
    return;

  last_prediction = now;

  float current_temperature = read_temperature();

  float final_prediction;

  float model_pred;

  if (sensor_connected)
  {
    float delta = current_temperature - last_temperature;

    if (fabs(delta) >= SPIKE_THRESHOLD)
    {
      for (int i = 0; i < SEQ_LEN; i++)
        temperature_history[i] = current_temperature;

      history_index = 0;
      history_count = SEQ_LEN;

      final_prediction = current_temperature;
    }
    else
    {
      temperature_history[history_index] = current_temperature;

      history_index = (history_index + 1) % SEQ_LEN;

      if (history_count < SEQ_LEN)
        history_count++;

      model_pred = make_prediction();

      if (ENABLE_LINEAR_CORRECTION &&
          fabs(current_temperature - last_temperature) < STABLE_DELTA_THRESH &&
          history_count >= SEQ_LEN)
      {
        update_linear_correction(model_pred, current_temperature);
      }

      float corrected = apply_linear_correction(model_pred);

      final_prediction =
          BLEND_BETA * corrected +
          (1.0f - BLEND_BETA) * current_temperature;
    }
  }
  else
  {
    simulate_temperature_variation();

    model_pred = make_prediction();

    float corrected =
        ENABLE_LINEAR_CORRECTION ?
        apply_linear_correction(model_pred) :
        model_pred;

    final_prediction =
        BLEND_BETA * corrected +
        (1.0f - BLEND_BETA) * last_temperature;
  }

  last_temperature = current_temperature;

  // ======================================================
  // CSV OUTPUT
  // ======================================================

  unsigned long timestamp = millis();

  String csv;

  csv += String(timestamp);
  csv += ",";

  if (sensor_connected)
    csv += String(current_temperature, 2);

  csv += ",";
  csv += String(final_prediction, 2);
  csv += ",";

  csv += (sensor_connected ? "CONNECTED" : "DISCONNECTED");

  if (bt_connected)
    BT.println(csv);
  else
    Serial.println(csv);

  log_to_sd(csv + "\n");
}

// ======================================================
// SENSOR STATUS CHECK
// ======================================================
void checkSensorStatus()
{
  sensors.requestTemperatures();

  float tempC = sensors.getTempCByIndex(0);

  if (tempC != DEVICE_DISCONNECTED_C)
  {
    if (!sensor_connected)
    {
      sensor_connected = true;

      for (int i = 0; i < SEQ_LEN; i++)
        temperature_history[i] = tempC;

      history_index = 0;
      history_count = SEQ_LEN;

      last_temperature = tempC;

      reset_linear_params();

      Serial.println("Sensor reconnected.");
    }
  }
  else
  {
    sensor_connected = false;
  }
}

// ======================================================
// SD Logging
// ======================================================
void log_to_sd(String data)
{
  logFile = SD.open("/temp_log.csv", FILE_WRITE);

  if (logFile)
  {
    logFile.print(data);
    logFile.close();
  }
}

// ======================================================
// Linear Correction
// ======================================================
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
  ema_MS = (1 - BIAS_ALPHA) * ema_MS + BIAS_ALPHA * model_pred * measured;
  ema_M2 = (1 - BIAS_ALPHA) * ema_M2 + BIAS_ALPHA * model_pred * model_pred;

  float denom = ema_M2 - ema_M * ema_M;

  float slope =
      fabs(denom) > EPS_NUM ?
      (ema_MS - ema_M * ema_S) / denom :
      1.0f;

  slope = constrain(slope, SLOPE_MIN, SLOPE_MAX);

  learned_slope =
      (1 - PARAM_ALPHA) * learned_slope +
      PARAM_ALPHA * slope;

  learned_intercept = ema_S - learned_slope * ema_M;
}

float apply_linear_correction(float model_pred)
{
  return learned_slope * model_pred + learned_intercept;
}

void reset_linear_params()
{
  linear_initialized = false;

  learned_slope = 1.0f;
  learned_intercept = 0.0f;
}

// ======================================================
// Temperature Reading
// ======================================================
float read_temperature()
{
  sensors.requestTemperatures();

  float tempC = sensors.getTempCByIndex(0);

  if (tempC == DEVICE_DISCONNECTED_C)
  {
    sensor_connected = false;

    return last_temperature;
  }

  return tempC;
}

// ======================================================
// Simulation when sensor disconnected
// ======================================================
void simulate_temperature_variation()
{
  float drift = random(-10, 10) / 100.0f;

  temperature_history[history_index] =
      last_temperature + drift;

  history_index = (history_index + 1) % SEQ_LEN;
}

// ======================================================
// FNN Prediction
// ======================================================
float make_prediction()
{
  if (history_count < SEQ_LEN)
    return last_temperature;

  float raw[SEQ_LEN];
  float scaled[SEQ_LEN];

  for (int i = 0; i < SEQ_LEN; i++)
  {
    raw[i] =
        temperature_history[(history_index + i) % SEQ_LEN];

    scaled[i] =
        scale_input_val(i, raw[i]);
  }

  float model_raw = predict_temperature(scaled);

  return scale_output_val(model_raw);
}