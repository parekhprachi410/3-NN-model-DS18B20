
// 3m_temp_predict.ino
#include <OneWire.h>
#include <DallasTemperature.h>

#include "fnn_inference.h"
#include "trn_inference.h"
#include "lstm_inference.h"

// DS18B20 setup (adjust pin)
#define ONE_WIRE_BUS 4
OneWire oneWire(ONE_WIRE_BUS);
DallasTemperature sensors(&oneWire);

const unsigned int SEQ_LEN = 60; // history length
static float temperature_history[SEQ_LEN];
static unsigned int history_idx = 0;
static unsigned int history_count = 0;

void setup() {
  Serial.begin(115200);
  delay(200);
  Serial.println();
  Serial.println("=== 3-model Temperature Predictor (ESP32) ===");
  sensors.begin();
  // init history with some reasonable defaults
  for (unsigned int i=0;i<SEQ_LEN;i++) temperature_history[i] = 25.0f;
}

void loop() {
  sensors.requestTemperatures();
  float t = sensors.getTempCByIndex(0);
  if (t == DEVICE_DISCONNECTED_C) {
    Serial.println("temperature sensor disconnected");
    delay(1000);
    return;
  }

  // store circularly
  temperature_history[history_idx] = t;
  history_idx = (history_idx + 1) % SEQ_LEN;
  if (history_count < SEQ_LEN) history_count++;

  // print realtime
  Serial.println("real time:");
  Serial.print("temperature = ");
  Serial.print(t, 2);
  Serial.println(" °C");
  Serial.println();

  // Only run prediction when we have at least SEQ_LEN history
  Serial.println("prediction:");
  if (history_count < SEQ_LEN) {
    Serial.print("Buffer not full yet; need ");
    Serial.print(SEQ_LEN);
    Serial.println(" historical samples to run predictions.");
    Serial.println();
    delay(1000);
    return;
  }

  // Prepare a linear ordered input sequence (oldest -> newest)
  static float seq_in[SEQ_LEN];
  unsigned int ix = history_idx;
  for (unsigned int i=0;i<SEQ_LEN;i++) {
    seq_in[i] = temperature_history[ix];
    ix = (ix + 1) % SEQ_LEN;
  }

  // Call models (they only return float)
  float lstm_out = predict_lstm(seq_in, SEQ_LEN);
  float fnn_out  = predict_fnn(seq_in, SEQ_LEN);
  float trn_out  = predict_trn(seq_in, SEQ_LEN);

  // Print results in the exact requested format
  Serial.println("prediction:");
  Serial.print("LSTM temperature = ");
  Serial.print(lstm_out, 2);
  Serial.println(" °C (model1)");

  Serial.print("FNN temperature  = ");
  Serial.print(fnn_out, 2);
  Serial.println(" °C (model2)");

  Serial.print("1D CNN temperature = ");
  Serial.print(trn_out, 2);
  Serial.println(" °C (model3)");

  Serial.println();

  delay(1000);
}
