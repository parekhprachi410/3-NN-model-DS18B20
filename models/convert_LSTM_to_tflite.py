# convert_LSTM_to_tflite.py
import tensorflow as tf
import os, traceback

KERAS_PATH = "LSTM.keras"   # <-- your file
OUT_TFLITE_F16 = "LSTM_select_float16.tflite"
OUT_TFLITE_DEFAULT = "LSTM_select_default.tflite"
OUT_TFLITE_SIMPLE = "LSTM_simple.tflite"

def try_convert_select_f16(model):
    try:
        c = tf.lite.TFLiteConverter.from_keras_model(model)
        c.optimizations = [tf.lite.Optimize.DEFAULT]
        c.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS, tf.lite.OpsSet.SELECT_TF_OPS]
        # leave tensor-list ops as SELECT_TF_OPS
        c._experimental_lower_tensor_list_ops = False
        c.target_spec.supported_types = [tf.float16]
        tflite_model = c.convert()
        with open(OUT_TFLITE_F16, "wb") as f:
            f.write(tflite_model)
        print("✅ SELECT_TF_OPS (float16) conversion succeeded ->", OUT_TFLITE_F16)
        return True
    except Exception as e:
        print("❌ SELECT_TF_OPS (float16) failed:", e)
        traceback.print_exc()
        return False

def try_convert_select_default(model):
    try:
        c = tf.lite.TFLiteConverter.from_keras_model(model)
        c.optimizations = [tf.lite.Optimize.DEFAULT]
        c.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS, tf.lite.OpsSet.SELECT_TF_OPS]
        c._experimental_lower_tensor_list_ops = False
        tflite_model = c.convert()
        with open(OUT_TFLITE_DEFAULT, "wb") as f:
            f.write(tflite_model)
        print("✅ SELECT_TF_OPS (default) conversion succeeded ->", OUT_TFLITE_DEFAULT)
        return True
    except Exception as e:
        print("❌ SELECT_TF_OPS (default) failed:", e)
        traceback.print_exc()
        return False

def try_simple_convert(model):
    try:
        c = tf.lite.TFLiteConverter.from_keras_model(model)
        tflite_model = c.convert()
        with open(OUT_TFLITE_SIMPLE, "wb") as f:
            f.write(tflite_model)
        print("✅ Simple conversion succeeded ->", OUT_TFLITE_SIMPLE)
        return True
    except Exception as e:
        print("❌ Simple conversion failed:", e)
        traceback.print_exc()
        return False

def main():
    if not os.path.exists(KERAS_PATH):
        print("Keras model not found at path:", KERAS_PATH)
        return
    print("Loading model:", KERAS_PATH)
    model = tf.keras.models.load_model(KERAS_PATH, compile=False)
    print(model.summary())

    print("\nAttempt 1: SELECT_TF_OPS (float16) with _experimental_lower_tensor_list_ops=False")
    if try_convert_select_f16(model):
        print("\n*** NOTE: This file contains SELECT_TF_OPS and WILL NOT run on TFLite Micro (ESP32). ***")
        return

    print("\nAttempt 2: SELECT_TF_OPS (default)")
    if try_convert_select_default(model):
        print("\n*** NOTE: This file contains SELECT_TF_OPS and WILL NOT run on TFLite Micro (ESP32). ***")
        return

    print("\nAttempt 3: Simple conversion (no select ops)")
    if try_simple_convert(model):
        print("\nSimple conversion succeeded — this TFLite is the one to try on ESP32/TFLite-Micro.")
        return

    print("\nAll conversion attempts failed. See tracebacks above. Proceed to model modifications (unroll/replace).")

if __name__ == "__main__":
    main()
