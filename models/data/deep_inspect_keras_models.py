import sys
import tensorflow as tf
from pathlib import Path
from contextlib import redirect_stdout

def inspect_layer(layer):
    info = {}
    info["name"] = layer.name
    info["type"] = layer.__class__.__name__
    info["output_shape"] = layer.output_shape if hasattr(layer, "output_shape") else None
    info["activation"] = getattr(layer, "activation", None)
    if info["activation"]:
        info["activation"] = info["activation"].__name__

    config = layer.get_config()
    info["config"] = config

    weights = layer.get_weights()
    info["weights"] = weights
    return info


def print_layer_details(layer_info):
    print(f"\n--- Layer: {layer_info['name']} ({layer_info['type']}) ---")
    if layer_info["output_shape"]:
        print(f"Output shape: {layer_info['output_shape']}")
    if layer_info["activation"]:
        print(f"Activation: {layer_info['activation']}")

    cfg = layer_info["config"]
    if layer_info["type"] == "Dense":
        units = cfg.get("units", "N/A")
        print(f"Units: {units}")
    elif layer_info["type"] == "LSTM":
        units = cfg.get("units", "N/A")
        return_sequences = cfg.get("return_sequences", False)
        print(f"Units: {units}")
        print(f"Return sequences: {return_sequences}")
        print("Gate order: [input, forget, cell, output]")
    elif layer_info["type"] == "Conv1D":
        filters = cfg.get("filters", "N/A")
        kernel_size = cfg.get("kernel_size", "N/A")
        strides = cfg.get("strides", "N/A")
        padding = cfg.get("padding", "N/A")
        print(f"Filters: {filters}, Kernel size: {kernel_size}, Strides: {strides}, Padding: {padding}")

    weights = layer_info["weights"]
    if weights:
        print("Weights:")
        for i, w in enumerate(weights):
            print(f"  W{i}: shape={w.shape}, dtype={w.dtype}, total={w.size}")


def print_model_summary(model, model_name):
    print(f"\n{'='*90}")
    print(f"MODEL: {model_name}")
    print(f"{'='*90}\n")
    model.summary(line_length=110, expand_nested=True, show_trainable=True)

    print(f"\nModel input shape: {model.input_shape}")
    print(f"Model output shape: {model.output_shape}")
    print(f"Total layers: {len(model.layers)}")


def inspect_model(filepath):
    path = Path(filepath)
    if not path.exists():
        print(f"[!] File not found: {path}")
        return

    try:
        model = tf.keras.models.load_model(path)
        model_name = path.stem

        # Console header
        print_model_summary(model, model_name)

        all_layers = [inspect_layer(layer) for layer in model.layers]

        print(f"\n{'-'*60}\nDETAILED LAYER INFORMATION\n{'-'*60}")
        for li in all_layers:
            print_layer_details(li)

        # Save all info to file
        out_file = path.with_suffix(".detailed_info.txt")
        with open(out_file, "w", encoding="utf-8") as f:
            with redirect_stdout(f):
                print_model_summary(model, model_name)
                print(f"\n{'-'*60}\nDETAILED LAYER INFORMATION\n{'-'*60}")
                for li in all_layers:
                    print_layer_details(li)

        print(f"\n✅ Saved detailed inspection → {out_file.name}")

    except Exception as e:
        print(f"[!] Error inspecting {path}: {e}")


def main():
    if len(sys.argv) < 2:
        print("Usage: python deep_inspect_keras_models.py <model1.keras> <model2.keras> ...")
        return
    for model_path in sys.argv[1:]:
        inspect_model(model_path)


if __name__ == "__main__":
    main()
