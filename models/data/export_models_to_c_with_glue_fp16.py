#!/usr/bin/env python3
"""
export_models_to_c_with_glue_fp16.py

Exports Keras models' Dense/LSTM/Conv1D weights as float16 bit patterns (uint16 arrays),
writes metadata JSON and generates simple glue .h/.cpp files.

Usage:
    python export_models_to_c_with_glue_fp16.py LSTM.keras FNN.keras TRN.keras
"""
import sys, os, json, struct
from pathlib import Path
import numpy as np
import tensorflow as tf

OUT_DIR = Path("c_models")
OUT_DIR.mkdir(exist_ok=True)

def float_array_to_uint16_bits(arr: np.ndarray):
    # convert to float16 then to uint16 bit patterns
    a16 = arr.astype(np.float16).flatten()
    u16 = a16.view(np.uint16).tolist()
    return u16

def write_c_array_uint16(name, u16list, outfh):
    outfh.write(f"// {name}\n")
    outfh.write(f"const unsigned short {name}_data[] = {{\n")
    line = []
    for i,u in enumerate(u16list):
        line.append(f"0x{u:04x}u")
        if (i+1) % 16 == 0:
            outfh.write("  " + ", ".join(line) + ",\n")
            line = []
    if line:
        outfh.write("  " + ", ".join(line) + "\n")
    outfh.write("};\n")
    outfh.write(f"const unsigned int {name}_len = {len(u16list)}u;\n\n")

def export_model(path):
    p = Path(path)
    print(f"[+] Loading {p}")
    model = tf.keras.models.load_model(str(p), compile=False)
    mname = p.stem
    header_path = OUT_DIR / f"{mname}.h"
    meta = {"model": mname, "layers": []}

    with open(header_path, "w", encoding="utf-8") as h:
        h.write(f"// Auto-generated FP16 weights header for model {mname}\n")
        h.write("#ifndef {0}_H\n#define {0}_H\n\n#include <stdint.h>\n\n".format(mname.upper()))
        for idx, layer in enumerate(model.layers):
            cls = layer.__class__.__name__
            lname = layer.name
            print(f"  - Layer {idx}: {lname} ({cls})")
            layer_meta = {"index": idx, "name": lname, "class": cls}
            try:
                cfg = layer.get_config()
            except Exception:
                cfg = {}
            layer_meta["config"] = cfg

            if cls == "Dense":
                w_b = layer.get_weights()
                if not w_b:
                    print("    ! Dense has no weights; skipping")
                    continue
                w = np.array(w_b[0])
                b = np.array(w_b[1]) if len(w_b) > 1 else None
                arrname_k = f"{mname}_{lname}_kernel"
                write_c_array_uint16(arrname_k, float_array_to_uint16_bits(w), h)
                layer_meta["kernel_var"] = arrname_k + "_data"
                layer_meta["kernel_shape"] = list(w.shape)
                if b is not None and b.size>0:
                    arrname_b = f"{mname}_{lname}_bias"
                    write_c_array_uint16(arrname_b, float_array_to_uint16_bits(b), h)
                    layer_meta["bias_var"] = arrname_b + "_data"
                    layer_meta["bias_shape"] = list(b.shape)
            elif cls == "LSTM":
                wts = layer.get_weights()
                if len(wts) < 2:
                    print("    ! Unexpected LSTM weights; skipping")
                    continue
                kernel = np.array(wts[0]); recurrent = np.array(wts[1])
                bias = np.array(wts[2]) if len(wts) > 2 else np.zeros((kernel.shape[1],), dtype=np.float32)
                name_k = f"{mname}_{lname}_kernel"
                name_r = f"{mname}_{lname}_recurrent"
                name_b = f"{mname}_{lname}_bias"
                write_c_array_uint16(name_k, float_array_to_uint16_bits(kernel), h)
                write_c_array_uint16(name_r, float_array_to_uint16_bits(recurrent), h)
                write_c_array_uint16(name_b, float_array_to_uint16_bits(bias), h)
                layer_meta["kernel_var"] = name_k + "_data"
                layer_meta["recurrent_var"] = name_r + "_data"
                layer_meta["bias_var"] = name_b + "_data"
                layer_meta["kernel_shape"] = list(kernel.shape)
                layer_meta["recurrent_shape"] = list(recurrent.shape)
                layer_meta["bias_shape"] = list(bias.shape)
            elif cls == "Conv1D":
                wts = layer.get_weights()
                if len(wts) == 0:
                    print("    ! Conv1D has no weights; skipping")
                    continue
                kernel = np.array(wts[0])
                bias = np.array(wts[1]) if len(wts) > 1 else None
                name_k = f"{mname}_{lname}_kernel"
                write_c_array_uint16(name_k, float_array_to_uint16_bits(kernel), h)
                layer_meta["kernel_var"] = name_k + "_data"
                layer_meta["kernel_shape"] = list(kernel.shape)
                if bias is not None and bias.size>0:
                    name_b = f"{mname}_{lname}_bias"
                    write_c_array_uint16(name_b, float_array_to_uint16_bits(bias), h)
                    layer_meta["bias_var"] = name_b + "_data"
                    layer_meta["bias_shape"] = list(bias.shape)
            else:
                print(f"    - Skipping unsupported layer type: {cls}")
                layer_meta["skipped"] = True

            meta["layers"].append(layer_meta)

        h.write("#endif\n")
    meta_path = OUT_DIR / f"{mname}_meta.json"
    with open(meta_path, "w", encoding="utf-8") as jf:
        json.dump(meta, jf, indent=2)
    print(f"  -> Wrote {header_path} and {meta_path}")

    # For simplicity, we won't auto-generate glue in this FP16 exporter.
    # You can reuse the previous glue generator or ask me to generate glue that uses FP16 tables.
    print(f"  (Note: glue files not auto-generated by this FP16 exporter. I can generate glue using these var names if you want.)")

def main():
    if len(sys.argv) < 2:
        print("Usage: python export_models_to_c_with_glue_fp16.py model1.keras model2.keras ...")
        return
    for path in sys.argv[1:]:
        export_model(path)
    print("[*] Done. Files in c_models/")

if __name__ == "__main__":
    main()
