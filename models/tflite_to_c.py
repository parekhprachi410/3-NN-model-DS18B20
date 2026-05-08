#!/usr/bin/env python3
# tflite_to_c.py
# Usage: python tflite_to_c.py model1.tflite model2.tflite ...

import sys
import os

PROGEM_ATTR = ""  # change to "PROGMEM" if targeting AVR; ESP32 doesn't need it
LINE_BYTES = 12   # how many bytes per line in the array

def sanitize_name(fname):
    name = os.path.splitext(os.path.basename(fname))[0]
    # make a C-safe identifier
    name = name.replace('-', '_').replace('.', '_')
    return name

def write_header(input_path, out_dir=None):
    with open(input_path, "rb") as f:
        data = f.read()
    name = sanitize_name(input_path)
    array_name = f"{name}_tflite"
    len_name = f"{array_name}_length"
    header_name = (out_dir + os.sep + name + ".h") if out_dir else (name + ".h")
    with open(header_name, "w") as h:
        h.write(f"// Auto-generated from {os.path.basename(input_path)}\n")
        h.write(f"#ifndef {name.upper()}_H\n")
        h.write(f"#define {name.upper()}_H\n\n")
        h.write("#include <stdint.h>\n\n")
        h.write(f"const unsigned char {array_name}[] = {{\n")
        for i in range(0, len(data), LINE_BYTES):
            chunk = data[i:i+LINE_BYTES]
            line = ", ".join(f"0x{b:02x}" for b in chunk)
            if i + LINE_BYTES < len(data):
                h.write(f"  {line},\n")
            else:
                h.write(f"  {line}\n")
        h.write("};\n\n")
        h.write(f"const unsigned int {len_name} = {len(data)}u;\n\n")
        h.write(f"#endif // {name.upper()}_H\n")
    print(f"Wrote {header_name} ({len(data)} bytes)")

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python tflite_to_c.py model1.tflite model2.tflite ...")
        sys.exit(1)
    for path in sys.argv[1:]:
        if not os.path.exists(path):
            print("File not found:", path)
            continue
        write_header(path)
