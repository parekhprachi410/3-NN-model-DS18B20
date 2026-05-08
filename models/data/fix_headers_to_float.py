#!/usr/bin/env python3
"""
fix_headers_to_float.py

- Backs up every .h in c_export/ to c_export_backup/
- Replaces occurrences like:
    static const uint16_t name_data[] = { <float literals> };
  with:
    static const float name_data[] = { <float literals> };
- Leaves *_shape arrays unchanged.

Run from sketch folder where `c_export/` is located:
    python fix_headers_to_float.py
"""
import os, shutil, re
from pathlib import Path

C_EXPORT = Path("c_export")
BACKUP = Path("c_export_backup")
BACKUP.mkdir(exist_ok=True)

if not C_EXPORT.exists():
    print("c_export/ not found - please run from sketch folder.")
    raise SystemExit(1)

hdrs = list(C_EXPORT.glob("*.h"))
if not hdrs:
    print("No .h files in c_export/")
    raise SystemExit(1)

pat = re.compile(r"\b(static\s+const\s+)uint16_t(\s+)([A-Za-z0-9_]+_data\s*\[\]\s*=\s*\{)", flags=re.I)

for p in hdrs:
    print("Backing up:", p)
    shutil.copy2(p, BACKUP / p.name)

    text = p.read_text(encoding="utf-8")
    # Replace uint16 declaration with float
    new_text, n = pat.subn(r"\1float\2\3", text)
    if n == 0:
        # maybe other variant: 'const unsigned short' etc
        new_text = text.replace("const unsigned short", "const float")
        new_text = new_text.replace("uint16_t", "float")
    p.write_text(new_text, encoding="utf-8")
    print(f"Patched {p.name} (replaced {n} declarations if any)")

print("Done. Backups saved to c_export_backup/")
