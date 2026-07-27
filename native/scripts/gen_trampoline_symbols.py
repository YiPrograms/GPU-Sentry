#!/usr/bin/env python3

import re
import subprocess
import sys
from pathlib import Path

if len(sys.argv) != 2:
    print(f"Usage: {sys.argv[0]} [Path to libcuda.so.1]", file=sys.stderr)
    sys.exit(1)

real_libcuda = sys.argv[1]
out_symbols = "include/cuda_symbols.def"
out_trampolines = "include/cuda_trampolines.inc"

out = subprocess.check_output(
    ["readelf", "-Ws", "--wide", real_libcuda],
    text=True,
)

symbols = set()
hooked_symbols = set(Path("hooked_symbols").read_text().splitlines())

for line in out.splitlines():
    parts = line.split()
    if len(parts) < 8:
        continue

    sym_type = parts[3]
    ndx = parts[6]
    name = parts[7]

    if sym_type != "FUNC":
        continue
    if ndx == "UND":
        continue

    name = name.split("@", 1)[0]

    if not name.startswith("cu"):
        continue

    if not re.match(r"^[A-Za-z_][A-Za-z0-9_]*$", name):
        continue

    symbols.add(name)

symbols = sorted(symbols)

Path(out_symbols).write_text(
    "".join(f"CUDA_SYMBOL({s})\n" for s in symbols)
)

Path(out_trampolines).write_text(
    # Only generate trampolines for symbols that are not hooked.
    "".join(f"CUDA_TRAMP {s}\n" for s in symbols if s not in hooked_symbols)
)

print(f"Generated for {len(symbols)} CUDA symbols")
