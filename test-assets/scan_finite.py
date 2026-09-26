# -*- coding: utf-8 -*-
"""全量扫描 safetensors 数值有限性 + header 对齐检查。"""
import json
import struct
import sys

path = sys.argv[1] if len(sys.argv) > 1 else "test-assets/bs_roformer_1296.safetensors"
with open(path, "rb") as f:
    d = f.read()

hl = struct.unpack("<Q", d[:8])[0]
print(f"header_len={hl} 对齐={(8 + hl) % 8 == 0}")
h = json.loads(d[8 : 8 + hl])
p = 8 + hl
bad = 0
total = 0
for n, m in h.items():
    s, e = m["data_offsets"]
    raw = d[p + s : p + e]
    v = struct.unpack(f"<{len(raw)//4}f", raw)
    total += 1
    if any(x != x for x in v) or any(x in (float("inf"), float("-inf")) for x in v):
        bad += 1
        print("非有限:", n)
print(f"tensor 总数 {total}, 非有限 {bad}")
