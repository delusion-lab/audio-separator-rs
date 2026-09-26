# -*- coding: utf-8 -*-
"""检查 bs_roformer safetensors 产物中 band_split.to_features.*.0.gamma 的数值特征。
只读分析：列出每个 band gamma 的 min/max/NaN 数，判定是转换器错读还是模型本身异常。
"""
import json
import struct
import sys

path = sys.argv[1] if len(sys.argv) > 1 else "test-assets/bs_roformer_1296.safetensors"

with open(path, "rb") as f:
    data = f.read()

header_len = struct.unpack("<Q", data[:8])[0]
header = json.loads(data[8 : 8 + header_len])
payload = data[8 + header_len :]

keys = [k for k in header if k.startswith("band_split.to_features.") and k.endswith(".0.gamma")]
keys.sort(key=lambda k: int(k.split(".")[3]))

bad = 0
for k in keys:
    meta = header[k]
    offs = meta["data_offsets"]
    raw = payload[offs[0] : offs[1]]
    vals = struct.unpack(f"<{len(raw)//4}f", raw)
    n_nan = sum(1 for v in vals if v != v)
    n_inf = sum(1 for v in vals if v in (float("inf"), float("-inf")))
    if n_nan or n_inf:
        bad += 1
        print(f"{k}: shape={meta['shape']} nan={n_nan} inf={n_inf} min={min(vals):.6g} max={max(vals):.6g}")
        # 打印前 8 个值
        print(f"  first8: {[round(v, 4) for v in vals[:8]]}")

print(f"检查 {len(keys)} 个 gamma，异常 {bad} 个")
