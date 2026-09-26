# -*- coding: utf-8 -*-
"""检查 safetensors header 中 data_offsets 是否连续、是否有重叠/空洞，
定位 band gamma 异常根因（对齐 vs 其他）。"""
import json
import struct
import sys

path = sys.argv[1] if len(sys.argv) > 1 else "test-assets/bs_roformer_1296.safetensors"
with open(path, "rb") as f:
    data = f.read()

header_len = struct.unpack("<Q", data[:8])[0]
print(f"header_len 字段 = {header_len} (0x{header_len:x}), 文件总长 = {len(data)}")
print(f"8+header_len = {8 + header_len}, 是否 8 对齐: {(8 + header_len) % 8 == 0}")
header = json.loads(data[8 : 8 + header_len])
payload_start = 8 + header_len
print(f"payload 起点 = {payload_start}, 数据区总长 = {len(data) - payload_start}")

items = [(name, meta["data_offsets"]) for name, meta in header.items()]
# 按 start 排序检查连续性
items.sort(key=lambda x: x[1][0])
prev_end = payload_start  # 相对文件的预期数据起点
holes = 0
overlaps = 0
for name, (s, e) in items:
    abs_s, abs_e = payload_start + s, payload_start + e
    if abs_s < prev_end:
        overlaps += 1
        print(f"重叠: {name} [{abs_s},{abs_e}) 上一个结束 {prev_end}")
    elif abs_s > prev_end:
        holes += 1
        print(f"空洞: {name} [{abs_s},{abs_e}) 上一个结束 {prev_end}")
    prev_end = max(prev_end, abs_e)

# 目标 tensor 的 offsets 与邻居
target = "band_split.to_features.10.0.gamma"
meta = header[target]
s, e = meta["data_offsets"]
print(f"\n{target}: offsets [{s},{e}) 长度 {e - s} shape {meta['shape']}")
# 找前后邻居
names = sorted(header.keys())
idx = names.index(target)
for n in names[max(0, idx - 2) : idx + 3]:
    o = header[n]["data_offsets"]
    print(f"  {n}: [{o[0]},{o[1]}) len={o[1]-o[0]}")
print(f"header 键数 = {len(names)}")
