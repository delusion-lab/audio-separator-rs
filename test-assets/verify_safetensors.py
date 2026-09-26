"""独立验证：手工解析 safetensors 文件（不依赖库），核对 header/形状/数值 sanity。
safetensors 格式：8 字节 LE header 长度 + JSON header + 8 字节对齐数据区。"""
import json, struct, sys

path = r"D:\alice\audio-separator-rs\test-assets\bs_roformer_1296.safetensors"
with open(path, "rb") as f:
    data = f.read()

hdr_len = struct.unpack("<Q", data[:8])[0]
hdr = json.loads(data[8:8 + hdr_len])
n = len(hdr)
print(f"文件大小: {len(data)/1e6:.1f} MB, header 长度 {hdr_len}, tensor 数 {n}")

names = sorted(hdr.keys())
print(f"\n前 12 个 tensor 名:")
for k in names[:12]:
    v = hdr[k]
    print(f"  {k}: {v['dtype']} {v['shape']} off={v['data_offsets']}")

# dtype 统计
from collections import Counter
dtypes = Counter(v["dtype"] for v in hdr.values())
print(f"\ndtype 统计: {dict(dtypes)}")

# 形状统计（按维数）
dims = Counter(len(v["shape"]) for v in hdr.values())
print(f"维数统计: {dict(dims)}")

# 总元素数
import functools
total_elems = sum(functools.reduce(lambda x, y: x * y, v["shape"], 1) for v in hdr.values())
print(f"总元素数: {total_elems:,}")

# 数值 sanity：抽样检查若干 f32 tensor（min/max/mean/std，NaN 检查）
import random, statistics
random.seed(42)
checked = 0
bad = 0
sample_names = random.sample(names, min(200, n))
for k in sample_names:
    v = hdr[k]
    if v["dtype"] != "F32":
        continue
    s, e = v["data_offsets"]
    blk = data[s:e]
    assert (e - s) % 4 == 0, f"{k} 长度非 4 对齐"
    vals = struct.unpack(f"<{ (e-s)//4 }f", blk)
    if any(x != x for x in vals):
        print(f"  !! {k} 含 NaN")
        bad += 1
        continue
    mn, mx = min(vals), max(vals)
    mean = sum(vals) / len(vals)
    var = sum((x-mean)**2 for x in vals) / len(vals)
    if checked < 8:
        print(f"  {k}: shape={v['shape']} min={mn:.4f} max={mx:.4f} mean={mean:.4f} std={var**0.5:.4f}")
    checked += 1
print(f"\n数值检查: {checked} 个 f32 tensor, NaN 数 {bad}")

# 关键结构权重检查（bs_roformer 1296）
print("\n关键权重形状:")
for key in ["layers.0.0.layers.0.0.rotary_embed.freqs",
            "layers.0.0.layers.0.0.norm.gamma",
            "layers.0.0.layers.0.0.to_qkv.weight",
            "layers.0.0.layers.0.0.to_out.weight",
            "layers.0.0.layers.0.0.layers.0.0.norm.gamma"]:
    if key in hdr:
        print(f"  {key}: {hdr[key]['dtype']} {hdr[key]['shape']}")
    else:
        print(f"  {key}: 不存在")
