"""安全分析 bs_roformer ckpt：zip 结构 + pickle opcode 流（不执行 pickle，纯结构分析）。
用标准库 zipfile + pickletools 只读检查，不反序列化任意对象。"""
import zipfile, pickletools, sys

path = r"C:\Users\alice\AppData\Local\asep-ort\model_bs_roformer_ep_368_sdr_12.9628.ckpt"
zf = zipfile.ZipFile(path)
names = zf.namelist()
print(f"zip 条目数: {len(names)}")
for n in names[:12]:
    info = zf.getinfo(n)
    print(f"  {n}  ({info.file_size} bytes)")

# 读取 data.pkl 并 disassemble（不执行）
pkl = zf.read("last_bs_roformer/data.pkl")
print(f"\ndata.pkl 大小: {len(pkl)} bytes")
ops = list(pickletools.genops(pkl))
print(f"opcode 数: {len(ops)}")
# 打印前 60 个 opcode（截断长参数）
from collections import Counter
op_count = Counter(op.name for op, arg, pos in ops)
print("opcode 统计:", dict(op_count))
print("\n前 80 个 opcode 流:")
for i, (op, arg, pos) in enumerate(ops[:80]):
    arg_s = repr(arg)
    if len(arg_s) > 60:
        arg_s = arg_s[:60] + "..."
    print(f"  {i:3d} {op.name:24s} {arg_s}")
