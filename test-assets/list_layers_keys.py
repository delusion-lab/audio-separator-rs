import json, sys, struct

path = sys.argv[1]
with open(path, "rb") as f:
    n = struct.unpack("<Q", f.read(8))[0]
    header = json.loads(f.read(n))
keys = list(header.keys())
for k in keys:
    if k.startswith("layers.0.0."):
        print(k)
