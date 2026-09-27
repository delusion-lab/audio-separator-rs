import json, sys, struct

path = sys.argv[1]
with open(path, "rb") as f:
    n = struct.unpack("<Q", f.read(8))[0]
    header = json.loads(f.read(n))
    keys = list(header.keys())
    keys.remove("__metadata__") if "__metadata__" in keys else None
    print("total tensors:", len(keys))
    # key pattern counts
    import collections
    pats = collections.Counter()
    for k in keys:
        if k.startswith("band_split."):
            pats["band_split"] += 1
        elif k.startswith("layers."):
            # layers.{i}.{0=time,1=freq}.{sub}
            parts = k.split(".")
            # e.g. layers.0.0.layers.0.norm.gamma / layers.0.0.layers.0.rotary_embed.freqs
            sub = parts[3]
            idx = parts[1] + "." + parts[2]
            if sub == "layers":
                pats["layers." + idx + ".layers." + parts[4]] += 1
            elif sub == "norm":
                pats["layers." + idx + ".norm"] += 1
        elif k.startswith("mask_estimators."):
            pats["mask_estimators"] += 1
        elif "final_norm" in k:
            pats["final_norm"] += 1
        elif "rotary_embed" in k:
            pats["rotary_embed.freqs"] += 1
        else:
            pats["OTHER:" + k] += 1
    for p, c in sorted(pats.items()):
        print(f"  {p}: {c}")
    # sample keys
    print("sample keys:")
    for k in keys[:8]:
        print("  ", k)
    print("  ...")
    for k in keys[-8:]:
        print("  ", k)
    # dims of band_split inputs
    for k in keys:
        if k.startswith("band_split.to_features.") and k.endswith(".weight"):
            info = header[k]
            print("band_split", k, info.get("shape"), info.get("dtype"))
    # check mask estimator linear layers count
    mk = [k for k in keys if k.startswith("mask_estimators.0.to_freqs.0.0.")]
    print("mask_estimators.0.to_freqs.0.0.* :", mk)
    # check for final_norm
    print("has final_norm:", any("final_norm" in k for k in keys))
