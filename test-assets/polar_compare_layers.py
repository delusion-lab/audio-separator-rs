import numpy as np

base = r"D:\alice\audio-separator-rs\test-assets"
# torch time: [62(band), 44(frame), 256]；candle time 后: [1, 44, 62, 256] → [44, 62, 256] 需转 [62, 44, 256]
for i in range(12):
    t = np.load(rf"{base}\torch_real_layer{i}_time.npy")  # [62, 44, 256]
    c = np.fromfile(rf"{base}\candle_real_layer{i}_time.npy", dtype=np.float32).reshape(1, 44, 62, 256)
    c = c.transpose(0, 3, 1, 2)[0]  # [256? no]  -> 目标 [62, 44, 256]: candle [1,44,62,256] -> swap frame/band -> [1,62,44,256]
    c = c[0].transpose(1, 0, 2) if False else None
# 直接用正确的转置
results = []
for i in range(12):
    t = np.load(rf"{base}\torch_real_layer{i}_time.npy")          # [62, 44, 256]
    c = np.fromfile(rf"{base}\candle_real_layer{i}_time.npy", dtype=np.float32).reshape(1, 44, 62, 256)
    c = c.transpose(0, 2, 1, 3)[0]                                 # [62, 44, 256]
    corr = float(np.corrcoef(t.ravel(), c.ravel())[0, 1])
    et = float(np.sum(t**2)); ec = float(np.sum(c**2))
    results.append((i, "time", corr, et, ec))
    tf = np.load(rf"{base}\torch_real_layer{i}_freq.npy")          # [44, 62, 256]
    cf = np.fromfile(rf"{base}\candle_real_layer{i}_freq.npy", dtype=np.float32).reshape(1, 44, 62, 256)[0]  # [44, 62, 256]
    corr = float(np.corrcoef(tf.ravel(), cf.ravel())[0, 1])
    et = float(np.sum(tf**2)); ec = float(np.sum(cf**2))
    results.append((i, "freq", corr, et, ec))
for i, n, corr, et, ec in results:
    print(f"layer{i}_{n}: corr={corr:.4f} torch_e={et:.3e} candle_e={ec:.3e}")
