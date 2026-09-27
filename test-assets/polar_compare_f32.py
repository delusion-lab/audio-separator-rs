import numpy as np

base = r"D:\alice\audio-separator-rs\test-assets"

def corr(a, b):
    return float(np.corrcoef(a.ravel(), b.ravel())[0, 1])

for name in ["time", "freq"]:
    print(f"--- {name} (candle vs torch-f32) ---")
    for i in range(12):
        t = np.load(rf"{base}\torch_real_layer{i}_{name}_f32.npy")
        # torch: time=[62,44,256](band,frame)  freq=[44,62,256](frame,band)
        raw = np.fromfile(rf"{base}\candle_real_layer{i}_{name}.npy", dtype=np.float32)
        # candle 文件布局 [1, bands=62, frames=44, 256]
        c = raw.reshape(1, 62, 44, 256)[0]      # [62, 44, 256]
        if name == "freq":
            c = c.transpose(1, 0, 2)            # [44, 62, 256]
        print(f"layer{i}: corr={corr(t, c):.4f}")

tx = np.load(rf"{base}\torch_real_x_f32.npy")
cx = np.fromfile(rf"{base}\candle_real_x.npy", dtype=np.float32).reshape(tx.shape)
print("x corr:", corr(tx, cx))
tm = np.load(rf"{base}\torch_real_mask_f32.npy")[0, :, :8]
cm = np.fromfile(rf"{base}\candle_real_mask0.npy", dtype=np.float32).reshape(44, 8)
print("mask0 corr:", corr(tm, cm))
tv = np.load(rf"{base}\torch_real_vocals_f32.npy")
cv = np.fromfile(rf"{base}\candle_real_vocals.npy", dtype=np.float32).reshape(2, tv.shape[1])
for ch in range(2):
    print(f"vocals ch{ch}: corr={corr(tv[ch], cv[ch]):.4f} torch_e={np.sum(tv[ch]**2):.3e} candle_e={np.sum(cv[ch]**2):.3e}")
