import numpy as np

# x: torch [1, 44, 62, 256] vs candle [1, 44, 62, 256]
tx = np.load(r"D:\alice\audio-separator-rs\test-assets\torch_real_x.npy")  # [1,44,62,256]
cx = np.fromfile(r"D:\alice\audio-separator-rs\test-assets\candle_real_x.npy", dtype=np.float32).reshape(1, 44, 62, 256)
a = tx[0].reshape(-1); b = cx[0].reshape(-1)
print(f"x overall: corr={np.corrcoef(a, b)[0,1]:.4f} torch_e={np.sum(a**2):.3e} candle_e={np.sum(b**2):.3e}")

# 逐 band
band_corr = []
for i in range(62):
    av = tx[0, :, i, :].ravel(); bv = cx[0, :, i, :].ravel()
    band_corr.append(float(np.corrcoef(av, bv)[0, 1]))
band_corr = np.array(band_corr)
bad = np.where(band_corr < 0.99)[0]
print("band corr min/mean:", band_corr.min(), band_corr.mean(), "bad(<0.99):", bad.tolist())

# mask0: torch [1,44,4100] band0 段 = 前 8 列；candle [44,8]
tm = np.load(r"D:\alice\audio-separator-rs\test-assets\torch_real_mask.npy")[0, :, :8]  # [44,8]
cm = np.fromfile(r"D:\alice\audio-separator-rs\test-assets\candle_real_mask0.npy", dtype=np.float32).reshape(44, 8)
print(f"mask0: corr={np.corrcoef(tm.ravel(), cm.ravel())[0,1]:.4f} torch_e={np.sum(tm**2):.3e} candle_e={np.sum(cm**2):.3e}")
