import numpy as np

# torch [2, 22016] vs candle [22016]（ch0）
t = np.load(r"D:\alice\audio-separator-rs\test-assets\torch_real_vocals.npy")  # [2, 22016]
c = np.fromfile(r"D:\alice\audio-separator-rs\test-assets\candle_real_vocals.npy", dtype=np.float32)
print("shapes:", t.shape, c.shape)
n = t.shape[1]
c = c.reshape(2, n)  # [ch, t]
for ch in range(2):
    a = t[ch]
    print(f"ch{ch}: corr={np.corrcoef(a, c[ch])[0,1]:.4f} torch_e={np.sum(a**2):.3e} candle_e={np.sum(c[ch]**2):.3e}")

# band_split 对比（真实歌曲）
tb = np.load(r"D:\alice\audio-separator-rs\test-assets\torch_real_band.npy")[0]  # [44, 62, 256]
cb = np.fromfile(r"D:\alice\audio-separator-rs\test-assets\candle_real_band.npy", dtype=np.float32).reshape(62, 44, 256)
self_corr = []
for i in range(62):
    self_corr.append(float(np.corrcoef(tb[:, i, :].ravel(), cb[i].ravel())[0, 1]))
self_corr = np.array(self_corr)
bad = np.where(self_corr < 0.99)[0]
print("real band self corr min/mean:", self_corr.min(), self_corr.mean())
print("real band corr<0.99:", bad.tolist())
