import numpy as np

t = np.load(r"D:\alice\audio-separator-rs\test-assets\torch_polar_band_noise.npy")[0]  # [44, 62, 256]
c = np.fromfile(r"D:\alice\audio-separator-rs\test-assets\candle_polar_all_bands_noise.npy", dtype=np.float32).reshape(62, 44, 256)

row_self = []
for i in range(62):
    corrs = [float(np.corrcoef(t[:, i, :].ravel(), c[j].ravel())[0, 1]) for j in range(62)]
    row_self.append(corrs[i])
row_self = np.array(row_self)
print("noise 版 self corr 低 (<0.99) 的 torch band:", np.where(row_self < 0.99)[0].tolist())
print("noise 版 self corr min/mean:", row_self.min(), row_self.mean())

# 对比无噪声 vs 噪声的 band61
t0 = np.load(r"D:\alice\audio-separator-rs\test-assets\torch_polar_band.npy")[0]
c0 = np.fromfile(r"D:\alice\audio-separator-rs\test-assets\candle_polar_all_bands.npy", dtype=np.float32).reshape(62, 44, 256)
for name, a, b in [("band29", t[:, 29, :], c[29]), ("band60", t[:, 60, :], c[60]), ("band61", t[:, 61, :], c[61])]:
    print(f"noise {name}: corr={np.corrcoef(a.ravel(), b.ravel())[0,1]:.4f}")
