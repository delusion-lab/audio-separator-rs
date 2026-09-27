import numpy as np

# torch band_split: [1, 44, 62, 256]；candle band0/61: [1, 1, 44, 256] → [44, 256]
t = np.load(r"D:\alice\audio-separator-rs\test-assets\torch_polar_band.npy")
c0 = np.fromfile(r"D:\alice\audio-separator-rs\test-assets\candle_polar_band0.npy", dtype=np.float32).reshape(44, 256)
c61 = np.fromfile(r"D:\alice\audio-separator-rs\test-assets\candle_polar_band61.npy", dtype=np.float32).reshape(44, 256)

for band, c in [(0, c0), (61, c61)]:
    a = t[0, :, band, :]  # [44, 256]
    corr = float(np.corrcoef(a.ravel(), c.ravel())[0, 1])
    e_a = float((a**2).sum()); e_c = float((c**2).sum())
    # 逐帧相关性均值
    per_frame = [float(np.corrcoef(a[i], c[i])[0, 1]) for i in range(44)]
    print(f"band{band}: corr={corr:.4f} torch_energy={e_a:.3e} candle_energy={e_c:.3e} ratio={e_c/e_a:.3f} per_frame_mean={np.mean(per_frame):.4f} min={np.min(per_frame):.4f}")
