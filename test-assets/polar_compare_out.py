import numpy as np

t = np.load(r"D:\alice\audio-separator-rs\test-assets\torch_polar_out.npy")  # [2, 22016]
c = np.fromfile(r"D:\alice\audio-separator-rs\test-assets\candle_polar_out.npy", dtype=np.float32).reshape(2, 22016)

for ch, name in [(0, "L"), (1, "R")]:
    a, b = t[ch], c[ch]
    e_a = float((a ** 2).sum())
    e_b = float((b ** 2).sum())
    corr = float(np.corrcoef(a, b)[0, 1])
    print(f"ch{name}: torch_energy={e_a:.3e} candle_energy={e_b:.3e} ratio={e_b/e_a:.3f} corr={corr:.4f}")
    # 增益对齐后相关性
    if e_a > 0:
        scale = np.sqrt(e_a / e_b)
        corr_s = float(np.corrcoef(a, b * scale)[0, 1])
        print(f"  gain-aligned corr={corr_s:.4f} scale={scale:.4f}")
