import numpy as np

t = np.load(r"D:\alice\audio-separator-rs\test-assets\torch_mel_mix_out.npy")  # [2, 22050]
c = np.fromfile(r"D:\alice\audio-separator-rs\test-assets\candle_mel_mix_out.npy", dtype=np.float32)  # [2*22050]
c = c.reshape(2, 22050)

for ch, name in [(0, "L"), (1, "R")]:
    a, b = t[ch], c[ch]
    e_a = float((a ** 2).sum())
    e_b = float((b ** 2).sum())
    corr = float(np.corrcoef(a, b)[0, 1])
    print(f"ch{name}: torch_energy={e_a:.3e} candle_energy={e_b:.3e} ratio={e_b/e_a:.3f} corr={corr:.4f}")
print("total torch=%.3e candle=%.3e" % (float((t**2).sum()), float((c**2).sum())))
