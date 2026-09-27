import numpy as np

base = r"D:\alice\audio-separator-rs\test-assets"

def load(path, shape):
    return np.fromfile(path, dtype=np.float32).reshape(shape)

def rms_norm(a):
    dim = a.shape[-1]
    out = np.zeros_like(a)
    for idx in np.ndindex(a.shape[:-1]):
        v = a[idx]
        out[idx] = v / max(np.sqrt(np.sum(v * v)), 1e-12) * np.sqrt(dim)
    return out

# band features cat → [1, 44, 62, 256] → time 输入 [1, 62, 44, 256] → squeeze [62, 44, 256]
band = load(rf"{base}\candle_real_band.npy", (62, 44, 256))          # [band, frame, dim]
time_in = band.transpose(1, 0, 2)                                     # [62, 44, 256]
attn0 = load(rf"{base}\candle_attn_0_xn.npy", (62, 44, 256))
print("attn0_xn vs rms(band->time_in):", float(np.corrcoef(attn0.ravel(), rms_norm(time_in).ravel())[0, 1]))

# layer0_time 输出 [1, 62, 44, 256] → freq 输入 [1, 44, 62, 256] → squeeze [44, 62, 256]
t0 = load(rf"{base}\candle_real_layer0_time.npy", (1, 62, 44, 256))[0]  # [62, 44, 256]
freq_in = t0.transpose(1, 0, 2)                                        # [44, 62, 256]
attn1 = load(rf"{base}\candle_attn_1_xn.npy", (44, 62, 256))
print("attn1_xn vs rms(time_out->freq_in):", float(np.corrcoef(attn1.ravel(), rms_norm(freq_in).ravel())[0, 1]))

# torch 侧：torch_attn_xn = rms(freq_in)；candle attn1 vs torch attn xn
t_xn = np.load(rf"{base}\torch_attn_xn.npy")
print("candle attn1 vs torch attn xn:", float(np.corrcoef(attn1.ravel(), t_xn.ravel())[0, 1]))
