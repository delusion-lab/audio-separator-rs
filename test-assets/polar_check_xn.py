import numpy as np

base = r"D:\alice\audio-separator-rs\test-assets"

def load(path, shape):
    return np.fromfile(path, dtype=np.float32).reshape(shape)

def rms_norm(a, gamma):
    dim = a.shape[-1]
    out = np.zeros_like(a)
    for idx in np.ndindex(a.shape[:-1]):
        v = a[idx]
        out[idx] = v / max(np.sqrt(np.sum(v * v)), 1e-12) * np.sqrt(dim) * gamma
    return out

# time_in（candle band cat → [1,62,44,256] → squeeze [62,44,256]）
band = load(rf"{base}\candle_real_band.npy", (62, 44, 256))
time_in = band.transpose(1, 0, 2)  # [62, 44, 256]

# gamma：safetensors 权重不方便直接读；用 torch 模型 dump 一份？——先对比无 gamma 的 rms
r = rms_norm(time_in, np.ones(256))
t_xn = np.load(rf"{base}\torch_time_attn_xn.npy")
c_xn = load(rf"{base}\candle_attn_0_xn.npy", (62, 44, 256))
print("numpy rms(no gamma) vs torch xn:", float(np.corrcoef(r.ravel(), t_xn.ravel())[0, 1]))
print("numpy rms(no gamma) vs candle xn:", float(np.corrcoef(r.ravel(), c_xn.ravel())[0, 1]))
print("torch xn vs candle xn:", float(np.corrcoef(t_xn.ravel(), c_xn.ravel())[0, 1]))

# 能量
print("torch xn energy:", float(np.sum(t_xn**2)), "candle xn energy:", float(np.sum(c_xn**2)))
# 每个 frame 的 rms（输入 time_in 的逐行 norm）——若输入一致，逐行 norm 应一致
t_norm = np.linalg.norm(time_in, axis=-1)  # [62, 44]
print("time_in row norms sample:", t_norm[0, :5])
