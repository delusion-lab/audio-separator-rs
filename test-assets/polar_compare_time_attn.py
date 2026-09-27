import numpy as np

base = r"D:\alice\audio-separator-rs\test-assets"

def load(path, shape):
    return np.fromfile(path, dtype=np.float32).reshape(shape)

def corr(a, b):
    return float(np.corrcoef(a.ravel(), b.ravel())[0, 1])

# torch time attn (id=0 对应 candle)
t_xn = np.load(rf"{base}\torch_time_attn_xn.npy")     # [62, 44, 256]
t_q = np.load(rf"{base}\torch_time_attn_q.npy")       # [62, 8, 44, 64]
t_k = np.load(rf"{base}\torch_time_attn_k.npy")
t_q2 = np.load(rf"{base}\torch_time_attn_q2.npy")     # [62, 8, 44, 128]
t_k2 = np.load(rf"{base}\torch_time_attn_k2.npy")
t_probs = np.load(rf"{base}\torch_time_attn_probs.npy")  # [62, 8, 44, 44]
t_out = np.load(rf"{base}\torch_time_attn_out.npy")   # [62, 44, 256]

c_xn = load(rf"{base}\candle_attn_0_xn.npy", t_xn.shape)
c_q = load(rf"{base}\candle_attn_0_q.npy", t_q.shape)
c_k = load(rf"{base}\candle_attn_0_k.npy", t_k.shape)
c_v = load(rf"{base}\candle_attn_0_v.npy", t_q.shape)
print(f"xn: {corr(t_xn, c_xn):.6f}")
print(f"q: {corr(t_q, c_q):.6f}  k: {corr(t_k, c_k):.6f}  v: {corr(t_k, c_v):.6f}")

for h in range(8):
    c_q2 = load(rf"{base}\candle_attn_0_qp_{h}.npy", t_q2.shape)
    c_k2 = load(rf"{base}\candle_attn_0_kp_{h}.npy", t_k2.shape)
    c_p = load(rf"{base}\candle_attn_0_probs_{h}.npy", t_probs.shape)
    print(f"h{h}: q2={corr(t_q2, c_q2):.6f} k2={corr(t_k2, c_k2):.6f} probs={corr(t_probs, c_p):.6f}")
