import numpy as np
import glob

base = r"D:\alice\audio-separator-rs\test-assets"
# id=1 = layer0 freq attention
def load(path, shape=None):
    a = np.fromfile(path, dtype=np.float32)
    return a.reshape(shape) if shape else a

# torch 形状
t_xn = np.load(rf"{base}\torch_attn_xn.npy")      # [44, 62, 256]
t_q = np.load(rf"{base}\torch_attn_q.npy")        # [44, 8, 62, 64]
t_k = np.load(rf"{base}\torch_attn_k.npy")
t_q2 = np.load(rf"{base}\torch_attn_q2.npy")      # [44, 8, 62, 128]
t_k2 = np.load(rf"{base}\torch_attn_k2.npy")
t_probs = np.load(rf"{base}\torch_attn_probs.npy")  # [44, 8, 62, 62]
t_out = np.load(rf"{base}\torch_attn_out.npy")    # [44, 62, 256]

def corr(a, b):
    return float(np.corrcoef(a.ravel(), b.ravel())[0, 1])

# candle id=1（freq）：q/k/v/xn [44,8,62,64] 或 [44,62,256]
c_xn = load(rf"{base}\candle_attn_1_xn.npy", t_xn.shape)
c_q = load(rf"{base}\candle_attn_1_q.npy", t_q.shape)
c_k = load(rf"{base}\candle_attn_1_k.npy", t_k.shape)
c_v = load(rf"{base}\candle_attn_1_v.npy", t_q.shape)
print(f"xn: corr={corr(t_xn, c_xn):.6f}")
print(f"q: corr={corr(t_q, c_q):.6f}")
print(f"k: corr={corr(t_k, c_k):.6f}")
print(f"v: corr={corr(np.load(rf'{base}\torch_attn_k.npy'), c_v):.6f}")

# q2/k2/probs 逐头对比
for h in range(8):
    c_q2 = load(rf"{base}\candle_attn_1_qp_{h}.npy", t_q2.shape)
    c_k2 = load(rf"{base}\candle_attn_1_kp_{h}.npy", t_k2.shape)
    c_p = load(rf"{base}\candle_attn_1_probs_{h}.npy", t_probs.shape)
    print(f"h{h}: q2={corr(t_q2, c_q2):.6f} k2={corr(t_k2, c_k2):.6f} probs={corr(t_probs, c_p):.6f}")
