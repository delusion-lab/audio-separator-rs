import numpy as np

base = r"D:\alice\audio-separator-rs\test-assets"
# torch q2 [62, 8, 44, 128] 头0 → [44, 128]
t = np.load(rf"{base}\torch_time_attn_q2.npy")  # [62,8,44,128]
t0 = t[:, 0, :, :]  # [62, 44, 128]
# candle qp_0 174592 = 62*44*64 → reshape [62, 44, 64]
c = np.fromfile(rf"{base}\candle_attn_0_qp_0.npy", dtype=np.float32).reshape(62, 44, 64)
for name, a in [("前64", t0[..., :64]), ("后64", t0[..., 64:])]:
    print(f"torch {name} vs candle qp_0: corr={np.corrcoef(a.ravel(), c.ravel())[0,1]:.4f}")
# candle qp 若包含 128 维（数据 174592=62*44*64 只可能 64 维）
print("candle qp_0 energy:", float(np.sum(c**2)), "torch q2 head0 energy:", float(np.sum(t0**2)))
