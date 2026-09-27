import numpy as np

# torch band_split [1, 44, 62, 256]；candle 全部 [62, 44, 256]
t = np.load(r"D:\alice\audio-separator-rs\test-assets\torch_polar_band.npy")[0]  # [44, 62, 256]
c = np.fromfile(r"D:\alice\audio-separator-rs\test-assets\candle_polar_all_bands.npy", dtype=np.float32).reshape(62, 44, 256)

# 找 torch band i 对应 candle 哪个 band（最佳相关）
for i in [0, 30, 60, 61]:
    corrs = [float(np.corrcoef(t[:, i, :].ravel(), c[j].ravel())[0, 1]) for j in range(62)]
    best = int(np.argmax(corrs))
    print(f"torch band {i}: best candle band {best} corr={corrs[best]:.4f} (self={corrs[i]:.4f})")

# 全矩阵：torch band j -> candle band i 的相关性矩阵的每行 argmax
row_self = []
row_best = []
for i in range(62):
    corrs = [float(np.corrcoef(t[:, i, :].ravel(), c[j].ravel())[0, 1]) for j in range(62)]
    row_best.append(int(np.argmax(corrs)))
    row_self.append(corrs[i])
row_best = np.array(row_best)
row_self = np.array(row_self)
print("candle band j 的 argmax 排列（torch band i 对应 candle band j）:")
print("row_best:", row_best.tolist())
print("self corr 低 (<0.99) 的 torch band:", np.where(row_self < 0.99)[0].tolist())
