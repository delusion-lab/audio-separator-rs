import numpy as np

base = r"D:\alice\audio-separator-rs\test-assets"
for name in ["time", "freq"]:
    print(f"--- torch fp16 vs torch f32: {name} ---")
    for i in range(12):
        a = np.load(rf"{base}\torch_real_layer{i}_{name}.npy")
        b = np.load(rf"{base}\torch_real_layer{i}_{name}_f32.npy")
        print(f"layer{i}: corr={np.corrcoef(a.ravel(), b.ravel())[0,1]:.4f} shape_a={a.shape} shape_b={b.shape}")
