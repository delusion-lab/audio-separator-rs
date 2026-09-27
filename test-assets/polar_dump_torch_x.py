import numpy as np
import torch
import yaml
import scipy.io.wavfile as wav

from hunterFormsBS import BandSplitRotator

SR, d = wav.read(r"D:\alice\audio-separator-rs\test-assets\qi-feng-le-zh.wav")
samples = 512 * 43
audio = d[:samples].astype(np.float32) / 32768.0
x = torch.tensor(audio.T, dtype=torch.float32).unsqueeze(0)

cfg = yaml.safe_load(open(r"C:\Users\alice\AppData\Local\asep-ort\model_bs_polarformer_float16.yaml", encoding="utf-8").read().replace("!!python/tuple", ""))
mc = dict(cfg["model"]); mc["mask_estimator_depth"] = 1
model = BandSplitRotator(**mc, **{k: v for k, v in cfg["audio"].items() if k in ("stft_n_fft", "stft_hop_length", "stft_win_length", "stft_normalized", "sample_rate")})
st = torch.load(r"C:\Users\alice\AppData\Local\asep-ort\model_bs_polarformer_float16.ckpt", map_location="cpu")
if "state_dict" in st:
    st = st["state_dict"]
model.load_state_dict(st, strict=True)
model.eval()

captured = {}
def make_hook(name):
    def hook(m, i, o):
        captured[name] = o.detach().float().numpy()
        print(f"{name}: in={tuple(i[0].shape)} out={tuple(o.shape)}")
        return o
    return hook
model.final_norm.register_forward_hook(make_hook("final_norm"))
model.mask_estimators[0].register_forward_hook(make_hook("mask0"))
# 每层 time transformer 输出（bandSplitRotator forward: layer[-2] = time, layer[-1] = freq）
for i, layer in enumerate(model.layers):
    layer[-2].register_forward_hook(make_hook(f"layer{i}_time"))
    layer[-1].register_forward_hook(make_hook(f"layer{i}_freq"))

with torch.no_grad():
    out = model(x)

np.save(r"D:\alice\audio-separator-rs\test-assets\torch_real_x.npy", captured["final_norm"])
np.save(r"D:\alice\audio-separator-rs\test-assets\torch_real_mask.npy", captured["mask0"])
for i in range(12):
    np.save(rf"D:\alice\audio-separator-rs\test-assets\torch_real_layer{i}_time.npy", captured[f"layer{i}_time"])
    np.save(rf"D:\alice\audio-separator-rs\test-assets\torch_real_layer{i}_freq.npy", captured[f"layer{i}_freq"])
print("x shape:", captured["final_norm"].shape, "mask0 shape:", captured["mask0"].shape)
