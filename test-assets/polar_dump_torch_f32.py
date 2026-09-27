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
# 转 float32（关键：candle 是 f32）
model = model.float()

captured = {}
def make_hook(name):
    def hook(m, i, o):
        captured[name] = o.detach().float().numpy()
    return hook
model.final_norm.register_forward_hook(make_hook("final_norm"))
model.mask_estimators[0].register_forward_hook(make_hook("mask0"))
for i, layer in enumerate(model.layers):
    layer[-2].register_forward_hook(make_hook(f"layer{i}_time"))
    layer[-1].register_forward_hook(make_hook(f"layer{i}_freq"))

with torch.no_grad():
    out = model(x)

np.save(r"D:\alice\audio-separator-rs\test-assets\torch_real_x_f32.npy", captured["final_norm"])
np.save(r"D:\alice\audio-separator-rs\test-assets\torch_real_mask_f32.npy", captured["mask0"])
for i in range(12):
    np.save(rf"D:\alice\audio-separator-rs\test-assets\torch_real_layer{i}_time_f32.npy", captured[f"layer{i}_time"])
    np.save(rf"D:\alice\audio-separator-rs\test-assets\torch_real_layer{i}_freq_f32.npy", captured[f"layer{i}_freq"])
np.save(r"D:\alice\audio-separator-rs\test-assets\torch_real_vocals_f32.npy", out[0, 0].numpy())
print("done; vocals energy %.3e" % float((out[0, 0] ** 2).sum()))
