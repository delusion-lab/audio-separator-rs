import numpy as np
import torch
import yaml

from hunterFormsBS import BandSplitRotator

SR = 44100
SAMPLES = 512 * 43  # 22016

cfg = yaml.safe_load(open(r"C:\Users\alice\AppData\Local\asep-ort\model_bs_polarformer_float16.yaml", encoding="utf-8").read().replace("!!python/tuple", ""))
mc = dict(cfg["model"]); mc["mask_estimator_depth"] = 1
model = BandSplitRotator(**mc, **{k: v for k, v in cfg["audio"].items() if k in ("stft_n_fft", "stft_hop_length", "stft_win_length", "stft_normalized", "sample_rate")})
st = torch.load(r"C:\Users\alice\AppData\Local\asep-ort\model_bs_polarformer_float16.ckpt", map_location="cpu")
if "state_dict" in st:
    st = st["state_dict"]
model.load_state_dict(st, strict=True)
model.eval()

# hooks: band_split 输出 与 mask_estimators[0] 输出
captured = {}
def make_hook(name):
    def hook(m, i, o):
        captured[name] = o.detach().float().numpy()
    return hook
model.band_split.register_forward_hook(make_hook("band_split"))
model.mask_estimators[0].register_forward_hook(make_hook("mask0"))

t = np.arange(SAMPLES) / SR
left = 0.3 * np.sin(2 * np.pi * 330 * t) + 0.2 * np.sin(2 * np.pi * 440 * t) + 0.1 * np.sin(2 * np.pi * 550 * t)
right = 0.3 * np.sin(2 * np.pi * 660 * t) + 0.2 * np.sin(2 * np.pi * 880 * t) + 0.1 * np.sin(2 * np.pi * 1100 * t)
x = torch.tensor(np.stack([left, right]), dtype=torch.float32).unsqueeze(0)

with torch.no_grad():
    out = model(x)

print("band_split out:", captured["band_split"].shape)
print("mask0 out:", captured["mask0"].shape)
np.save(r"D:\alice\audio-separator-rs\test-assets\torch_polar_band.npy", captured["band_split"])
np.save(r"D:\alice\audio-separator-rs\test-assets\torch_polar_mask.npy", captured["mask0"])
print("vocals energy: %.3e" % float((out[0, 0] ** 2).sum()))
