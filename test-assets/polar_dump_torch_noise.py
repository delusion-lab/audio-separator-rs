import numpy as np
import torch

from hunterFormsBS import BandSplitRotator

SR = 44100
SAMPLES = 512 * 43
rng = np.random.default_rng(42)

cfg = yaml = None
import yaml
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
    return hook
model.band_split.register_forward_hook(make_hook("band_split"))
model.mask_estimators[0].register_forward_hook(make_hook("mask0"))

t = np.arange(SAMPLES) / SR
noise = 0.001 * rng.standard_normal(SAMPLES)
left = 0.3 * np.sin(2 * np.pi * 330 * t) + 0.2 * np.sin(2 * np.pi * 440 * t) + 0.1 * np.sin(2 * np.pi * 550 * t) + noise
right = 0.3 * np.sin(2 * np.pi * 660 * t) + 0.2 * np.sin(2 * np.pi * 880 * t) + 0.1 * np.sin(2 * np.pi * 1100 * t) + noise
x = torch.tensor(np.stack([left, right]), dtype=torch.float32).unsqueeze(0)

with torch.no_grad():
    out = model(x)

np.save(r"D:\alice\audio-separator-rs\test-assets\torch_polar_band_noise.npy", captured["band_split"])
np.save(r"D:\alice\audio-separator-rs\test-assets\torch_polar_mask_noise.npy", captured["mask0"])
print("vocals energy (noise): %.3e" % float((out[0, 0] ** 2).sum()))
