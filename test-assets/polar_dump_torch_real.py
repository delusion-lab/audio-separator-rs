import numpy as np
import torch
import yaml
import scipy.io.wavfile as wav

from hunterFormsBS import BandSplitRotator

SR, d = wav.read(r"D:\alice\audio-separator-rs\test-assets\qi-feng-le-zh.wav")
samples = 512 * 43
audio = d[:samples].astype(np.float32) / 32768.0
x = torch.tensor(audio.T, dtype=torch.float32).unsqueeze(0)  # [1, 2, t]

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

with torch.no_grad():
    out = model(x)

np.save(r"D:\alice\audio-separator-rs\test-assets\torch_real_band.npy", captured["band_split"])
np.save(r"D:\alice\audio-separator-rs\test-assets\torch_real_mask.npy", captured["mask0"])
np.save(r"D:\alice\audio-separator-rs\test-assets\torch_real_vocals.npy", out[0, 0].numpy())
print("vocals shape:", tuple(out[0, 0].shape), "energy %.3e" % float((out[0, 0] ** 2).sum()))
