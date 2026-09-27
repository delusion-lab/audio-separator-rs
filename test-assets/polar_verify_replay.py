import numpy as np
import torch
import yaml
import scipy.io.wavfile as wav
from einops import pack, unpack, rearrange
from hunterFormsBS import BandSplitRotator

base = r"D:\alice\audio-separator-rs\test-assets"
SR, d = wav.read(base + r"\qi-feng-le-zh.wav")
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

band = {}
model.band_split.register_forward_hook(lambda m, i, o: band.__setitem__("b", o.detach().float().numpy()))
real_time = {}
model.layers[0][-2].register_forward_hook(lambda m, i, o: real_time.__setitem__("t", o.detach().float().numpy()))
with torch.no_grad():
    model(x)

xb = torch.tensor(band["b"])
x1 = rearrange(xb, "b t f d -> b f t d")
x1p, ps = pack([x1], "* t d")
time_out = model.layers[0][-2](x1p)
real = real_time["t"]
print("corr manual vs real:", float(np.corrcoef(time_out.detach().numpy().ravel(), real.ravel())[0, 1]))
print("shapes manual", tuple(time_out.shape), "real", tuple(real.shape))
