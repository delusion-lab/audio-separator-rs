import numpy as np
import torch
import yaml
import scipy.io.wavfile as wav
from einops import rearrange, pack
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
with torch.no_grad():
    model(x)

xb = torch.tensor(band["b"])
x1 = rearrange(xb, "b t f d -> b f t d")
x1p, ps = pack([x1], "* t d")
time_in = x1p  # [62, 44, 256]
attn = model.layers[0][-2].layers[0][0]  # time Attention
xn = attn.norm(time_in)
q, k, v = rearrange(attn.to_qkv(xn), "b n (qkv h d) -> qkv b h n d", qkv=3, h=8)
from PoPE_pytorch import apply_pope_to_qk
pope_emb = attn.pope_embed(time_in.shape[1])
q2, k2 = apply_pope_to_qk(pope_emb, q, k)
scores = (q2 @ k2.transpose(-1, -2)) * attn.scale
probs = torch.softmax(scores, dim=-1)
attn_out = probs @ v
gates = attn.to_gates(xn)
attn_out = attn_out * rearrange(gates, "b n h -> b h n 1").sigmoid()
attn_out = rearrange(attn_out, "b h n d -> b n (h d)")
attn_out = attn.to_out(attn_out)

np.save(base + r"\torch_time_attn_xn.npy", xn.detach().float().numpy())
np.save(base + r"\torch_time_attn_q.npy", q.detach().float().numpy())
np.save(base + r"\torch_time_attn_k.npy", k.detach().float().numpy())
np.save(base + r"\torch_time_attn_q2.npy", q2.detach().float().numpy())
np.save(base + r"\torch_time_attn_k2.npy", k2.detach().float().numpy())
np.save(base + r"\torch_time_attn_probs.npy", probs.detach().float().numpy())
np.save(base + r"\torch_time_attn_out.npy", attn_out.detach().float().numpy())
print("time attn in:", tuple(time_in.shape), "q:", tuple(q.shape), "probs:", tuple(probs.shape))
