import numpy as np
import torch
import yaml
import scipy.io.wavfile as wav
from einops import rearrange

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

band_captured = {}
model.band_split.register_forward_hook(lambda m, i, o: band_captured.__setitem__("band", o.detach().float().numpy()))
with torch.no_grad():
    out = model(x)

# 手动重放第一层 freq attention
xb = torch.tensor(band_captured["band"])  # [1, 44, 62, 256]
x1 = rearrange(xb, 'b t f d -> b f t d')
# einops pack: 参考 bandSplitRotator.forward 的 pack([x], '* t d')
from einops import pack, unpack
x1p, ps = pack([x1], '* t d')  # [62, 44, 256]
time_out = model.layers[0][-2](x1p)
(x1,) = unpack(time_out, ps, '* t d')  # [1, 62, 44, 256]
freq_in = rearrange(x1, 'b f t d -> b t f d')[0].unsqueeze(0)  # [1, 44, 62, 256]？——torch freq transformer 输入应是 pack 后的 [44, 62, 256]
# 实际 torch forward: freq 前 pack([x], '* f d')：x [1,44,62,256] -> [44, 62, 256]
freq_in_packed, ps_f = pack([rearrange(x1, 'b f t d -> b t f d')], '* f d')  # [44, 62, 256]
freq_out = model.layers[0][-1](freq_in_packed)

# freq attention 中间值（重放）
attn = model.layers[0][-1].layers[0][0]  # Transformer.layers[0][0] = Attention
xn = attn.norm(freq_in_packed)
q, k, v = rearrange(attn.to_qkv(xn), 'b n (qkv h d) -> qkv b h n d', qkv=3, h=8)
# PoPE_pytorch apply_pope_to_qk（pope 需 PolarEmbedReturn，由 PoPE.forward(seq_len) 产生）
from PoPE_pytorch import apply_pope_to_qk
pope_emb = attn.pope_embed(freq_in_packed.shape[1])
q2, k2 = apply_pope_to_qk(pope_emb, q, k)
scores = (q2 @ k2.transpose(-1, -2)) * attn.scale
probs = torch.softmax(scores, dim=-1)
attn_out = probs @ v
gates = attn.to_gates(xn)
attn_out = attn_out * rearrange(gates, 'b n h -> b h n 1').sigmoid()
attn_out = rearrange(attn_out, 'b h n d -> b n (h d)')
attn_out = attn.to_out(attn_out)

np.save(r"D:\alice\audio-separator-rs\test-assets\torch_attn_xn.npy", xn.detach().float().numpy())
np.save(r"D:\alice\audio-separator-rs\test-assets\torch_attn_q.npy", q.detach().float().numpy())
np.save(r"D:\alice\audio-separator-rs\test-assets\torch_attn_k.npy", k.detach().float().numpy())
np.save(r"D:\alice\audio-separator-rs\test-assets\torch_attn_q2.npy", q2.detach().float().numpy())
np.save(r"D:\alice\audio-separator-rs\test-assets\torch_attn_k2.npy", k2.detach().float().numpy())
np.save(r"D:\alice\audio-separator-rs\test-assets\torch_attn_probs.npy", probs.detach().float().numpy())
np.save(r"D:\alice\audio-separator-rs\test-assets\torch_attn_out.npy", attn_out.detach().float().numpy())
np.save(r"D:\alice\audio-separator-rs\test-assets\torch_freq_in.npy", freq_in_packed.detach().float().numpy())
print("freq attention in:", tuple(freq_in_packed.shape), "q:", tuple(q.shape), "probs:", tuple(probs.shape))
