# -*- coding: utf-8 -*-
"""torch 对照：0.5s 混合复音信号（L: 330/440/550Hz, R: 660/880/1100Hz）→ 输出 npy"""
import sys, torch
import numpy as np

MSST = r"C:\Users\alice\AppData\Local\asep-ort\MSST-ref"
sys.path.insert(0, MSST)
from models.bs_roformer.mel_band_roformer import MelBandRoformer

torch.set_grad_enabled(False)
model = MelBandRoformer(
    dim=384, depth=12, stereo=True, num_stems=1,
    time_transformer_depth=1, freq_transformer_depth=1,
    num_bands=60, dim_head=64, heads=8,
    attn_dropout=0.0, ff_dropout=0.0,
    flash_attn=False,
    dim_freqs_in=1025, sample_rate=44100,
    stft_n_fft=2048, stft_hop_length=441, stft_win_length=2048,
    stft_normalized=False,
    zero_dc=True,
    mask_estimator_depth=2,
    mlp_expansion_factor=4,
)
ckpt = torch.load(r"C:\Users\alice\AppData\Local\asep-ort\model_mel_band_roformer_ep_3005_sdr_11.4360.ckpt", map_location="cpu")
sd = ckpt.get("model", ckpt.get("state_dict", ckpt))
if any(k.startswith("model.") for k in sd):
    sd = {k[6:]: v for k, v in sd.items()}
model.load_state_dict(sd, strict=True)
model.eval()

sr = 44100
samples = 22050  # 0.5s
t = np.arange(samples) / sr
L = 0.3 * np.sin(2 * np.pi * 330 * t) + 0.2 * np.sin(2 * np.pi * 440 * t) + 0.1 * np.sin(2 * np.pi * 550 * t)
R = 0.3 * np.sin(2 * np.pi * 660 * t) + 0.2 * np.sin(2 * np.pi * 880 * t) + 0.1 * np.sin(2 * np.pi * 1100 * t)
audio = np.stack([L, R]).astype(np.float32)
x = torch.from_numpy(audio).unsqueeze(0)
out = model(x)
out = out[..., :samples]  # 裁剪到输入长度
o = out[0].numpy()
print("torch out energy: %.9f" % float((o ** 2).sum()))
print("torch out max abs: %.6f" % float(np.abs(o).max()))
np.save(r"D:\alice\audio-separator-rs\test-assets\torch_mel_mix_out.npy", o)
print("saved")
