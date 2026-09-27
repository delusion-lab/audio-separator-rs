# -*- coding: utf-8 -*-
"""mel_band_roformer torch 对照验证：MSST 实现 vs 我们的 candle 输出。
用法: venv-mel python mel_verify_torch.py
"""
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
missing, unexpected = model.load_state_dict(sd, strict=True)
print("load_state_dict strict OK, missing:", len(missing), "unexpected:", len(unexpected))
model.eval()

sr = 44100
samples = 4410
audio = np.zeros((2, samples), dtype=np.float32)
t = np.arange(samples) / sr
audio[0] = 0.1 * np.sin(2 * np.pi * 440.0 * t)
audio[1] = 0.1 * np.sin(2 * np.pi * 880.0 * t)
x = torch.from_numpy(audio).unsqueeze(0)  # [1,2,4410]
out = model(x)  # [1,2,6458]
print("torch out shape:", tuple(out.shape))
out = out[..., :4410]
o = out[0].numpy()
print("torch out energy: %.9f" % float((o ** 2).sum()))
print("torch out max abs: %.6f" % float(np.abs(o).max()))

# 保存输出供 candle 侧对比
np.save(r"D:\alice\audio-separator-rs\test-assets\torch_mel_out.npy", o)
print("saved torch_mel_out.npy")
