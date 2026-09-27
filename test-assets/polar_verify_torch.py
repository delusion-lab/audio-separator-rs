import numpy as np
import torch
import yaml

from hunterFormsBS import BandSplitRotator

SR = 44100
SAMPLES = 512 * 43  # 22016, 0.499s

# 1) 配置（与 models.json config_url 同源 yaml）；mask_estimator_depth 权重实际存储 2 层 → depth=1
cfg = yaml.safe_load(open(r"C:\Users\alice\AppData\Local\asep-ort\model_bs_polarformer_float16.yaml", encoding="utf-8").read().replace("!!python/tuple", ""))
model_cfg = dict(cfg["model"])
model_cfg["mask_estimator_depth"] = 1  # 权重为 2 层 Linear（hunterFormsBS 早期发布行为）
print("model_cfg:", {k: model_cfg[k] for k in ("dim", "depth", "heads", "dim_head", "use_pope", "mask_estimator_depth", "stft_hop_length", "stft_n_fft") if k in model_cfg})

model = BandSplitRotator(**model_cfg, **{k: v for k, v in cfg["audio"].items() if k in ("stft_n_fft", "stft_hop_length", "stft_win_length", "stft_normalized", "sample_rate")})

# 2) 加载原始 fp16 ckpt
state = torch.load(r"C:\Users\alice\AppData\Local\asep-ort\model_bs_polarformer_float16.ckpt", map_location="cpu")
if "state_dict" in state:
    state = state["state_dict"]
missing, unexpected = model.load_state_dict(state, strict=True)
print("load_state_dict OK, missing:", len(missing), "unexpected:", len(unexpected))
model.eval()

# 3) 0.5s 混合信号（与 candle 侧 polar_candle_mix_out 相同）
t = np.arange(SAMPLES) / SR
left = 0.3 * np.sin(2 * np.pi * 330 * t) + 0.2 * np.sin(2 * np.pi * 440 * t) + 0.1 * np.sin(2 * np.pi * 550 * t)
right = 0.3 * np.sin(2 * np.pi * 660 * t) + 0.2 * np.sin(2 * np.pi * 880 * t) + 0.1 * np.sin(2 * np.pi * 1100 * t)
audio = np.stack([left, right])  # [2, n]
x = torch.tensor(audio, dtype=torch.float32).unsqueeze(0)  # [1, 2, n]

with torch.no_grad():
    out = model(x)  # [b, stems, ch, n]
vocals = out[0, 0].numpy()  # [2, n]
print("torch vocals shape:", vocals.shape, "energy: %.3e" % float((vocals**2).sum()))
np.save(r"D:\alice\audio-separator-rs\test-assets\torch_polar_out.npy", vocals)
print("saved torch_polar_out.npy")
