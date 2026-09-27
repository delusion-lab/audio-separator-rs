import numpy as np
import torch

SR = 44100
SAMPLES = 512 * 43
t = np.arange(SAMPLES) / SR
left = 0.3 * np.sin(2 * np.pi * 330 * t) + 0.2 * np.sin(2 * np.pi * 440 * t) + 0.1 * np.sin(2 * np.pi * 550 * t)
right = 0.3 * np.sin(2 * np.pi * 660 * t) + 0.2 * np.sin(2 * np.pi * 880 * t) + 0.1 * np.sin(2 * np.pi * 1100 * t)
audio = np.stack([left, right])
x = torch.tensor(audio, dtype=torch.float32)

win = torch.hann_window(2048)
spec = torch.stft(x, n_fft=2048, hop_length=512, win_length=2048, window=win, return_complex=True, normalized=False)
print("torch stft:", tuple(spec.shape))  # [ch, 1025, frames]
spec = torch.view_as_real(spec).numpy()  # [ch, 1025, frames, 2]

c0 = np.fromfile(r"D:\alice\audio-separator-rs\test-assets\candle_polar_spec0.npy", dtype=np.float32).reshape(-1, 2)
# candle: [bins*frames] 每值 (re, im)；torch: [ch, bin, frame, (re,im)]
frames = SAMPLES // 512 + 1
for ch in range(2):
    c = c0 if ch == 0 else np.fromfile(r"D:\alice\audio-separator-rs\test-assets\candle_polar_spec1.npy", dtype=np.float32).reshape(-1, 2)
    t_ch = spec[ch]  # [1025, frames, 2]
    corrs = []
    for b in range(1025):
        a = t_ch[b].ravel()
        cc = c[b * frames:(b + 1) * frames].ravel()
        corrs.append(float(np.corrcoef(a, cc)[0, 1]))
    corrs = np.array(corrs)
    print(f"ch{ch}: bin corr mean={corrs.mean():.4f} min={corrs.min():.4f}")
    bad = np.where(corrs < 0.99)[0]
    print("  bins corr<0.99:", bad[:20], "count:", len(bad))
    if len(bad):
        b = bad[0]
        a = t_ch[b].ravel(); cc = c[b*frames:(b+1)*frames].ravel()
        print(f"  first bad bin {b}: corr={corrs[b]:.4f} torch[{a[:6]}] candle[{cc[:6]}]")
