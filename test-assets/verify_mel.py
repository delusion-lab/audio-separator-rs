import math, sys

def hz_to_mel_slaney(f):
    f_min = 0.0
    f_sp = 200.0 / 3
    mels = (f - f_min) / f_sp
    if f >= 1000.0:
        min_log_hz = 1000.0
        min_log_mel = (min_log_hz - f_min) / f_sp
        logstep = math.log(6.4) / 27.0
        mels = min_log_mel + math.log(f / min_log_hz) / logstep
    return mels

def mel_to_hz_slaney(m):
    f_min = 0.0
    f_sp = 200.0 / 3
    freqs = f_min + f_sp * m
    if m >= 15.0:
        min_log_hz = 1000.0
        min_log_mel = (min_log_hz - f_min) / f_sp
        logstep = math.log(6.4) / 27.0
        freqs = min_log_hz * math.exp(logstep * (m - min_log_mel))
    return freqs

def mel_filter_bank(sr, n_fft, n_mels):
    n_freqs = n_fft // 2 + 1
    fft_freqs = [float(sr) / 2 * j / (n_freqs - 1) for j in range(n_freqs)]  # linspace(0, sr/2, n_freqs)
    min_mel = hz_to_mel_slaney(0.0)
    max_mel = hz_to_mel_slaney(float(sr) / 2)
    mels = [min_mel + (max_mel - min_mel) * i / (n_mels + 1) for i in range(n_mels + 2)]
    mel_f = [mel_to_hz_slaney(m) for m in mels]
    fdiff = [mel_f[i + 1] - mel_f[i] for i in range(n_mels + 1)]
    weights = [[0.0] * n_freqs for _ in range(n_mels)]
    for i in range(n_mels):
        for j in range(n_freqs):
            lower = -(mel_f[i] - fft_freqs[j]) / fdiff[i]
            upper = (mel_f[i + 2] - fft_freqs[j]) / fdiff[i + 1]
            weights[i][j] = max(0.0, min(lower, upper))
    for i in range(n_mels):
        enorm = 2.0 / (mel_f[i + 2] - mel_f[i])
        for j in range(n_freqs):
            weights[i][j] *= enorm
    return weights

sr, n_fft, n_mels = 44100, 2048, 60
W = mel_filter_bank(sr, n_fft, n_mels)
W[0][0] = 1.0
W[-1][-1] = 1.0

# freq_indices / counts
freqs_per_band = [[w > 0 for w in row] for row in W]
num_freqs_per_band = [sum(r) for r in freqs_per_band]
num_bands_per_freq = [sum(1 for b in range(n_mels) if freqs_per_band[b][f]) for f in range(n_fft // 2 + 1)]
print("num_freqs_per_band:", num_freqs_per_band)
print("sum freqs:", sum(num_freqs_per_band))
print("num_bands_per_freq min/max:", min(num_bands_per_freq), max(num_bands_per_freq))
# 对照权重输入维度 in = 2*num_freqs*2(stereo)
print("band input dims (stereo):", [2 * nf * 2 for nf in num_freqs_per_band])
# 验证 band0 = 28, band1 = 24
print("band0/band1 input:", 2 * num_freqs_per_band[0] * 2, 2 * num_freqs_per_band[1] * 2)

# 与 librosa 对照（若可用）
try:
    import numpy as np
    import librosa
    Wl = librosa.filters.mel(sr=sr, n_fft=n_fft, n_mels=n_mels)
    Wl[0][0] = 1.0
    Wl[-1][-1] = 1.0
    max_diff = max(abs(W[i][j] - float(Wl[i][j])) for i in range(n_mels) for j in range(n_fft // 2 + 1))
    print("librosa max diff:", max_diff)
    fl = [float(x) for x in librosa.core.audio.hz_to_mel(np.array([44100/2]))]
    print("librosa max_mel:", fl, "mine:", hz_to_mel_slaney(sr/2))
except ImportError:
    print("librosa 未安装，跳过对照")
