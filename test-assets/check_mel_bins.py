import sys
sys.path.insert(0, r"D:\alice\audio-separator-rs\test-assets")
from verify_mel import mel_filter_bank

W = mel_filter_bank(44100, 2048, 60)
W[0][0] = 1.0
W[-1][-1] = 1.0
freqs_per_band = [[w > 0 for w in row] for row in W]

# 每个 band 覆盖的 freq 列表
coverage = []
for i in range(60):
    cov = [f for f in range(1025) if freqs_per_band[i][f]]
    coverage.append(cov)

# bin20=440Hz, bin41=880Hz
for bin_idx, label in [(20, "440Hz"), (41, "880Hz")]:
    bands = [i for i in range(60) if freqs_per_band[i][bin_idx]]
    print(f"{label} bin{bin_idx} covered by bands {bands}, num_bands={len(bands)}")

print("band13 freqs:", coverage[13])
print("band23 freqs:", coverage[23])
print("band13 contains bin20:", 20 in coverage[13])
print("band23 contains bin41:", 41 in coverage[23])
# 440Hz 与 880Hz 实际在哪些 band
for bin_idx, label in [(20, "440Hz"), (41, "880Hz")]:
    for i in range(60):
        if bin_idx in coverage[i]:
            print(f"  {label} bin{bin_idx} in band {i} (len {len(coverage[i])})")
