"""验证 MDX 分离结果：输入为 440Hz(人声) + 880Hz(伴奏) 混合，
期望 vocals.wav 主能量在 440Hz，instrumental.wav 主能量在 880Hz。"""
import wave, math, sys

def read_wav_mono(path):
    with wave.open(path, 'rb') as w:
        n = w.getnframes(); ch = w.getnchannels(); sr = w.getframerate()
        data = w.readframes(n)
    import array
    samples = array.array('h', data)
    # 转 float，取左声道
    mono = [samples[i] / 32768.0 for i in range(0, len(samples), ch)]
    return mono, sr

def dft_amp(sig, sr, freq):
    # Goertzel 简化：直接 DFT 单频
    n = len(sig)
    k = int(round(freq * n / sr))
    re = im = 0.0
    for i, x in enumerate(sig):
        ang = 2 * math.pi * freq * i / sr
        re += x * math.cos(ang)
        im -= x * math.sin(ang)
    return (re * re + im * im) ** 0.5 / n

for path in [r"D:\alice\audio-separator-rs\out-test\vocals.wav",
             r"D:\alice\audio-separator-rs\out-test\instrumental.wav"]:
    sig, sr = read_wav_mono(path)
    a440 = dft_amp(sig, sr, 440.0)
    a880 = dft_amp(sig, sr, 880.0)
    a_total = sum(abs(x) for x in sig) / len(sig)
    name = path.split('\\')[-1]
    print(f"{name}: sr={sr} len={len(sig)/sr:.1f}s")
    print(f"  440Hz 幅度={a440:.6f}  880Hz 幅度={a880:.6f}  平均|幅度|={a_total:.6f}")
    # 总能量中该频率占比（相对总幅度）
    print(f"  440Hz 占比={a440/max(a_total,1e-9):.3f}  880Hz 占比={a880/max(a_total,1e-9):.3f}")
