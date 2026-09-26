"""验证真实歌曲分离质量：两轨应明显不同（低相关），且各有有效能量。"""
import wave, array, math

def read_mono(path):
    with wave.open(path, 'rb') as w:
        n = w.getnframes(); ch = w.getnchannels()
        data = w.readframes(n)
    s = array.array('h', data)
    return [s[i] / 32768.0 for i in range(0, len(s), ch)]

def rms(x):
    return math.sqrt(sum(a*a for a in x) / len(x))

def corr(x, y):
    n = min(len(x), len(y))
    x, y = x[:n], y[:n]
    mx, my = sum(x)/n, sum(y)/n
    dx = [a-mx for a in x]; dy = [b-my for b in y]
    num = sum(a*b for a, b in zip(dx, dy))
    den = math.sqrt(sum(a*a for a in dx) * sum(b*b for b in dy))
    return num/den if den else 0.0

v = read_mono(r"D:\alice\audio-separator-rs\out-song-url\vocals.wav")
i = read_mono(r"D:\alice\audio-separator-rs\out-song-url\instrumental.wav")
n = min(len(v), len(i))
print(f"时长: {n/44100:.1f}s  vocals RMS={rms(v):.4f}  instrumental RMS={rms(i):.4f}")
print(f"两轨相关系数: {corr(v, i):.4f} (接近 0 说明分离有效)")
# 零交叉率粗略判断 vocals 是否含语音性内容
zc_v = sum(1 for k in range(1, len(v), 100) if v[k-1]*v[k] < 0) / (len(v)//100)
zc_i = sum(1 for k in range(1, len(i), 100) if i[k-1]*i[k] < 0) / (len(i)//100)
print(f"零交叉率(采样间隔100): vocals={zc_v:.4f}  instrumental={zc_i:.4f}")
