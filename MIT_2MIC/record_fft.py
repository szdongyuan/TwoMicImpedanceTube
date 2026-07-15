import numpy as np
import sounddevice as sd
import matplotlib.pyplot as plt
from scipy.signal import chirp


def compute_fft(y, sr):
    """
    计算输入信号的快速傅里叶变换（FFT）并返回频率和幅值谱

    参数:
        y : 输入信号（一维数组）
        sr: 采样率（Hz）

    返回:
        f  : 频率数组
        mag: 幅值谱
    """
    Y = np.fft.rfft(y)  # 对信号进行实数快速傅里叶变换，返回正频率部分
    f = np.fft.rfftfreq(len(y), 1/sr)  # 计算对应的频率坐标
    return f, np.abs(Y)  # 返回频率和幅值谱


def main():
    sr = 48000
    dur = 3.0
    t = np.linspace(0, dur, int(sr * dur), endpoint=False)
    x = (0.05 * chirp(t, f0=20, f1=12000, t1=dur, method='logarithmic')).astype(np.float32)
    y = sd.playrec(x, samplerate=sr, channels=1, dtype=np.float32)
    sd.wait()
    y = y.flatten()
    tt = np.linspace(0, len(y) / sr, len(y), endpoint=False)
    f, mag = compute_fft(y, sr)
    plt.figure(figsize=(8, 5))
    plt.subplot(2, 1, 1)
    plt.plot(tt, y)
    plt.subplot(2, 1, 2)
    plt.semilogx(f, mag)
    plt.tight_layout()
    plt.show()


if __name__ == '__main__':
    main()
