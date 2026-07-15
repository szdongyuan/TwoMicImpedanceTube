import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import bilinear, freqz


# -----------------------------
# 1. 定义 A 计权滤波器 (NUM/DEN)
# -----------------------------
def A_weighting_filter(fs):
    f1 = 20.6
    f2 = 107.7
    f3 = 737.9
    f4 = 12194.0
    A1000 = 2.0
    pi = np.pi

    NUM = [(2 * pi * f4) ** 2 * (10 ** (A1000 / 20.0)), 0, 0, 0, 0]
    DEN = np.convolve(
        [1, 4 * pi * f4, (2 * pi * f4) ** 2], [1, 4 * pi * f1, (2 * pi * f1) ** 2]
    )
    DEN = np.convolve(DEN, [1, 2 * pi * f2])
    DEN = np.convolve(DEN, [1, 2 * pi * f3])

    b, a = bilinear(NUM, DEN, fs)
    return b, a


# -----------------------------
# 2. 频率响应公式 R_A(f)
# -----------------------------
def A_weighting_R(f):
    f1, f2, f3, f4 = 20.6, 107.7, 737.9, 12194.0
    A1000 = 2.0
    s = 1j * 2 * np.pi * f  # s = j*omega
    NUM = (2 * np.pi * f4) ** 2 * 10 ** (A1000 / 20) * s**4
    DEN = (
        (s**2 + 4 * np.pi * f4 * s + (2 * np.pi * f4) ** 2)
        * (s**2 + 4 * np.pi * f1 * s + (2 * np.pi * f1) ** 2)
        * (s + 2 * np.pi * f2)
        * (s + 2 * np.pi * f3)
    )
    R = np.abs(NUM / DEN)
    return 20 * np.log10(R)  # dB


# -----------------------------
# 3. 设置采样率与频率
# -----------------------------
fs = 48000
b, a = A_weighting_filter(fs)

# 用 freqz 获取数字滤波器幅值响应
w, h = freqz(b, a, worN=8000, fs=fs)
h_db = 20 * np.log10(abs(h))

# 频率响应公式
f = np.logspace(np.log10(10), np.log10(20000), 8000)
r_db = A_weighting_R(f)

# -----------------------------
# 4. 绘图对比
# -----------------------------
plt.figure(figsize=(8, 5))
plt.semilogx(w, h_db, label="时域滤波器 freqz")
plt.semilogx(f, r_db, "--", label="频率响应公式 R(f)")
plt.xlabel("Frequency [Hz]")
plt.ylabel("Gain [dB]")
plt.title("A-Weighting Comparison")
plt.grid(True, which="both", ls="--")
plt.legend()
plt.show()
