import numpy as np
import matplotlib.pyplot as plt
from pydsm.audio_weightings import (
    a_weighting,
    b_weighting,
    c_weighting,
    d_weighting,
    f_weighting,
)

# 生成频率范围（10 Hz 至 20 kHz）
frequencies = np.logspace(np.log10(10), np.log10(20000), 1000)

# 计算各加权滤波器的频率响应（线性）
a_response = a_weighting(frequencies, normal=True, power=True)
b_response = b_weighting(frequencies, normal=True, power=True)
c_response = c_weighting(frequencies, normal=True, power=True)
d_response = d_weighting(frequencies, normal=True, power=True)
f_response = f_weighting(frequencies, normal=True, power=True)

# 未加权（Flat）曲线
flat_response = np.ones_like(frequencies)

# 转换为 dB
a_db = 20 * np.log10(a_response)
b_db = 20 * np.log10(b_response)
c_db = 20 * np.log10(c_response)
d_db = 20 * np.log10(d_response)
f_db = 20 * np.log10(f_response)
flat_db = 20 * np.log10(flat_response)  # 等于 0 dB

# 绘制频率响应图（dB）
plt.figure(figsize=(8, 6))  # 正方形
plt.semilogx(
    frequencies, flat_db, label="Flat (Unweighted)", linestyle="--", color="black"
)
plt.semilogx(frequencies, a_db, label="A-weighting")
plt.semilogx(frequencies, b_db, label="B-weighting")
plt.semilogx(frequencies, c_db, label="C-weighting")
plt.semilogx(frequencies, d_db, label="D-weighting")
plt.semilogx(frequencies, f_db, label="F-weighting")

plt.yscale("linear")
plt.title("Frequency Response of Audio Weighting Filters (dB)")
plt.xlabel("Frequency (Hz)")
plt.ylabel("Gain (dB)")
plt.grid(True, which="both", axis="both")
plt.xlim(10, 100000)  # x轴范围 10 Hz - 100 kHz
plt.legend()
plt.tight_layout()
plt.show()
