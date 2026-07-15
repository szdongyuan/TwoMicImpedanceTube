import numpy as np
import matplotlib.pyplot as plt
from calculate_t_coef import calculate_t_coef, read_txt

# 数据文件路径
mic1a = read_txt(r'！隔声采集/MIC1_第一阶段.txt')
mic2a = read_txt(r'！隔声采集/MIC2_第一阶段.txt')
mic3a = read_txt(r'！隔声采集/MIC3_第一阶段.txt')
mic4a = read_txt(r'！隔声采集/MIC4_第一阶段.txt')

mic1b = read_txt(r'！隔声采集/MIC1_第二阶段.txt')
mic2b = read_txt(r'！隔声采集/MIC2_第二阶段.txt')
mic3b = read_txt(r'！隔声采集/MIC3_第二阶段.txt')
mic4b = read_txt(r'！隔声采集/MIC4_第二阶段.txt')

mic1 = read_txt(r'！隔声校准/MIC1.txt')
mic2 = read_txt(r'！隔声校准/MIC2.txt')
mic3 = read_txt(r'！隔声校准/MIC3.txt')
mic4 = read_txt(r'！隔声校准/MIC4.txt')

# 参数（来自图片）
dia_tube = 29.0  # 阻抗管内径（mm）
D = 43         # 样品到样品前表面距离（mm）
s1 = 20.0        # 传声器之间的距离1（mm）
s2 = 20.0        # 传声器之间的距离2（mm）
L1 = 90       # Mic1到样品前表面距离（mm）
L2 = 247.0       # Mic3到样品后表面距离（mm）
temp = 22.0      # 管中温度（℃）
sf = 48000     # 采样率（Hz）
f1 = 172.26      # 频率下限（Hz）
f2 = 6890.0     # 频率上限（Hz）

# 调用计算函数
data = calculate_t_coef(
    mic1a, mic2a, mic3a, mic4a,
    mic1b, mic2b, mic3b, mic4b,
    mic1, mic2, mic3, mic4,
    dia_tube, D, s1, s2, L1, L2, temp, sf, f1, f2
)

f = data['frequency']
t = data['transmission_coef']

# 输出f和t
print('f:', f)
print('t:', t)

# 画图
plt.figure(figsize=(8, 5))
plt.semilogx(f, np.abs(t))
plt.xlabel('Frequency (Hz)')
plt.ylabel('Transmission Coefficient |t|')
plt.ylim(0, 1)
plt.title('Transmission Coefficient vs Frequency')
plt.grid(True, which='both', ls='--', lw=0.5)
plt.tight_layout()
plt.show() 