import numpy as np
from scipy.signal import csd, get_window, welch


def tfestimate(x, y, fs, nfft, window):
    f, Pxy = csd(
        x, y, fs=fs, window=window, nperseg=nfft, noverlap=nfft // 2, nfft=nfft
    )
    f = f[1:]
    Pxy = Pxy[1:]
    _, Pxx = welch(x, fs=fs, window=window, nperseg=nfft, noverlap=nfft // 2, nfft=nfft)
    Pxx = Pxx[1:]
    Hxy = Pxy / Pxx
    return Hxy, f


def compute_absorption_coef(
    mic1_step1,
    mic2_step1,
    mic1_step2,
    mic2_step2,
    mic1,
    mic2,
    dia_tube_mm,
    s,
    x1,
    temp,
    sf,
    f1,
    f2,
):
    """
    计算管道中测试样品的反射系数、吸收系数和声阻抗。

    参数:
    --------
    mic1_step1, mic2_step1 : array_like
        第一次校准步骤中两个麦克风采集的数据。
    mic1_step2, mic2_step2 : array_like
        第二次校准步骤中两个麦克风采集的数据。
    mic1, mic2 : array_like
        实际测量过程中两个麦克风采集的数据。
    dia_tube_mm : float
        管道内径（单位：mm）。
    s : float
        两个麦克风之间的间距（单位：mm）。
    x1 : float
        第一个麦克风距离样品的位置（单位：mm）。
    temp : float
        管道内空气温度（单位：摄氏度）。
    sf : float
        采样率（单位：Hz）。
    f1, f2 : float
        分析频率范围（单位：Hz）。

    返回:
    --------
    dict
        包含频率、反射系数、吸收系数、复数阻抗及其实部与虚部。
        - 'frequency': 频率数组
        - 'absorption_coefficient': 吸收系数 alpha
        - 'impedance_real': 阻抗实部
        - 'impedance_imag': 阻抗虚部
    """

    # 单位转换：mm -> m
    dia_tube_mm /= 1000
    s /= 1000
    x1 /= 1000

    # 环境参数
    pa = 101  # 大气压 kPa
    c0 = 343.2 * np.sqrt((temp + 273) / 293)  # 温度修正后的声速
    rho = 1.186 * pa * 293 / (101.325 * (temp + 273))  # 空气密度
    Z0 = rho * c0  # 特征声阻抗

    # 设置频谱估计参数
    nfft = sf
    w = get_window("hann", nfft)

    # 计算三次传递函数
    H12_A, f = tfestimate(
        mic1_step1, mic2_step1, fs=sf, nfft=nfft, window=w
    )  # 校准步骤1
    H12_B, _ = tfestimate(
        mic1_step2, mic2_step2, fs=sf, nfft=nfft, window=w
    )  # 校准步骤2
    H12_p, _ = tfestimate(mic1, mic2, fs=sf, nfft=nfft, window=w)  # 实际测量

    # 截取所需频率范围
    idx = (f >= f1) & (f <= f2)
    f = f[idx]
    H12_A = H12_A[idx]
    H12_B = H12_B[idx]
    H12_p = H12_p[idx]

    # H12 校准归一化处理
    Hc = np.sqrt(H12_A * H12_B)
    H12 = H12_p / Hc

    # 复数波数（考虑粘性损耗）
    k = 2 * np.pi * f / c0 - 1j * 0.0194 * np.sqrt(f) / (c0 * dia_tube_mm)

    # 反射系数 r 的计算
    r = (
        (H12 - np.exp(-1j * k * s))
        / (np.exp(1j * k * s) - H12)
        * np.exp(2j * k * (s + x1))
    )

    # 吸收系数 alpha 计算
    a = 1 - np.abs(r) ** 2

    # 复数声阻抗 z 计算
    z = (1 + r) / (1 - r) * Z0
    z_re = np.real(z)
    z_im = np.imag(z)

    # 返回结果字典
    return {
        "frequency": f,
        "absorption_coefficient": a,
        "impedance_real": z_re,
        "impedance_imag": z_im,
    }
