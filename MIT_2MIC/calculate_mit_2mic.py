def calculate_impedance_velocity(
    mic2, mic1, mic2_cal, mic1_cal, sf, temp, dia_tube_mm, mic_dis_mm, s2, sen
):
    """
    基于双传声器法计算阻抗管的声阻抗 Z 和体积速度 V。

    参数:
    - mic2, mic1, mic2_cal, mic1_cal : ndarray
        四路传感器采集的时域数据（必须是同长度 numpy 数组）。
    - sf : int
        采样率（Hz）
    - temp : float
        实验温度（单位: 摄氏度）
    - atm : float
        实验大气压（单位: kPa）
    - dia_tube_mm : float
        管道直径（单位: mm）
    - mic_dis_mm : float
        麦克风间距（单位: mm）
    - s2 : float
        参考面积（单位: mm^2）
    - sen : float
        传感器灵敏度（用于换算体积速度）

    返回:
    - ff_subset : ndarray
        频率向量 (Z)
    - f_subset : ndarray
        频率向量 (V)
    - Z_abs, Z_Re, Z_Im : ndarray
        Z 的模、实部、虚部
    - V_abs, V_Re, V_Im : ndarray
        V 的模、实部、虚部
    """
    import numpy as np
    from scipy.signal import get_window, csd, welch
    from scipy.fft import fft

    # 常量
    temp0 = 293
    atm0 = 101.325
    rho0 = 1.186
    atm = 101

    # 计算声速、密度、特性阻抗
    c0 = 343.2 * np.sqrt((temp + 273) / temp0)
    rho = rho0 * (atm / atm0) * (temp0 / (temp + 273))
    Z0 = c0 * rho

    # 单位转换
    dia_tube = dia_tube_mm / 1000
    mic_dis = mic_dis_mm / 1000

    # 面积 & 因子
    s1 = np.pi * (dia_tube * 1000 / 2) ** 2
    factor = s1 / s2

    # 频率参数
    N = len(mic2)
    nfft = sf
    w = get_window("hann", nfft)

    # 传递函数估计
    def tfestimate(x, y, fs, nfft, window):
        f, Pxy = csd(
            x, y, fs=fs, window=window, nperseg=nfft, noverlap=nfft // 2, nfft=nfft
        )
        f = f[1:]
        Pxy = Pxy[1:]
        _, Pxx = welch(
            x, fs=fs, window=window, nperseg=nfft, noverlap=nfft // 2, nfft=nfft
        )
        Pxx = Pxx[1:]
        Hxy = Pxy / Pxx
        return Hxy, f

    atf, ff = tfestimate(mic2, mic1, fs=sf, nfft=nfft, window=w)
    atf_cal, _ = tfestimate(mic2_cal, mic1_cal, fs=sf, nfft=nfft, window=w)

    tf = atf / atf_cal
    k = 2 * np.pi * ff / c0 - 0.0194 * np.sqrt(ff) / (c0 * dia_tube)
    Z = Z0 / (1j * np.sin(k * mic_dis)) * (tf - np.cos(k * mic_dis)) / factor

    # FFT & 体积速度计算
    n = np.arange(N).astype(np.float64)
    f = n * sf / N
    k_v = 2 * np.pi * f / c0 - 0.0194 * np.sqrt(f) / (c0 * dia_tube)
    ayfft = fft(mic2, N)
    ayfft = ayfft.reshape(-1)
    k_v = k_v.reshape(-1)
    V = 1j * (ayfft / Z0) * np.sin(k_v * mic_dis) * factor / sen

    # 频率范围筛选
    ind = (ff >= 10) & (ff <= 10000)
    ind_f = (f >= 10) & (f <= 10000)
    ff_subset = ff[ind]
    f_subset = f[ind_f]
    Z_subset = Z[ind]
    V_subset = V[ind_f]

    # 模值和实虚部
    Z_abs = np.abs(Z_subset)
    Z_Re = np.real(Z_subset)
    Z_Im = np.imag(Z_subset)

    V_abs = np.abs(V_subset)
    V_Re = np.abs(np.real(V_subset))
    V_Im = np.abs(np.imag(V_subset))

    return ff_subset, f_subset, Z_abs, Z_Re, Z_Im, V_abs, V_Re, V_Im
