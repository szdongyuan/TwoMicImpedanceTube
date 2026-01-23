import numpy as np
from scipy.signal import get_window, csd, welch, medfilt
from scipy.fft import fft


def calculate_mit_4mic(
    ax=None, bx=None, cx=None, dx=None, 
    ax_cal=None, bx_cal=None, cx_cal=None, dx_cal=None,
    sf=44100, temp=25, atm=101, dia_tube_mm=20, 
    L_mm=100, x1_mm=10, x2_mm=30, x3_mm=50, 
    s2=28, sens=0.212, vfactor=1.0
):
    """
    基于四传声器法计算阻抗管的声阻抗 Z 和体积速度 V。
    
    参数:
    - ax, bx, cx, dx : ndarray
        采集数据：四个麦克风的时域数据
    - ax_cal, bx_cal, cx_cal, dx_cal : ndarray
        校准数据：四个麦克风的时域数据
    - sf : int, default=44100
        采样率（Hz）
    - temp : float, default=25
        实验温度（摄氏度）
    - atm : float, default=101
        实验大气压（kPa）
    - dia_tube_mm : float, default=20
        管道直径（mm）
    - L_mm : float, default=40
        管道长度（mm）
    - x1_mm, x2_mm, x3_mm : float, default=270,75,50
        麦克风位置（mm）
    - s2 : float, default=28
        参考面积（mm²）
    - sens : float, default=0.212
        传感器灵敏度
    - vfactor : float, default=1.0
        速度因子
    
    返回:
    - out_ff : ndarray
        频率向量 (列向量)
    - out_absZ : ndarray
        Z 的模 (列向量)
    - out_realZ : ndarray
        Z 的实部绝对值 (列向量)
    - out_imagZ : ndarray
        Z 的虚部绝对值 (列向量)
    - out_absV : ndarray
        V 的模 (列向量)
    - out_realV : ndarray
        V 的实部绝对值 (列向量)
    - out_imagV : ndarray
        V 的虚部绝对值 (列向量)
    """
    
    # 常量定义
    atm0 = 101.325  # 标准大气压 千帕斯卡
    temp0 = 298     # 298 K, 25 摄氏度
    rho0 = 1.186    # 空气标准密度 千克立方米
    
    # 计算声速、密度、特性阻抗
    c0 = 343.2 * np.sqrt((temp + 273) / temp0)  # 实际声速 米每秒
    rho = rho0 * (atm / atm0) * (temp0 / (temp + 273))  # 空气实际密度 千克立方米
    Z0 = c0 * rho0
    
    # 单位转换：mm -> m
    dia_tube = dia_tube_mm / 1000
    L = L_mm / 1000
    x1 = x1_mm / 1000
    x2 = x2_mm / 1000
    x3 = x3_mm / 1000
    
    # 计算面积因子
    s1 = np.pi * ((dia_tube * 1000) / 2) ** 2  # 圆管横截面积
    factor = s1 / s2  # 实际截面积比例
    
    # FFT参数设置
    nfft = sf
    w = get_window('hann', nfft)  # 汉宁窗
    
    # 传递函数估计函数
    def tfestimate(x, y, fs, nfft, window):
        f, Pxy = csd(x, y, fs=fs, window=window, nperseg=nfft, noverlap=nfft//2, nfft=nfft)
        f = f[1:]
        Pxy = Pxy[1:]
        _, Pxx = welch(x, fs=fs, window=window, nperseg=nfft, noverlap=nfft//2, nfft=nfft)
        Pxx = Pxx[1:]
        Hxy = Pxy / Pxx
        return Hxy, f
    
    # 传递函数计算
    tf43, ff = tfestimate(dx, cx, fs=sf, nfft=nfft, window=w )      # H43, frequency
    tf13, _ = tfestimate(ax, cx, fs=sf, nfft=nfft, window=w)      # H13, frequency, 221Hz - 1500Hz
    tf23, _ = tfestimate(bx, cx, fs=sf, nfft=nfft, window=w)      # H23, frequency, 1501Hz - 10kHz
    tf43_cal, _ = tfestimate(dx_cal, cx_cal, fs=sf, nfft=nfft, window=w)  # H43 calibrate, frequency
    tf13_cal, _ = tfestimate(ax_cal, cx_cal, fs=sf, nfft=nfft, window=w)  # H13 calibrate, frequency, 221Hz - 1500Hz
    tf23_cal, _ = tfestimate(bx_cal, cx_cal, fs=sf, nfft=nfft, window=w)  # H23, frequency, 1501Hz - 10kHz
    
    # MIT方法计算
    tf = np.zeros(len(ff), dtype=complex)
    k = np.zeros(len(ff))
    Z1 = np.zeros(len(ff), dtype=complex)
    
    for n in range(len(ff)):
        tf[n] = tf43[n] / tf43_cal[n]
        k[n] = 2 * np.pi * ff[n] / c0 - 0.0194 * np.sqrt(ff[n]) / (c0 * dia_tube)
        Z1[n] = Z0 / (1j * np.sin(k[n] * L)) * (tf[n] - np.cos(k[n] * L)) / factor
    
    # 计算速度
    N = len(dx)
    n = np.arange(N)
    p4fft = fft(dx, N)
    p4fftmag = vfactor * p4fft * 2 / N  # 转化为实际幅值
    f = n * sf / N
    
    k_v = np.zeros(len(f))
    v = np.zeros(len(f), dtype=complex)
    
    for n in range(len(f)):
        k_v[n] = 2 * np.pi * f[n] / c0 - 0.0194 * np.sqrt(f[n]) / (c0 * dia_tube)
        v[n] = p4fftmag[n] / Z0 * 1j * np.sin(k_v[n] * L) * factor / sens
    
    # 221Hz - 1500Hz 有效
    tf2 = np.zeros(len(ff), dtype=complex)
    Z2 = np.zeros(len(ff), dtype=complex)
    r13 = np.zeros(len(ff), dtype=complex)
    
    for n in range(len(ff)):
        tf2[n] = tf13[n] / tf13_cal[n]
        k[n] = 2 * np.pi * ff[n] / c0 - 0.0194 * np.sqrt(ff[n]) / (c0 * dia_tube)
        r13[n] = (tf2[n] * np.exp(1j * k[n] * x1) - np.exp(1j * k[n] * x3)) / (np.exp(-1j * k[n] * x3) - tf2[n] * np.exp(-1j * k[n] * x1))
        Z2[n] = Z0 / (1j * np.sin(k[n] * L)) * (tf[n] * ((1 + r13[n]) / (np.exp(1j * k[n] * x3) + r13[n] * np.exp(-1j * k[n] * x3))) - np.cos(k[n] * L)) / factor
    
    # 1501Hz - 10kHz 有效
    tf3 = np.zeros(len(ff), dtype=complex)
    Z3 = np.zeros(len(ff), dtype=complex)
    r23 = np.zeros(len(ff), dtype=complex)
    
    for n in range(len(ff)):
        tf3[n] = tf23[n] / tf23_cal[n]
        k[n] = 2 * np.pi * ff[n] / c0 - 0.0194 * np.sqrt(ff[n]) / (c0 * dia_tube)
        r23[n] = (tf3[n] * np.exp(1j * k[n] * x2) - np.exp(1j * k[n] * x3)) / (np.exp(-1j * k[n] * x3) - tf3[n] * np.exp(-1j * k[n] * x2))
        Z3[n] = Z0 / (1j * np.sin(k[n] * L)) * (tf[n] * ((1 + r23[n]) / (np.exp(1j * k[n] * x3) + r23[n] * np.exp(-1j * k[n] * x3))) - np.cos(k[n] * L)) / factor
    
    # 合并数据
    V = np.zeros(10000, dtype=complex)
    fre = np.zeros(10000)
    Z = np.zeros(10000, dtype=complex)
    
    # 速度数据 (11:10001)
    for n in range(10, 10001):
        V[n-10] = v[n]
    
    # 频率和阻抗数据 (11:221)
    for n in range(10, 221):
        fre[n-10] = ff[n]
        Z[n-10] = Z1[n]
    
    # 频率和阻抗数据 (222:1501)
    for n in range(221, 1501):
        fre[n-10] = ff[n]
        Z[n-10] = Z2[n]
    
    # 频率和阻抗数据 (1502:10001)
    for n in range(1501, 10001):
        fre[n-10] = ff[n]
        Z[n-10] = Z3[n]
    
    # 数据输出（中值滤波）
    out_ff = fre  
    out_absZ = medfilt(np.abs(Z), 11)
    out_realZ = medfilt(np.abs(np.real(Z)), 11)
    out_imagZ = medfilt(np.abs(np.imag(Z)), 11)
    out_absV = medfilt(np.abs(V), 11)
    out_realV = medfilt(np.abs(np.real(V)), 11)
    out_imagV = medfilt(np.abs(np.imag(V)), 11)
    
    return out_ff, out_absZ, out_realZ, out_imagZ, out_absV, out_realV, out_imagV



