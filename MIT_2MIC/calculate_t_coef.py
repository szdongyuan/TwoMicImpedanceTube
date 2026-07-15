import numpy as np
from scipy.signal import csd, get_window, welch


# 定义一个通用读取函数
def read_txt(filepath):
    return np.loadtxt(filepath)


# 路径前缀
base_path = r"C:\Users\Win\Desktop\za\新建文件夹 (2)(1)\新建文件夹 (2)"

# 第一阶段采集
ax_a = read_txt(rf"{base_path}\！隔声采集\MIC1_第一阶段.txt")
bx_a = read_txt(rf"{base_path}\！隔声采集\MIC2_第一阶段.txt")
cx_a = read_txt(rf"{base_path}\！隔声采集\MIC3_第一阶段.txt")
dx_a = read_txt(rf"{base_path}\！隔声采集\MIC4_第一阶段.txt")

# 第二阶段采集（rigid termination mic data）
ax_b = read_txt(rf"{base_path}\！隔声采集\MIC1_第二阶段.txt")
bx_b = read_txt(rf"{base_path}\！隔声采集\MIC2_第二阶段.txt")
cx_b = read_txt(rf"{base_path}\！隔声采集\MIC3_第二阶段.txt")
dx_b = read_txt(rf"{base_path}\！隔声采集\MIC4_第二阶段.txt")

# 校准数据（calibration data）
ax_cal = read_txt(rf"{base_path}\！隔声校准\MIC1.txt")
bx_cal = read_txt(rf"{base_path}\！隔声校准\MIC2.txt")
cx_cal = read_txt(rf"{base_path}\！隔声校准\MIC3.txt")
dx_cal = read_txt(rf"{base_path}\！隔声校准\MIC4.txt")


def calculate_t_coef(
    mic1a,
    mic2a,
    mic3a,
    mic4a,  # 消声端测量
    mic1b,
    mic2b,
    mic3b,
    mic4b,  # 刚性端测量
    mic1,
    mic2,
    mic3,
    mic4,  # 校准测量
    dia_tube,
    D,
    s1,
    s2,
    L1,
    L2,
    temp,
    sf,
    f1,
    f2,
):
    """
    使用四传声器法测量材料的声学参数：传输损失、吸收系数、复数波数与特征阻抗。

    参数:
    --------
    micXa, micXb, micX : array_like
        分别为消声端测量、刚性端测量和校准测量的传声器信号。
    dia_tube : float
        管道内径 [mm]
    D : float
        材料样品长度 [mm]
    s1, s2 : float
        Mic1-Mic2 和 Mic3-Mic4 的间距 [mm]
    L1, L2 : float
        Mic1 到样品、Mic3 到样品的距离 [mm]
    temp : float
        空气温度 [摄氏度]
    sf : float
        采样率 [Hz]
    f1, f2 : float
        计算频率范围 [Hz]

    返回:
    --------
    dict
        包含频率、传输损失、吸收系数、复波数和特征阻抗的实部与虚部
    """

    # 单位转换：mm → m
    dia_tube /= 1000
    D /= 1000
    s1 /= 1000
    s2 /= 1000
    L1 /= 1000
    L2 /= 1000

    # 环境与标准参数
    atm = 101.0  # 实际大气压 kPa
    atm0 = 101.325  # 标准大气压
    temp0 = 293  # 标准温度 K
    rho0 = 1.186  # 标准空气密度 kg/m³
    c0 = 343.2 * np.sqrt((temp + 273) / temp0)  # 声速
    rho = rho0 * (atm / atm0) * (temp0 / (temp + 273))  # 实际空气密度
    Z0 = c0 * rho  # 特征阻抗

    # 设置频谱估计参数
    nfft = sf
    w = get_window("hann", nfft)

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

    # 计算 H12, H13, H14（3组传递函数）
    def compute_transfer_functions(m1, m2, m3, m4):
        H12, f = tfestimate(m1, m2, fs=sf, nfft=nfft, window=w)
        H13, _ = tfestimate(m1, m3, fs=sf, nfft=nfft, window=w)
        H14, _ = tfestimate(m1, m4, fs=sf, nfft=nfft, window=w)
        return H12, H13, H14, f

    H12_a, H13_a, H14_a, f = compute_transfer_functions(mic1a, mic2a, mic3a, mic4a)
    H12_b, H13_b, H14_b, _ = compute_transfer_functions(mic1b, mic2b, mic3b, mic4b)
    H12_cal, H13_cal, H14_cal, _ = compute_transfer_functions(mic1, mic2, mic3, mic4)

    # 限定频率范围
    idx = (f >= f1) & (f <= f2)
    f = f[idx]
    H12_a = H12_a[idx] / H12_cal[idx]
    H13_a = H13_a[idx] / H13_cal[idx]
    H14_a = H14_a[idx] / H14_cal[idx]
    H12_b = H12_b[idx] / H12_cal[idx]
    H13_b = H13_b[idx] / H13_cal[idx]
    H14_b = H14_b[idx] / H14_cal[idx]

    # 复波数（包含粘性耗散）
    k = 2 * np.pi * f / c0 - 1j * 0.0194 * np.sqrt(f) / (c0 * dia_tube)

    # 计算入射与反射波系数 A/B（消声端）
    A_a = -(np.exp(-1j * k * L1) - H12_a * np.exp(-1j * k * (L1 + s1))) / (
        2j * np.sin(k * s1)
    )
    B_a = -(H12_a * np.exp(1j * k * (L1 + s1)) - np.exp(1j * k * L1)) / (
        2j * np.sin(k * s1)
    )
    C_a = -(H13_a * np.exp(1j * k * (L2 + s2)) - H14_a * np.exp(1j * k * L2)) / (
        2j * np.sin(k * s2)
    )
    D_a = -(H14_a * np.exp(-1j * k * L2) - H13_a * np.exp(-1j * k * (L2 + s2))) / (
        2j * np.sin(k * s2)
    )

    # 计算入射/反射声压与粒子速度（消声端）
    p0a = A_a + B_a
    u0a = (A_a - B_a) / Z0
    pda = C_a * np.exp(-1j * k * D) + D_a * np.exp(1j * k * D)
    uda = (C_a * np.exp(-1j * k * D) - D_a * np.exp(1j * k * D)) / Z0

    # 同理计算刚性端
    A_b = -(np.exp(-1j * k * L1) - H12_b * np.exp(-1j * k * (L1 + s1))) / (
        2j * np.sin(k * s1)
    )
    B_b = -(H12_b * np.exp(1j * k * (L1 + s1)) - np.exp(1j * k * L1)) / (
        2j * np.sin(k * s1)
    )
    C_b = -(H13_b * np.exp(1j * k * (L2 + s2)) - H14_b * np.exp(1j * k * L2)) / (
        2j * np.sin(k * s2)
    )
    D_b = -(H14_b * np.exp(-1j * k * L2) - H13_b * np.exp(-1j * k * (L2 + s2))) / (
        2j * np.sin(k * s2)
    )

    p0b = A_b + B_b
    u0b = (A_b - B_b) / Z0
    pdb = C_b * np.exp(-1j * k * D) + D_b * np.exp(1j * k * D)
    udb = (C_b * np.exp(-1j * k * D) - D_b * np.exp(1j * k * D)) / Z0

    # 计算传递矩阵元件
    M = pda * udb - pdb * uda
    T11 = (p0a * udb - p0b * uda) / M
    T12 = (p0b * pda - p0a * pdb) / M
    T21 = (u0a * udb - u0b * uda) / M
    T22 = (pda * u0b - pdb * u0a) / M

    # 计算传输损失、反射率、吸收系数、复数波数与特征阻抗
    t = abs(2 * np.exp(1j * k * D) / (T11 + T12 / Z0 + T21 * Z0 + T22))
    TL = 20 * np.log10(np.abs(1 / t))
    R = abs((T11 - Z0 * T21) / (T11 + Z0 * T21))
    alpha = 1 - np.abs(R) ** 2
    # k_eff = np.arccos(T11) / D

    # z0 = acos(T11)
    z0 = np.arccos(T11)

    # z = z0
    z = z0.copy()

    # prev = z0(2)
    prev = z0[1]   # MATLAB 索引 2 → Python 索引 1

    # for ii = 3:length(z0)
    for ii in range(2, len(z0)):  # MATLAB 3 → Python 2
        a = z0[ii]

        c1 = a + 2 * np.pi * np.round(np.real(prev - a) / (2 * np.pi))
        c2 = -a + 2 * np.pi * np.round(np.real(prev + a) / (2 * np.pi))

        if np.abs(c2 - prev) < np.abs(c1 - prev):
            z[ii] = c2
            prev = c2
        else:
            z[ii] = c1
            prev = c1

    # k_b = z./D
    k_eff = z / D

    Zc = np.sqrt(T12 / T21)

    transmission_coef = t
    # ✅ 打印最大最小值（调试用）
    print("透射系数 t 最大值:", np.max(transmission_coef))
    print("透射系数 t 最小值:", np.min(transmission_coef))

    return {
        "frequency": f,
        "transmission_coef": t,
        "transmission_loss": TL,
        "absorption_coefficient": alpha,
        "k_real": np.real(k_eff),
        "k_imag": np.imag(k_eff),
        "Zc_real": np.real(Zc),
        "Zc_imag": np.imag(Zc),
    }
