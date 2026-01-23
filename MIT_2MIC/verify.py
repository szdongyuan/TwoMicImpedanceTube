import os
import numpy as np
from calculate_mit_2mic import calculate_impedance_velocity


def load_calibration_data():
    # 构造校准文件路径
    cal_dir = os.path.join(os.getcwd(), "！校准")
    mic1_path = os.path.join(cal_dir, "MIC1.txt")
    mic2_path = os.path.join(cal_dir, "MIC2.txt")

    # 加载校准数据（float32 精度）
    mic1_cal = np.loadtxt(mic1_path, dtype=np.float32)
    mic2_cal = np.loadtxt(mic2_path, dtype=np.float32)
    return mic1_cal, mic2_cal


def load_record_data():
    # 构造校准文件路径
    cal_dir = os.path.join(os.getcwd(), "！采集")
    mic1_path = os.path.join(cal_dir, "MIC1.txt")
    mic2_path = os.path.join(cal_dir, "MIC2.txt")

    # 加载校准数据（float32 精度）
    rec_cal = np.loadtxt(mic1_path, dtype=np.float32)
    rec_cal2 = np.loadtxt(mic2_path, dtype=np.float32)
    return rec_cal, rec_cal2


rec_cal, rec_cal2 = load_record_data()
mic1_cal_data, mic2_cal_data = load_calibration_data()
ff, f, Z_abs, Z_Re, Z_Im, V_abs, V_Re, V_Im = calculate_impedance_velocity(
    mic2=rec_cal2,
    mic1=rec_cal,
    mic2_cal=mic2_cal_data,
    mic1_cal=mic1_cal_data,
    sf=44100,
    temp=22,
    dia_tube_mm=20,
    mic_dis_mm=17,
    s2=28,
    sen=0.112,
)

import matplotlib.pyplot as plt

plt.figure()
plt.loglog(ff, Z_abs)
plt.xlabel("Frequency [Hz]")
plt.ylabel("Impedance Magnitude |Z| [Pa·s/m³]")
plt.title("Acoustic Impedance Magnitude vs Frequency")
plt.grid(True, which="both", ls="--")
plt.show()

plt.figure()
plt.loglog(f, V_abs)
plt.xlabel("Frequency [Hz]")
plt.ylabel("Volume Velocity Magnitude |V| [m³/s]")
plt.title("Volume Velocity Magnitude vs Frequency")
plt.grid(True, which="both", ls="--")
plt.show()
