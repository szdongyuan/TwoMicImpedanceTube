import json
import os
import shutil
import sys

import sounddevice as sd
import numpy as np
import yaml
from PyQt5.QtCore import Qt
from PyQt5.QtGui import QPixmap
from PyQt5.QtWidgets import (
    QLabel,
    QMessageBox,
    QDialog,
    QGraphicsPixmapItem,
    QGraphicsScene,
    QVBoxLayout, QApplication,
)
from scipy.signal import get_window, csd, welch, savgol_filter
from scipy.fft import fft

from control.log_manager import LogManager
from control.output_voltage_interface import SoundcardCalibrationManager
from custom.running_consts import base_dir, DEFAULT_DIR


utils_logger = LogManager.set_log_handler("core")

trim_time = 0.2

def get_screen_metrics(screen=None):
    """Return the current screen's size, available area, DPI and scale factor."""
    app = QApplication.instance()
    if app is None:
        raise RuntimeError("QApplication must be created before reading screen metrics")

    screen = screen or app.primaryScreen()
    if screen is None:
        raise RuntimeError("No available screen was detected")

    geometry = screen.geometry()
    available = screen.availableGeometry()
    return {
        "name": screen.name(),
        "width": geometry.width(),
        "height": geometry.height(),
        "available_width": available.width(),
        "available_height": available.height(),
        "logical_dpi": round(screen.logicalDotsPerInch(), 2),
        "device_pixel_ratio": screen.devicePixelRatio(),
    }


def get_font_size_for_resolution(width, height):
    """Choose a readable point size from the screen's shorter edge."""
    short_side = min(int(width), int(height))
    if short_side <= 768:
        return 10
    if short_side <= 1080:
        return 12
    if short_side <= 1440:
        return 14
    return 16


def set_adjust_button_enabled(button, enabled: bool):
    button.setEnabled(enabled)
    if enabled:
        button.setStyleSheet("""
               QToolButton {
                       padding-top: 6px;
                       padding-bottom: 3px;
                       color: black;
                }
           """)
    else:
        # 禁用样式（灰色）
        button.setStyleSheet("""
               QToolButton {
                       padding-top: 6px;
                       padding-bottom: 3px;
                       color: gray;
                   }
           """)

def set_run_button_enabled(button, enabled: bool):
    """
    控制 run_test_button 启用状态和样式
    :param enabled: True 启用按钮，False 禁用按钮
    """
    button.setEnabled(enabled)
    if enabled:
        # 启用样式（亮蓝色）
        button.setStyleSheet("""
             QToolButton {
                    background-color: rgb(0, 170, 255);
                    border-radius: 5px;
                    font: bold 12pt "Microsoft YaHei";
                }
        """)
    else:
        # 禁用样式（灰色）
        button.setStyleSheet("""
            QToolButton {
                    background-color: rgb(209, 209, 209);
                    border-radius: 5px;
                    font: bold 12pt "Microsoft YaHei";
                    color: gray;
                }
        """)

def is_float(str):
    try:
        float(str)
        return True
    except ValueError:
        return False

def get_popup_image_max_size(parent, popup_size=(820, 560), ratio=0.8):
    """
    根据弹窗所属界面计算图片最大显示尺寸。
    大图不会超过 popup_size，也不会超过父界面的 ratio 比例。
    """
    max_w, max_h = popup_size

    if parent is not None:
        parent_w = parent.width()
        parent_h = parent.height()
        if parent_w > 0 and parent_h > 0:
            max_w = min(max_w, int(parent_w * ratio))
            max_h = min(max_h, int(parent_h * ratio))

    if max_w <= 0 or max_h <= 0:
        screen = QApplication.primaryScreen()
        size = screen.availableGeometry()
        max_w = int(size.width() * ratio)
        max_h = int(size.height() * ratio)

    return max(max_w, 1), max(max_h, 1)


def show_image_popup(parent, image_path, title="图片预览", popup_size=(820, 560), ratio=0.8):
    # 加载要弹出的原始图片。
    pixmap = QPixmap(image_path)
    if pixmap.isNull():
        QMessageBox.warning(parent, "提示", "图片加载失败！")
        return

    max_w, max_h = get_popup_image_max_size(parent, popup_size, ratio)

    # 按最大宽高缩放，保持原图比例不变。
    pixmap = pixmap.scaled(max_w, max_h, Qt.KeepAspectRatio, Qt.SmoothTransformation)

    # 创建一个简单弹窗，把缩放后的图片放进去。
    dialog = QDialog(parent)
    dialog.setWindowTitle(title)

    image_label = QLabel(dialog)
    image_label.setAlignment(Qt.AlignCenter)
    image_label.setPixmap(pixmap)

    layout = QVBoxLayout(dialog)
    layout.setContentsMargins(8, 8, 8, 8)
    layout.addWidget(image_label)

    dialog.resize(pixmap.width() + 20, pixmap.height() + 20)
    dialog.exec_()

def set_clickable_graphics_image(parent, view, image_path, preview_size, title="图片预览", popup_size=(820, 560),
                                 popup_ratio=0.8):
    # 先在界面上的 QGraphicsView 里显示缩略图。
    pixmap = QPixmap(image_path)
    if pixmap.isNull():
        QMessageBox.warning(parent, "提示", "图片加载失败！")
        return

    scene = QGraphicsScene(view)
    scene.addItem(QGraphicsPixmapItem(pixmap.scaled(preview_size[0], preview_size[1],
                                                    Qt.KeepAspectRatio, Qt.SmoothTransformation)))
    view.setScene(scene)
    view.setCursor(Qt.PointingHandCursor)
    view.setToolTip("点击查看大图")

    # 鼠标左键点击缩略图时，弹出大图窗口。
    original_mouse_press_event = view.mousePressEvent

    def mouse_press_event(event):
        if event.button() == Qt.LeftButton:
            show_image_popup(parent, image_path, title, popup_size, popup_ratio)
        else:
            original_mouse_press_event(event)

    view.mousePressEvent = mouse_press_event

def show_error_message(msg: str, parent=None) -> None:
    """
    在主线程弹出错误提示框。
    parmas：
        msg (str): 错误提示文本。
        parent (QWidget): 父窗口对象，用于对话框定位（可选）。
    """
    QMessageBox.critical(parent, "错误", msg)

def show_success_message(msg: str, parent=None) -> None:
    """在主线程弹出成功提示框。"""
    QMessageBox.information(parent, "成功", msg)

def show_warning_message(msg: str, parent=None) -> None:
    """在主线程弹出警告提示框。"""
    QMessageBox.warning(parent, "警告", msg)


def generate_chirp_wrapper(info: dict, samplerate: int):
    return generate_chirp(
        duration=info["signal_time"],
        f_start=info["hz_down"],
        f_stop=info["hz_up"],
        amplitude=info["signal_amplitude"],
        sample_rate = samplerate
    )


def generate_chirp(duration: float,
                   f_start: float,
                   f_stop: float,
                   amplitude: float = 1.0,
                   sample_rate: int = 44100,
                   mirror: bool = False,
                   repeat_times: int = 1):
    """
    生成对数 chirp 信号 (logarithmic sweep)

    Parameters
    ----------
    duration      : float   单次信号时长 (s)
    f_start       : float   起始频率  (Hz)
    f_stop        : float   终止频率  (Hz)
    amplitude     : float   ±幅值   (默认 1.0)
    sample_rate   : int     采样率 (Hz，默认 44100Hz)
    mirror        : bool    True 时生成镜像扫频 (先↑再↓并反相)，时长加倍
    repeat_times  : int     将整段信号重复播放 N 次

    Returns
    -------
    t : np.ndarray   时间轴 (s)
    y : np.ndarray   波形数据 (float32)
    """
    # ---------- 单段 chirp ----------
    t_single = np.linspace(0, duration, int(sample_rate * duration), endpoint=False)

    if f_start == f_stop:  # 万一两频率一样，就退化成恒频
        phase = 2 * np.pi * f_start * t_single
    else:
        ln_ratio = np.log(f_stop / f_start)
        phase = 2 * np.pi * f_start * duration / ln_ratio * (
                np.exp(ln_ratio * t_single / duration) - 1
        )

    y_single = amplitude * np.sin(phase)

    # ---------- 镜像扫频（可选） ----------
    if mirror:
        # 反向 + 反相，拼接在后
        y_single = np.concatenate([y_single[::-1], -y_single])
        t_single = np.linspace(0, duration * 2, y_single.size, endpoint=False)

    # ---------- 重复播放（可选） ----------
    if repeat_times > 1:
        y = np.tile(y_single, repeat_times)
        t = np.linspace(0, t_single[-1] * repeat_times + t_single[1], y.size, endpoint=False)
    else:
        y, t = y_single, t_single

    return t, y.astype(np.float32)

def calculate_impedance_velocity(
    mic2, mic1, mic2_cal, mic1_cal,
    sf,
    temp,
    dia_tube_mm,
    mic_dis_mm,
    s2,
    sen
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
    w = get_window('hann', nfft)

    # 传递函数估计
    def tfestimate(x, y, fs, nfft, window):
        f, Pxy = csd(x, y, fs=fs, window=window, nperseg=nfft, noverlap=nfft//2, nfft=nfft)
        f = f[1:]
        Pxy = Pxy[1:]
        _, Pxx = welch(x, fs=fs, window=window, nperseg=nfft, noverlap=nfft//2, nfft=nfft)
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

    index = np.where(ff_subset == 1000)[0]

    utils_logger.info(f"1000Hz 对应索引: {index}, Z_abs: {Z_abs[index]}")
    return ff_subset, f_subset, Z_abs, Z_Re, Z_Im, V_abs, V_Re, V_Im

def smooth_data(data, window_length=51, polyorder=3):
    """
    对数据进行Savitzky-Golay平滑。

    参数:
        - data: ndarray
            需要平滑的数据。
        - window_length: int, 可选
            滤波器的窗口大小，必须是奇数。默认值是51。
        - polyorder: int, 可选
            多项式的阶数。默认值是3。

    返回:
        - smoothed_data: ndarray
            平滑后的数据。
    """
    return savgol_filter(data, window_length=window_length, polyorder=polyorder)


def copy_all_configs_to_base_dir():
    logger = LogManager.set_log_handler("拷贝配置文件")
    # 创建目录
    target_config_dir = os.path.join(base_dir, "resources", "config")
    os.makedirs(target_config_dir, exist_ok=True)

    json_files = [
        "imp_tube_params_setting.json",
        "output_signal_setting.json",
        "basic_params.json",
        "calibration_coefficients.json",
        "mic_calibration.json",
        "tree_config.json",
        "speaker_cfg_test.yaml",
    ]

    for filename in json_files:
        target_path = os.path.join(target_config_dir, filename)
        # 若已存在就跳过，避免覆盖用户数据
        if os.path.exists(target_path):
            continue

        try:
            # 判断运行模式：打包或源码
            if getattr(sys, 'frozen', False):
                # exe 模式下资源路径
                sys_path = os.path.join(sys._MEIPASS, "resources", "config", filename)
                shutil.copyfile(sys_path, target_path)

                logger.info(f"[已复制] {filename} → {target_path}")
            else:
                # 源码运行时路径
                pass

        except Exception as e:
            logger.info(f"[错误] 复制 {filename} 失败: {e}")


def get_config_path(filename):
    """
    获取配置文件路径，兼容打包与源码运行
    """
    try:
        if getattr(sys, 'frozen', False):
            # exe 打包后, 从迁移的 base_dir中获取
            return os.path.join(base_dir, "resources", "config", filename)
        else:
            # 源码运行时, 根目录下
            return os.path.join(DEFAULT_DIR, "resources", "config", filename)
    except Exception as e:
        utils_logger.error(f"获取配置路径失败: {e}")
        return None

def get_config_content(filename):
    """
    读取配置文件内容（JSON）。
    参数:
        filename (str): 配置文件名，例如 "basic_params.json"
    返回:
            - (True, config) : 读取成功，返回配置字典
            - (False, None) : 读取失败
    """
    path = get_config_path(filename)
    try:
        with open(path, "r", encoding="utf-8") as f:
            config = json.load(f)
        return True, config
    except Exception:
        return False, None

def write_config_content(filename, config):
    """
    将更新后的配置内容保存回文件。
    参数:
        filename (str): 配置文件名，例如 "mic_calibration.json"
        config (dict): 要保存的配置字典
    返回:
        - True : 保存成功
        - False : 保存失败
    """
    path = get_config_path(filename)
    try:
        with open(path, "w", encoding="utf-8") as f:
            json.dump(config, f, indent=4, ensure_ascii=False)
        return True
    except Exception as e:
        utils_logger.error(f"写入文件失败: {e}")
        return False

def get_history_data_path(data_file, config_filename="tree_config.json"):
    """
    将配置里保存的 data_file 转成实际磁盘路径。
    data_file 通常是相对 tree_config.json 所在目录的路径，例如 history_lines/xxx.npz。
    """
    if not data_file:
        return None
    if os.path.isabs(data_file):
        return data_file
    config_path = get_config_path(config_filename)
    return os.path.normpath(os.path.join(os.path.dirname(config_path), data_file))

def save_history_line_data(config_filename, data_file, data):
    """
    保存单条历史曲线的大数组数据。
    npz 是 numpy 的压缩归档格式，适合保存 ff、Z_abs、V_abs 这类数组。
    """
    path = get_history_data_path(data_file, config_filename)
    if not path:
        return False

    try:
        # 创建resources/config/history_lines文件夹
        os.makedirs(os.path.dirname(path), exist_ok=True)
        array_data = {}
        for key, value in data.items():
            # 统一转成 numpy 数组后保存，读取时可以保持原来的数值结构。
            array_data[key] = np.asarray(value)
        # 创建并写入 .npz 文件
        np.savez_compressed(path, **array_data)
        return True
    except Exception as e:
        utils_logger.error(f"保存历史线数据失败: {e}")
        return False

def load_history_line_data(config_filename, data_file):
    """
    读取历史曲线数据。
    data_file 是配置中记录的 npz 文件路径。
    """
    if not data_file:
        return None

    path = get_history_data_path(data_file, config_filename)
    try:
        # allow_pickle=False 避免加载任意 Python 对象，只读取普通数组数据。
        with np.load(path, allow_pickle=False) as npz:
            return {key: npz[key] for key in npz.files}
    except Exception as e:
        utils_logger.error(f"读取历史线数据失败: {e}")
        return None

def delete_history_line_data(config_filename, data_file):
    """删除一条历史曲线对应的 npz 数据文件。"""
    path = get_history_data_path(data_file, config_filename)
    if not path or not os.path.exists(path):
        return True

    try:
        os.remove(path)
        return True
    except Exception as e:
        utils_logger.error(f"删除历史线数据失败: {e}")
        return False

def init_selected_devide():
    """
    初始化输入输出设备。
    - 从 basic_params.json 读取配置（设备名、类型）。
    - 在本机设备中查找匹配的输入/输出设备。
    - 找到后设置为默认设备，否则报错。
    """
    logger = LogManager.set_log_handler("初始化选择设备")

    try:
        config_path = get_config_path("basic_params.json")
        if not os.path.exists(config_path):
            logger.critical(f"配置文件{config_path}不存在")
            raise FileNotFoundError(f"配置文件{config_path}不存在")
        with open(config_path, "r", encoding="utf-8") as f:
            config = json.load(f)
            input_name = config["basic_params"]["input_selected_device_name"]
            input_type = config["basic_params"]["input_selected_device_type"]
            output_name = config["basic_params"]["output_selected_device_name"]
            output_type = config["basic_params"]["output_selected_device_type"]
            devices = sd.query_devices()
            hostapis = sd.query_hostapis()

            input_index = None
            output_index = None

            # 匹配输入设备
            for idx, dev in enumerate(devices):
                if dev["max_input_channels"] > 0:
                    dev_name = dev["name"]
                    dev_type = hostapis[dev["hostapi"]]["name"]
                    if dev_name == input_name and dev_type == input_type:
                        input_index = idx
                        logger.info(f"匹配到输入设备 index={input_index}：{dev_name} ({dev_type})")
                        break

            # 匹配输出设备
            for idx, dev in enumerate(devices):
                if dev["max_output_channels"] > 0:
                    dev_name = dev["name"]
                    dev_type = hostapis[dev["hostapi"]]["name"]
                    if dev_name == output_name and dev_type == output_type:
                        output_index = idx
                        logger.info(f"匹配到输出设备 index={output_index}：{dev_name} ({dev_type})")
                        break
            # 如果都匹配到了，则设置默认设备
            if input_index is not None and output_index is not None:
                sd.default.device = (input_index, output_index)
                logger.info(f"已设置默认输入输出设备: {input_index}, {output_index}")
            else:
                logger.critical(f"找不到匹配的输入或输出设备，请重新选择")
                raise RuntimeError("提示：默认设备已改变, 请重新选择设备！")

    except Exception as e:
        QMessageBox.warning(None, "初始化设备失败", f"{e}")


def save_selected_devices(input_device_index, output_device_index, input_device_name, output_device_name, input_type,
                          output_type, input_channel_count, input_sample_rate, output_sample_rate):
    """
    保存选择的设备信息到 basic_params.json。
    - 打开配置文件并更新输入/输出设备参数。
    - 将修改后的配置写回文件。
    """
    try:
        if input_sample_rate not in (44100, 48000) or output_sample_rate not in (44100, 48000):
            raise ValueError("输入和输出设备的采样率只能为 44100 Hz 或 48000 Hz。")
        config_path = get_config_path("basic_params.json")

        # 如果文件不存在，直接报错
        if not os.path.exists(config_path):
            raise FileNotFoundError(f"配置文件不存在: {config_path}")

        # 加载配置文件
        with open(config_path, "r", encoding="utf-8") as f:
            config = json.load(f)

        # 确保 basic_params 存在
        if "basic_params" not in config:
            raise KeyError("配置文件缺少 'basic_params' 字段")
        # 更新设备信息
        if (
                config["basic_params"].get("input_selected_device_id") != input_device_index
                or config["basic_params"].get("input_selected_device_name") != input_device_name
                or config["basic_params"].get("input_selected_device_type") != input_type
                or config["basic_params"].get("input_sample_rate") != input_sample_rate
        ):
            config["basic_params"]["is_clear"] = 1  # 1是清空mic_cal.json
        config["basic_params"]["input_selected_device_id"] = input_device_index
        config["basic_params"]["input_selected_device_name"] = input_device_name
        config["basic_params"]["input_selected_device_type"] = input_type
        config["basic_params"]["input_sample_rate"] = input_sample_rate

        config["basic_params"]["output_selected_device_id"] = output_device_index
        config["basic_params"]["output_selected_device_name"] = output_device_name
        config["basic_params"]["output_selected_device_type"] = output_type
        config["basic_params"]["output_sample_rate"] = output_sample_rate
        config["basic_params"]["input_channel_count"] = input_channel_count
        # 写入文件
        with open(config_path, "w", encoding="utf-8") as f:
            json.dump(config, f, indent=4, ensure_ascii=False)
    except Exception as e:
        raise RuntimeError(f"保存 selected_devices 到 basic_params.json 失败: {e}")

def refresh_mic_config_json(obj):
    try:
        config_path = get_config_path("basic_params.json")

        # 如果文件不存在，直接报错
        if not os.path.exists(config_path):
            raise FileNotFoundError(f"配置文件不存在: {config_path}")

        # 加载配置文件
        with open(config_path, "r", encoding="utf-8") as f:
            config = json.load(f)
            count = config["basic_params"]["input_channel_count"]
        mic_json_path = get_config_path("mic_calibration.json")
        if config["basic_params"]["is_clear"] == 1:
            # 按 count 生成 { "In1": {...}, ..., "InN": {...} }
            mic_cfg = {
                f"In{i}": {
                    "deviation_value": "1.0",
                    "binding_index": i - 1,
                    "sensitivity": "0"
                }
                for i in range(1, count + 1)
            }
            if count == 0:
                mic_cfg = {}
            with open(mic_json_path, "w", encoding="utf-8") as file:
                json.dump(mic_cfg, file, ensure_ascii=False, indent=4)
            config["basic_params"]["is_clear"] = 0
            with open(config_path, "w", encoding="utf-8") as f:
                json.dump(config, f, ensure_ascii=False, indent=4)
            obj.logger.info(
                f"is_clear=1，已按通道数({count})重建 mic_calibration.json -> {mic_json_path}"
            )
        else:
            obj.logger.info("is_clear=0，mic_calibration.json 保持不变")
    except Exception as e:
        raise RuntimeError(f"重载mic_calibration.json 失败: {e}")


def custom_log_tick_strings(values, scale, spacing):
    estrings = ["%0.1g" % x for x in 10 ** np.array(values).astype(float) * np.array(scale)]
    convdict = {
        "0": "⁰",
        "1": "¹",
        "2": "²",
        "3": "³",
        "4": "⁴",
        "5": "⁵",
        "6": "⁶",
        "7": "⁷",
        "8": "⁸",
        "9": "⁹",
    }
    dstrings = []
    for i, e in enumerate(estrings):
        if "e" in e:
            v, p = e.split("e")
            sign = "⁻" if p[0] == "-" else ""
            pot = "".join([convdict[pp] for pp in p[1:].lstrip("0")])
            if v == "1":
                v = ""
                dstrings.append(v + "10" + sign + pot)
            elif v == "2" or v == "5":
                v = v + "·"
                dstrings.append(v + "10" + sign + pot)
            else:
                dstrings.append("")
        else:
            dstrings.append(e)
    return dstrings

def safe_log(a):
    a = np.asarray(a, dtype=float)
    a[a <= 0] = np.nan  # 防止取对数报错
    return np.log10(a)


def trim_start_samples(recording, samplerate):
    """
    多录 trim_time 后，统一裁掉开头，避免启动瞬间爆音参与计算和保存。
    """
    trim_samples = int(round(trim_time * samplerate))
    if recording is None or trim_samples <= 0:
        return recording
    if recording.shape[0] <= trim_samples:
        return recording[:0]
    return recording[trim_samples:]

def generate_calibrated_signal(signal_info, samplerate):
    """
    生成校准后的激励信号
    参数:
        signal_info: dict, 包含信号参数，其中 "signal_amplitude" 表示目标电压幅值 (V)
        samplerate: int, 采样率

    返回:
        success (bool): 是否成功
        data (ndarray): 已按校准幅值缩放后的信号数据
        cal_amplitude (float): 校准后的幅值（实际用于缩放的因子，单位 V）
        error_msg (str): 失败时的错误信息，否则为 None
    """
    params_voltage = signal_info["signal_amplitude"]
    scm = SoundcardCalibrationManager()
    # 用y=ax+b校准函数求幅值因子cal_amplitude
    calibrate_code, calibrate_result = scm.calibrate_amplitude(params_voltage)
    if calibrate_code != True:
        return False, None, None, f"校准幅值失败：{calibrate_result}"
    cal_amplitude, max_voltage = calibrate_result
    if params_voltage > max_voltage:
        return False, None, None, f"参数电压过大（{params_voltage}V），最大可达 {max_voltage}V。"
    # 信号归一化
    signal_info['signal_amplitude'] = 1
    _, data = generate_chirp_wrapper(signal_info, samplerate)
    # 用幅值因子进行缩放
    data = data * cal_amplitude
    return True, data, cal_amplitude, None

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

def calculate_rms(signal: np.ndarray) -> float:
    """
    计算输入信号的均方根值 (RMS)
    :param signal: numpy 数组类型的信号数据
    :return: RMS 值（float）
    """
    if len(signal) == 0:
        return 0.0
    return np.sqrt(np.mean(np.square(signal)))


def calculate_scale(real_spl: float, rms: float) -> float:
    """
    根据真实声压级 (SPL, 单位 dB SPL) 和信号的 RMS 计算缩放系数。
    :param real_spl: 真实声压级，单位 dB SPL（如 94 dB）
    :param rms: 当前信号的均方根值（RMS）
    :return: 缩放系数
    """
    logger = LogManager.set_log_handler("计算缩放scale")
    if rms == 0 or not np.isfinite(rms):
        logger.error(f"RMS 值无效: {rms}")
        return 0.0
    p = 20e-6 * 10 ** (real_spl / 20)
    scale = p / rms
    return scale

def get_yaml_content(filename):
    """
    读取配置文件内容（YAML）。
    参数:
        filename (str): 配置文件名，例如 "basic_params.yaml"
    返回:
        - (True, config, None): 读取成功
        - (False, None, error_message): 读取失败，附带错误信息
    """
    path = get_config_path(filename)
    try:
        with open(path, "r", encoding="utf-8") as f:
            config = yaml.safe_load(f)
        return True, config, None
    except Exception as e:
        return False, None, str(e)


def resize_by_ui_with_screen(obj, ratio=0.9):
    """
    负责缩整个窗口。
    1.先读取当前窗口 .ui 里的原始宽高，比如设备选择界面是 1200 x 600。
    2.再读取电脑屏幕可用区域，比如扣掉任务栏后的大小。
    3.用 ratio 算出窗口最多能占屏幕多少，比如 ratio=0.9 表示最多占屏幕可用区域的 90%。
    4.如果窗口太大，就按比例缩小；如果本来就放得下，就保持原大小。
    5.把这个缩放比例保存到：obj._ui_scale
    """
    ui_w, ui_h = obj.width(), obj.height()

    screen = QApplication.primaryScreen()
    size = screen.availableGeometry()
    max_w = int(size.width() * ratio)
    max_h = int(size.height() * ratio)

    # 只在原始 UI 超过屏幕比例上限时缩小；能放下时保持 1.0。
    scale = min(max_w / ui_w, max_h / ui_h, 1.0)
    # resize 只改变窗口本身，scale 返回给界面内部控件继续同步缩放。
    obj.resize(int(ui_w * scale), int(ui_h * scale))
    obj._ui_scale = scale
    interface_name = obj.windowTitle() or obj.__class__.__name__
    utils_logger.info(f"[{interface_name}] 缩放比例scale:{obj._ui_scale}")
    return scale

def scale_ui_value(obj, value, minimum=1):
    """
    负责缩窗口里面写死尺寸的控件。

    resize_by_ui_with_screen(obj) 会把窗口缩放比例保存到 obj._ui_scale。
    这里把原始设计尺寸 value 乘以该比例，并用 minimum 做下限，
    避免按钮、下拉框等固定尺寸控件在小屏幕下缩得太小。
    例如：_ui_scale=0.8，value=350，minimum=180，返回 280。
    """
    ui_scale = getattr(obj, "_ui_scale", 1.0)
    scaled_value = int(round(value * ui_scale))
    return max(minimum, scaled_value)
