#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
扬声器频率响应测试程序 - Qt6版本
Speaker Frequency Response Test with Qt6
"""

import sys
import numpy as np
from PyQt6.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, 
                           QHBoxLayout, QGridLayout, QLabel, QLineEdit, QPushButton, 
                           QSlider, QGroupBox, QMessageBox, QProgressBar, QFrame, QCheckBox)
from PyQt6.QtCore import Qt, QThread, pyqtSignal, QTimer
from PyQt6.QtGui import QFont, QPixmap
import threading
import time

# 先测试基本导入
try:
    import matplotlib
    matplotlib.use('Qt5Agg')  # Qt6兼容Qt5Agg后端
    import matplotlib.pyplot as plt
    from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
    from matplotlib.figure import Figure
    MATPLOTLIB_AVAILABLE = True
    print("Matplotlib导入成功")
except Exception as e:
    print(f"Matplotlib导入失败: {e}")
    MATPLOTLIB_AVAILABLE = False

try:
    import sounddevice as sd
    SOUNDDEVICE_AVAILABLE = True
    print("Sounddevice导入成功")
except Exception as e:
    print(f"Sounddevice导入失败: {e}")
    SOUNDDEVICE_AVAILABLE = False

try:
    from scipy.fft import fft, fftfreq
    from scipy.signal import chirp, firwin2, lfilter
    SCIPY_AVAILABLE = True
    print("Scipy导入成功")
except Exception as e:
    print(f"Scipy导入失败: {e}")
    SCIPY_AVAILABLE = False

# 设置中文字体
if MATPLOTLIB_AVAILABLE:
    try:
        plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'DejaVu Sans']
        plt.rcParams['axes.unicode_minus'] = False
    except:
        pass

class TestWorker(QThread):
    """测试工作线程"""
    progress = pyqtSignal(str)  # 进度信息
    finished = pyqtSignal(object, object, object, object, float)  # 测试完成信号
    error = pyqtSignal(str)  # 错误信号
    
    def __init__(self, f0, f1, duration, volume, sample_rate=48000, apply_inverse_eq=False):
        super().__init__()
        self.f0 = f0
        self.f1 = f1
        self.duration = duration
        self.volume = volume
        self.sample_rate = sample_rate
        self.apply_inverse_eq = apply_inverse_eq
    
    def run(self):
        """执行测试"""
        try:
            # 检查依赖
            if not SOUNDDEVICE_AVAILABLE:
                self.error.emit("Sounddevice unavailable, cannot perform audio test")
                return
                
            if not SCIPY_AVAILABLE:
                self.error.emit("Scipy unavailable, cannot perform signal processing")
                return
            
            self.progress.emit("Generating test signal...")
            
            # 生成chirp信号
            signal = self.generate_log_chirp(self.f0, self.f1, self.duration, self.volume)
            
            self.progress.emit("Playing and recording audio...")
            
            # 播放并录制
            recorded = sd.playrec(signal.astype(np.float32), 
                                samplerate=self.sample_rate, 
                                channels=1, 
                                dtype=np.float32)
            
            sd.wait()
            
            self.progress.emit("Analyzing response...")
            
            # 智能去除异常值
            recorded_flat = recorded.flatten()
            clean_recorded = self.remove_initial_outliers(recorded_flat)
            
            # 重新计算FFT（使用清理后的数据）
            fft_length = self.sample_rate  # 1Hz分辨率
            
            # 如果清理后的信号长度不够，进行零填充
            if len(clean_recorded) < fft_length:
                padded_signal = np.pad(clean_recorded, 
                                     (0, fft_length - len(clean_recorded)), 'constant')
            else:
                padded_signal = clean_recorded[:fft_length]
            
            fft_result = fft(padded_signal)
            freqs = fftfreq(fft_length, 1/self.sample_rate)
            
            # 只取正频率部分
            positive_freqs = freqs[:fft_length//2]
            positive_fft = fft_result[:fft_length//2]
            
            # 计算幅度谱
            magnitude_spectrum = np.abs(positive_fft)
            
            # 转换为dB
            magnitude_db = 20 * np.log10(magnitude_spectrum + 1e-10)

            # 如果启用逆向均衡，则基于首次测量结果设计均衡器并进行二次测试
            if self.apply_inverse_eq:
                self.progress.emit("Designing inverse EQ and retesting...")

                # 计算与录音对齐的输入信号（去除相同的前置样本）
                original_length = int(self.duration * self.sample_rate)
                skip_samples = max(0, original_length - len(clean_recorded))

                aligned_input = signal[skip_samples:]
                if len(aligned_input) < fft_length:
                    padded_input = np.pad(aligned_input, (0, fft_length - len(aligned_input)), 'constant')
                else:
                    padded_input = aligned_input[:fft_length]

                fft_input = fft(padded_input)
                positive_input = fft_input[:fft_length//2]

                # 估计传递函数 H = Y/X，加入稳定项避免除零
                eps = 1e-50
                H = positive_fft / (positive_input + eps)
                H_mag = np.abs(H)
                H_mag_db = 20 * np.log10(H_mag + eps)

                # 绘制传递函数幅度 H_mag_db vs 频率（对数横轴），并保存为图片（子线程中不弹窗显示）
                if MATPLOTLIB_AVAILABLE:
                    try:
                        tmp_fig = Figure(figsize=(6, 4))
                        tmp_ax = tmp_fig.add_subplot(111)
                        positive_fft_db = 20 * np.log10(np.abs(positive_fft) + 1e-12)
                        tmp_ax.semilogx(positive_freqs, positive_fft_db, 'k-', linewidth=1)
                        tmp_ax.set_xlabel('Frequency (Hz)')
                        tmp_ax.set_ylabel('Mic FFT (dB)')
                        tmp_ax.set_title('Mic Positive FFT Magnitude')
                        tmp_ax.set_xlim(20, 12000)
                        tmp_ax.grid(True, alpha=0.3)
                        tmp_fig.tight_layout()
                        tmp_fig.savefig('mic_positive_fft_db.png', dpi=120)
                    except Exception as e:
                        print(f"Plot TF error: {e}")

                    # 追加：绘制 positive_input 的正频谱幅度（dB）
                    try:
                        in_fig = Figure(figsize=(6, 4))
                        in_ax = in_fig.add_subplot(111)
                        positive_input_db = 20 * np.log10(np.abs(positive_input) + 1e-12)
                        in_ax.semilogx(positive_freqs, positive_input_db, 'k-', linewidth=1)
                        in_ax.set_xlabel('Frequency (Hz)')
                        in_ax.set_ylabel('Input FFT (dB)')
                        in_ax.set_title('Input Positive FFT Magnitude')
                        in_ax.set_xlim(20, 12000)
                        in_ax.grid(True, alpha=0.3)
                        in_fig.tight_layout()
                        in_fig.savefig('input_positive_fft_db.png', dpi=120)
                    except Exception as e:
                        print(f"Plot input FFT error: {e}")

                # 在20Hz-12kHz范围内计算目标逆增益（dB），限制最大提升/衰减
                band_low = 20.0
                band_high = 12000.0
                inv_gain_db_raw = -H_mag_db
                # 更强的反相能力（注意：如需更大提升，请降低音量避免削顶）
                max_boost_db = 30.0
                max_cut_db = 30.0
                inv_gain_db = np.clip(inv_gain_db_raw, -max_cut_db, max_boost_db)
                inv_gain = 10 ** (inv_gain_db / 20.0)

                # 将目标增益扩展到整个0..fs/2频段（带外设为1）
                dense_freqs = np.linspace(0, self.sample_rate / 2, num=4097)
                inv_gain_full = np.ones_like(dense_freqs)
                # 对带内进行对数频率插值
                band_mask = (positive_freqs >= band_low) & (positive_freqs <= band_high)
                if np.any(band_mask):
                    inv_gain_band = inv_gain[band_mask]
                    freqs_band = positive_freqs[band_mask]
                    log_dense = np.log10(np.clip(dense_freqs, 1.0, None))
                    log_band = np.log10(freqs_band)
                    inv_interp = np.interp(log_dense, log_band, inv_gain_band, left=1.0, right=1.0)
                    # 保留带外为1
                    inv_gain_full = np.where((dense_freqs >= band_low) & (dense_freqs <= band_high), inv_interp, 1.0)

                # 设计FIR均衡滤波器
                num_taps = 4097
                taps = firwin2(num_taps, dense_freqs, inv_gain_full, fs=self.sample_rate)

                # 对输出信号进行滤波（时域），并归一化避免削顶
                corrected_signal = lfilter(taps, [1.0], signal.astype(np.float32))
                # RMS归一化到原始信号以保持总体电平
                def _rms(x: np.ndarray) -> float:
                    return float(np.sqrt(np.mean(np.square(x)) + 1e-12))
                rms_in = _rms(signal)
                rms_out = _rms(corrected_signal)
                if rms_out > 0:
                    corrected_signal = corrected_signal * (rms_in / rms_out)
                # 削顶保护
                peak = float(np.max(np.abs(corrected_signal)) + 1e-12)
                if peak > 0.99:
                    corrected_signal = corrected_signal / peak * 0.99

                self.progress.emit("Playing corrected signal and recording...")
                recorded_corr = sd.playrec(corrected_signal.astype(np.float32),
                                           samplerate=self.sample_rate,
                                           channels=1,
                                           dtype=np.float32)
                sd.wait()

                # 清理并计算校正后的响应
                recorded_corr_flat = recorded_corr.flatten()
                clean_recorded_corr = self.remove_initial_outliers(recorded_corr_flat)

                if len(clean_recorded_corr) < fft_length:
                    padded_corr = np.pad(clean_recorded_corr, (0, fft_length - len(clean_recorded_corr)), 'constant')
                else:
                    padded_corr = clean_recorded_corr[:fft_length]

                fft_corr = fft(padded_corr)
                positive_corr = fft_corr[:fft_length//2]
                magnitude_corr = np.abs(positive_corr)
                magnitude_db_corr = 20 * np.log10(magnitude_corr + 1e-10)

                # 使用校正后的结果进行显示
                self.finished.emit(signal, clean_recorded_corr, positive_freqs, magnitude_db_corr, self.duration)
                return

            # 未启用均衡，直接返回首次测量结果
            self.finished.emit(signal, clean_recorded, positive_freqs, magnitude_db, self.duration)
            
        except Exception as e:
            self.error.emit(f"Error during test: {str(e)}")
    
    def generate_log_chirp(self, f0, f1, duration, amplitude=0.3):
        """生成对数chirp信号"""
        t = np.linspace(0, duration, int(self.sample_rate * duration))
        signal = amplitude * chirp(t, f0=f0, f1=f1, t1=duration, method='logarithmic')
        
        # 添加淡入淡出效果
        fade_samples = int(0.1 * self.sample_rate)  # 100ms淡入淡出
        signal[:fade_samples] *= np.linspace(0, 1, fade_samples)
        signal[-fade_samples:] *= np.linspace(1, 0, fade_samples)
        
        return signal
    
    def remove_initial_outliers(self, signal):
        """智能去除前面异常的大幅值点"""
        try:
            # 计算信号的标准差和均值（使用后半部分稳定信号作为参考）
            stable_start = len(signal) // 4  # 从1/4处开始取稳定部分
            stable_signal = signal[stable_start:]
            std_threshold = np.std(stable_signal)
            mean_threshold = np.mean(np.abs(stable_signal))
            
            # 设置异常值检测阈值（比稳定信号大3倍标准差）
            outlier_threshold = mean_threshold + 3 * std_threshold
            
            # 从前往后扫描，找到第一个非异常值的位置
            skip_samples = 0
            max_skip = min(int(0.2 * self.sample_rate), len(signal) // 2)  # 最多跳过0.2秒或一半信号
            
            for i in range(max_skip):
                if i >= len(signal):
                    break
                # 检查当前点及其邻近点是否都在正常范围内
                window_size = 10
                start_idx = max(0, i)
                end_idx = min(len(signal), i + window_size)
                window_signal = signal[start_idx:end_idx]
                
                # 如果窗口内的最大绝对值都小于阈值，认为找到了稳定区域
                if np.max(np.abs(window_signal)) < outlier_threshold:
                    skip_samples = i
                    break
                skip_samples = i + 1
            
            # 确保至少跳过一些初始样本（最少跳过100个样本）
            skip_samples = max(skip_samples, 100)
            
            # 返回清理后的信号
            if skip_samples < len(signal):
                clean_signal = signal[skip_samples:]
                print(f"Removed {skip_samples} initial outlier samples ({skip_samples/self.sample_rate:.3f}s)")
                return clean_signal
            else:
                print("Warning: Too many outliers detected, using original signal")
                return signal
                
        except Exception as e:
            print(f"Error in outlier removal: {e}")
            # 如果出错，回退到固定跳过4000个点
            skip_samples = min(4000, len(signal) // 2)
            return signal[skip_samples:] if skip_samples < len(signal) else signal

class PlotCanvas(FigureCanvas):
    """绘图画布"""
    def __init__(self, parent=None, width=8, height=4, dpi=100):
        self.fig = Figure(figsize=(width, height), dpi=dpi)
        super().__init__(self.fig)
        self.setParent(parent)
        
        # 创建子图
        self.ax_time = self.fig.add_subplot(211)
        self.ax_freq = self.fig.add_subplot(212)
        
        # 时域图设置
        self.ax_time.set_xlabel('Time (s)')
        self.ax_time.set_ylabel('Amplitude')
        self.ax_time.set_title('Microphone Received Waveform')
        self.ax_time.grid(True, alpha=0.3)
        
        # 频域图设置
        self.ax_freq.set_xlabel('Frequency (Hz)')
        self.ax_freq.set_ylabel('Response (dB)')
        self.ax_freq.set_title('Frequency Response')
        self.ax_freq.grid(True, alpha=0.3)
        self.ax_freq.set_xscale('log')
        self.ax_freq.set_xlim(20, 12000)
        
        # 调整布局
        self.fig.subplots_adjust(left=0.1, right=0.95, top=0.95, bottom=0.1, hspace=0.3)
    
    def smooth_frequency_response(self, freqs, magnitude_db, smoothing_factor=0.2):
        """对频率响应进行对数域平滑
        
        Args:
            freqs: 频率数组
            magnitude_db: 幅度响应(dB)
            smoothing_factor: 平滑因子，值越大平滑程度越高 (0-1)
        
        Returns:
            smoothed_magnitude: 平滑后的幅度响应
        """
        try:
            # 过滤有效频率范围 (>0)
            valid_mask = freqs > 0
            valid_freqs = freqs[valid_mask]
            valid_magnitude = magnitude_db[valid_mask]
            
            if len(valid_freqs) < 3:
                return magnitude_db
            
            # 转换到对数频率域
            log_freqs = np.log10(valid_freqs)
            
            # 计算平滑窗口大小（基于对数频率间隔）
            log_freq_range = log_freqs[-1] - log_freqs[0]
            window_size = max(5, int(len(log_freqs) * smoothing_factor / 5))
            
            # 使用移动平均进行平滑
            smoothed_magnitude = np.copy(valid_magnitude)
            half_window = window_size // 2
            
            for i in range(len(valid_magnitude)):
                start_idx = max(0, i - half_window)
                end_idx = min(len(valid_magnitude), i + half_window + 1)
                
                # 计算加权平均（距离越近权重越大）
                weights = np.exp(-0.5 * np.abs(np.arange(start_idx, end_idx) - i))
                weights = weights / np.sum(weights)
                
                smoothed_magnitude[i] = np.sum(valid_magnitude[start_idx:end_idx] * weights)
            
            # 将结果映射回原始数组
            result = np.copy(magnitude_db)
            result[valid_mask] = smoothed_magnitude
            
            return result
            
        except Exception as e:
            print(f"Smoothing error: {e}")
            return magnitude_db
    
    def update_plot(self, signal, recorded_signal, freqs, magnitude_db, duration, sample_rate=48000):
        """更新图表"""
        try:
            # 计算跳过的样本数
            original_length = int(duration * sample_rate)
            current_length = len(recorded_signal)
            skip_samples = original_length - current_length
            skip_time = skip_samples / sample_rate if skip_samples > 0 else 0
            
            # 计算时间轴
            time_axis = np.linspace(skip_time, duration, len(recorded_signal))
            
            # 更新时域图
            self.ax_time.clear()
            self.ax_time.plot(time_axis, recorded_signal, 'b-', linewidth=1, alpha=0.8)
            self.ax_time.set_xlabel('Time (s)')
            self.ax_time.set_ylabel('Amplitude')
            self.ax_time.set_title(f'Microphone Waveform (Skip {skip_samples} outliers)')
            self.ax_time.grid(True, alpha=0.3)
            
            # 更新频域图
            self.ax_freq.clear()
            
            # 对频率响应进行平滑处理
            smoothed_magnitude_db = self.smooth_frequency_response(freqs, magnitude_db, smoothing_factor=0.35)
            
            # 绘制原始曲线（淡色）和平滑曲线
            self.ax_freq.semilogx(freqs, magnitude_db, 'r-', linewidth=0.5, alpha=0.3, label='Original')
            self.ax_freq.semilogx(freqs, smoothed_magnitude_db, 'r-', linewidth=2, alpha=0.9, label='Smoothed')
            self.ax_freq.grid(True, alpha=0.3)
            self.ax_freq.set_xlabel('Frequency (Hz)')
            self.ax_freq.set_ylabel('Response (dB)')
            self.ax_freq.set_title('Frequency Response (Log Smoothed)')
            self.ax_freq.set_xlim(20, 12000)
            self.ax_freq.legend(loc='upper right', fontsize=8)
            
            # 调整布局
            self.fig.subplots_adjust(left=0.1, right=0.95, top=0.95, bottom=0.1, hspace=0.3)
            
            # 刷新画布
            self.draw()
            
        except Exception as e:
            print(f"Error updating plot: {e}")

class FrequencyResponseQt6(QMainWindow):
    """主窗口类"""
    def __init__(self):
        super().__init__()
        self.sample_rate = 48000
        self.is_testing = False
        self.worker = None
        
        self.init_ui()
        
    def init_ui(self):
        """初始化用户界面"""
        self.setWindowTitle("Speaker Frequency Response Test - Qt6")
        self.setGeometry(100, 100, 900, 700)
        
        # 创建中央部件
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        
        # 主布局
        main_layout = QVBoxLayout(central_widget)
        main_layout.setSpacing(10)
        
        # 标题
        title_label = QLabel("Speaker Frequency Response Test")
        title_font = QFont()
        title_font.setPointSize(16)
        title_font.setBold(True)
        title_label.setFont(title_font)
        title_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        main_layout.addWidget(title_label)
        
        # 状态信息
        status_text = "Ready"
        if not MATPLOTLIB_AVAILABLE:
            status_text += " (Matplotlib unavailable)"
        if not SOUNDDEVICE_AVAILABLE:
            status_text += " (Sounddevice unavailable)"
        if not SCIPY_AVAILABLE:
            status_text += " (Scipy unavailable)"
        
        status_label = QLabel(status_text)
        status_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        main_layout.addWidget(status_label)
        
        # 参数设置组
        param_group = QGroupBox("Test Parameters")
        param_layout = QGridLayout(param_group)
        
        # 第一行：频率和时长
        param_layout.addWidget(QLabel("Start (Hz):"), 0, 0)
        self.f0_edit = QLineEdit("20")
        self.f0_edit.setMaximumWidth(80)
        param_layout.addWidget(self.f0_edit, 0, 1)
        
        param_layout.addWidget(QLabel("End (Hz):"), 0, 2)
        self.f1_edit = QLineEdit("12000")
        self.f1_edit.setMaximumWidth(80)
        param_layout.addWidget(self.f1_edit, 0, 3)
        
        param_layout.addWidget(QLabel("Duration (s):"), 0, 4)
        self.duration_edit = QLineEdit("3.0")
        self.duration_edit.setMaximumWidth(60)
        param_layout.addWidget(self.duration_edit, 0, 5)
        
        # 第二行：音量和按钮
        param_layout.addWidget(QLabel("Volume:"), 1, 0)
        self.volume_slider = QSlider(Qt.Orientation.Horizontal)
        self.volume_slider.setMinimum(0)
        self.volume_slider.setMaximum(100)
        self.volume_slider.setValue(30)
        self.volume_slider.valueChanged.connect(self.update_volume_label)
        param_layout.addWidget(self.volume_slider, 1, 1, 1, 2)
        
        self.volume_label = QLabel("30%")
        param_layout.addWidget(self.volume_label, 1, 3)

        # 逆向均衡开关
        self.inverse_eq_checkbox = QCheckBox("Inverse EQ (20Hz-12kHz)")
        self.inverse_eq_checkbox.setChecked(True)
        param_layout.addWidget(self.inverse_eq_checkbox, 1, 4)
        
        # 开始测试按钮
        self.test_button = QPushButton("Start Test")
        self.test_button.clicked.connect(self.start_test)
        self.test_button.setMinimumHeight(30)
        param_layout.addWidget(self.test_button, 1, 5, 1, 1)
        
        main_layout.addWidget(param_group)
        
        # 状态标签
        self.status_label = QLabel("Ready")
        self.status_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        main_layout.addWidget(self.status_label)
        
        # 进度条
        self.progress_bar = QProgressBar()
        self.progress_bar.setVisible(False)
        main_layout.addWidget(self.progress_bar)
        
        # 图表区域
        if MATPLOTLIB_AVAILABLE:
            self.plot_canvas = PlotCanvas(self, width=8, height=5, dpi=80)
            main_layout.addWidget(self.plot_canvas)
        else:
            no_plot_label = QLabel("Matplotlib unavailable, cannot display charts")
            no_plot_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
            no_plot_label.setStyleSheet("border: 1px solid gray; padding: 50px;")
            main_layout.addWidget(no_plot_label)
        
        # 统计信息组
        stats_group = QGroupBox("Statistics")
        stats_layout = QHBoxLayout(stats_group)
        
        self.max_label = QLabel("Max: --")
        self.min_label = QLabel("Min: --")
        self.avg_label = QLabel("Avg: --")
        self.range_label = QLabel("Range: --")
        
        stats_layout.addWidget(self.max_label)
        stats_layout.addWidget(self.min_label)
        stats_layout.addWidget(self.avg_label)
        stats_layout.addWidget(self.range_label)
        
        main_layout.addWidget(stats_group)
        
        # 设置样式
        self.setStyleSheet("""
            QGroupBox {
                font-weight: bold;
                border: 2px solid gray;
                border-radius: 5px;
                margin-top: 1ex;
                padding-top: 10px;
            }
            QGroupBox::title {
                subcontrol-origin: margin;
                left: 10px;
                padding: 0 5px 0 5px;
            }
            QPushButton {
                background-color: #4CAF50;
                color: white;
                border: none;
                padding: 8px 16px;
                border-radius: 4px;
                font-weight: bold;
            }
            QPushButton:hover {
                background-color: #45a049;
            }
            QPushButton:pressed {
                background-color: #3d8b40;
            }
            QPushButton:disabled {
                background-color: #cccccc;
                color: #666666;
            }
        """)
    
    def update_volume_label(self, value):
        """更新音量标签"""
        self.volume_label.setText(f"{value}%")
    
    def start_test(self):
        """开始测试"""
        if self.is_testing:
            return
        
        try:
            # 获取参数
            f0 = float(self.f0_edit.text())
            f1 = float(self.f1_edit.text())
            duration = float(self.duration_edit.text())
            volume = self.volume_slider.value() / 100.0
            
            # 参数验证
            if f0 <= 0 or f1 <= 0 or f0 >= f1:
                QMessageBox.warning(self, "Parameter Error", "Invalid frequency range")
                return
                
            if duration <= 0 or duration > 10:
                QMessageBox.warning(self, "Parameter Error", "Duration should be between 0-10 seconds")
                return
            
            # 开始测试
            self.is_testing = True
            self.test_button.setEnabled(False)
            self.progress_bar.setVisible(True)
            self.progress_bar.setRange(0, 0)  # 不确定进度
            
            # 创建并启动工作线程
            apply_inverse_eq = self.inverse_eq_checkbox.isChecked()
            self.worker = TestWorker(f0, f1, duration, volume, self.sample_rate, apply_inverse_eq=apply_inverse_eq)
            self.worker.progress.connect(self.update_status)
            self.worker.finished.connect(self.test_finished)
            self.worker.error.connect(self.test_error)
            self.worker.start()
            
        except ValueError:
            QMessageBox.warning(self, "Parameter Error", "Invalid parameter values")
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed to start test: {str(e)}")
    
    def update_status(self, message):
        """更新状态"""
        self.status_label.setText(message)
    
    def test_finished(self, signal, recorded_signal, freqs, magnitude_db, duration):
        """测试完成"""
        try:
            # 更新图表
            if MATPLOTLIB_AVAILABLE:
                self.plot_canvas.update_plot(signal, recorded_signal, freqs, magnitude_db, duration, self.sample_rate)
            
            # 更新统计信息
            self.update_statistics(magnitude_db)
            
            self.status_label.setText("Test completed")
            
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Error processing results: {str(e)}")
        finally:
            self.test_completed()
    
    def test_error(self, error_message):
        """测试错误"""
        QMessageBox.critical(self, "Test Error", error_message)
        self.status_label.setText("Test failed")
        self.test_completed()
    
    def test_completed(self):
        """测试完成清理"""
        self.is_testing = False
        self.test_button.setEnabled(True)
        self.progress_bar.setVisible(False)
        if self.worker:
            self.worker.quit()
            self.worker.wait()
            self.worker = None
    
    def update_statistics(self, magnitude_db):
        """更新统计信息"""
        try:
            max_response = np.max(magnitude_db)
            min_response = np.min(magnitude_db)
            avg_response = np.mean(magnitude_db)
            range_response = max_response - min_response
            
            self.max_label.setText(f"Max: {max_response:.1f}dB")
            self.min_label.setText(f"Min: {min_response:.1f}dB")
            self.avg_label.setText(f"Avg: {avg_response:.1f}dB")
            self.range_label.setText(f"Range: {range_response:.1f}dB")
            
        except Exception as e:
            QMessageBox.warning(self, "Statistics Error", f"Error updating statistics: {str(e)}")
    
    def closeEvent(self, event):
        """关闭事件"""
        if self.is_testing and self.worker:
            reply = QMessageBox.question(self, 'Warning', 
                                       'Test is running. Are you sure you want to exit?',
                                       QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No)
            
            if reply == QMessageBox.StandardButton.Yes:
                if self.worker:
                    self.worker.terminate()
                    self.worker.wait()
                event.accept()
            else:
                event.ignore()
        else:
            event.accept()

def main():
    """主函数"""
    app = QApplication(sys.argv)
    app.setApplicationName("Speaker Frequency Response Test")
    
    # 设置应用图标（如果有的话）
    # app.setWindowIcon(QIcon('icon.png'))
    
    window = FrequencyResponseQt6()
    window.show()
    
    try:
        sys.exit(app.exec())
    except SystemExit:
        pass

if __name__ == "__main__":
    main()
