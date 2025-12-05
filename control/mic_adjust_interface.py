import json
import os

import pyqtgraph
import numpy as np
import sounddevice as sd

from PyQt5.QtCore import QFile, Qt
from PyQt5.QtGui import QPixmap
from PyQt5.QtWidgets import QDialog, QGraphicsScene, QGraphicsPixmapItem, QMessageBox, QVBoxLayout, QApplication
from PyQt5.uic import loadUi
from pyqtgraph import mkPen

from control.log_manager import LogManager
from control.utils import utils
from control.utils.audio_thd_frequency_response_analysis import AudioThdFrequencyResponseAnalysis
from custom.customSignals import sign

class MicAdjustInterface(QDialog):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.parent = parent
        self.ui = None
        self.signal_info = None
        self.mic1_pa = None
        self.mic2_pa = None
        self.logger = LogManager.set_log_handler("麦克风校准")
        self.init_ui()
        self.init_fun()
        self.init_voltage_value()

    def init_ui(self):
        ui_file = QFile(":ui/mic_adjust.ui")
        if not ui_file.exists():
            self.logger.error("未找到资源文件 mic_adjust.ui")
            raise FileNotFoundError("未找到资源文件 mic_adjust.ui")
        ui_file.open(QFile.ReadOnly)
        self.ui = loadUi(ui_file, self)
        ui_file.close()
        self.setWindowTitle("麦克风校准")
        self.setWindowFlags(self.windowFlags() | Qt.WindowMinimizeButtonHint| Qt.WindowMaximizeButtonHint)
        self.init_images()
        self.init_view()
        self.show()
        self.logger.info("打开麦克风校准界面")

    def init_fun(self):
        self.start_adjust_button.clicked.connect(self.start_adjust)
        # 连接失去焦点事件
        self.target_voltage_value.focusOutEvent = self.focusOutEvent  # 将自定义的事件绑定到控件上

        sign.play_adjust_audio_sign.connect(self.play_adjust_audio,Qt.AutoConnection)
        sign.error_message_signal.connect(utils.show_error_message)
        sign.update_plot_sign.connect(self.update_plot)

    def init_voltage_value(self):
        ok, config = utils.get_config_content("output_signal_setting.json")
        if not ok:
            self.logger.error("读取配置失败: output_signal_setting.json")
            QMessageBox.warning(self, "错误", "读取配置失败：output_signal_setting.json")
            return
        mic_adj_v = config.get("mic_adjust_voltage", None)
        self.target_voltage_value.setText(str(mic_adj_v if mic_adj_v is not None else "0.01"))

    def focusOutEvent(self, event):
        """
        当控件失去焦点时触发，进行电压值的校验和保存操作。
        """
        self.validate_and_save_voltage()  # 触发校验和保存操作

        # 调用父类的 focusOutEvent（如果需要其他默认行为）
        super().focusOutEvent(event)

    def validate_and_save_voltage(self):
        text = self.target_voltage_value.text().strip()

        try:
            value = float(text)
            if value < 0:
                raise ValueError("电压必须大于 0")

            ok, config = utils.get_config_content("output_signal_setting.json")
            if ok:
                config["mic_adjust_voltage"] = value
                if utils.write_config_content("output_signal_setting.json", config):
                    self.logger.info(f"保存 mic_adjust_voltage = {value}")
                else:
                    QMessageBox.warning(self, "保存失败", "配置文件保存失败，请重试！")
            else:
                QMessageBox.warning(self, "读取失败", "配置文件读取失败，请检查文件是否存在！")

        except Exception as e:
            # 回退到上次保存的合法值
            QMessageBox.warning(self, "输入错误", f"请输入正确的有效数字: {e}")
            ok, cfg = utils.get_config_content("output_signal_setting.json")
            if ok:
                last_value = cfg.get("mic_adjust_voltage", 0.01)  # 默认值为 0.01
                self.target_voltage_value.setText(str(last_value))
            else:
                self.target_voltage_value.setText("0.01")

    def init_images(self):
        scene = QGraphicsScene()
        pixmap = QPixmap(":/images/2mic薄层校准_画板.png")
        scaled_pixmap = pixmap.scaled(400, 300, Qt.KeepAspectRatio, Qt.SmoothTransformation)
        pixmap_item = QGraphicsPixmapItem(scaled_pixmap)
        scene.addItem(pixmap_item)
        self.graphicsView.setScene(scene)

        scene_2 = QGraphicsScene()
        pixmap_2 = QPixmap(":/images/2mic薄层_画板.png")
        scaled_pixmap_2 = pixmap_2.scaled(400, 300, Qt.KeepAspectRatio, Qt.SmoothTransformation)
        pixmap_item_2 = QGraphicsPixmapItem(scaled_pixmap_2)
        scene_2.addItem(pixmap_item_2)
        self.graphicsView_2.setScene(scene_2)

    def init_view(self):
        # Mic1 - Time
        self.plot3 = pyqtgraph.PlotWidget()
        layout3 = QVBoxLayout()
        layout3.setContentsMargins(0, 0, 0, 0)
        layout3.addWidget(self.plot3)
        self.ui.frame_plot3.setLayout(layout3)
        self.plot3.setBackground('white')
        self.plot3.setLabel('bottom', 'Time', units='s')
        self.plot3.setLabel('left', 'Amplitude', units='Pa')
        self.plot3.getAxis('bottom').setTextPen("black")
        self.plot3.getAxis('left').setTextPen("black")
        self.plot3.showGrid(x=True, y=True, alpha=0.25)

        # Mic1 - Freq
        self.plot3_freq = pyqtgraph.PlotWidget()
        lay3f = QVBoxLayout()
        lay3f.setContentsMargins(0, 0, 0, 0)
        lay3f.addWidget(self.plot3_freq)
        self.ui.frame_plot3_2.setLayout(lay3f)
        self.plot3_freq.setBackground('white')
        self.plot3_freq.setLabel('bottom', 'Frequency(Hz)')
        self.plot3_freq.setLabel('left', 'Magnitude')
        self.plot3_freq.getAxis('bottom').enableAutoSIPrefix(False)
        self.plot3_freq.getAxis('left').enableAutoSIPrefix(False)  # 禁用自动转换单位
        self.plot3_freq.getAxis('bottom').setTextPen("black")
        self.plot3_freq.getAxis('left').setTextPen("black")
        self.plot3_freq.showGrid(x=True, y=True, alpha=0.25)
        self.plot3_freq.setLogMode(x=True, y=False)
        # 调整刻度字体大小
        font = pyqtgraph.QtGui.QFont()
        font.setPointSize(11)
        self.plot3_freq.getAxis('bottom').setTickFont(font)
        # 使用自定义的对数坐标轴刻度标签格式
        self.plot3_freq.getAxis('bottom').logTickStrings = utils.custom_log_tick_strings

        # Mic2 - Time
        self.plot4 = pyqtgraph.PlotWidget()
        layout4 = QVBoxLayout()
        layout4.setContentsMargins(0, 0, 0, 0)
        layout4.addWidget(self.plot4)
        self.ui.frame_plot4.setLayout(layout4)
        self.plot4.setBackground('white')
        self.plot4.setLabel('bottom', 'Time', units='s')
        self.plot4.setLabel('left', 'Amplitude', units='Pa')
        self.plot4.getAxis('bottom').setTextPen("black")
        self.plot4.getAxis('left').setTextPen("black")
        self.plot4.showGrid(x=True, y=True, alpha=0.25)

        # Mic2 - Freq
        self.plot4_freq = pyqtgraph.PlotWidget()
        lay4f = QVBoxLayout()
        lay4f.setContentsMargins(0, 0, 0, 0)
        lay4f.addWidget(self.plot4_freq)
        self.ui.frame_plot4_2.setLayout(lay4f)
        self.plot4_freq.setBackground('white')
        self.plot4_freq.setLabel('bottom', 'Frequency(Hz)')
        self.plot4_freq.setLabel('left', 'Magnitude')
        self.plot4_freq.getAxis('bottom').enableAutoSIPrefix(False)
        self.plot4_freq.getAxis('left').enableAutoSIPrefix(False)  # 禁用自动转换单位
        self.plot4_freq.getAxis('bottom').setTextPen("black")
        self.plot4_freq.getAxis('left').setTextPen("black")
        self.plot4_freq.showGrid(x=True, y=True, alpha=0.25)
        self.plot4_freq.setLogMode(x=True, y=False)
        # 调整刻度字体大小
        self.plot4_freq.getAxis('bottom').setTickFont(font)
        # 使用自定义的对数坐标轴刻度标签格式
        self.plot4_freq.getAxis('bottom').logTickStrings = utils.custom_log_tick_strings

    def init_config(self):
        config_path = utils.get_config_path("output_signal_setting.json")
        try:
            with open(config_path, "r", encoding="utf-8") as f:
                config = json.load(f)
                self.signal_info = config.get("signal_info", {})
        except Exception as e:
            self.logger.error(f"读取配置失败: {e}")
            QMessageBox.warning(self, "错误", f"读取配置失败：{e}")

    def record_and_plot(self):
        try:
            self.plot3.clear()
            self.plot4.clear()
            self.plot3_freq.clear()
            self.plot4_freq.clear()
            QApplication.processEvents()
            # 获取设备采样率
            samplerate = utils.get_device_info()
            duration = self.signal_info['signal_time']

            # 根据目标电压，校准并生成激励信号
            target_voltage = float(self.target_voltage_value.text())
            self.signal_info['signal_amplitude'] = target_voltage
            result, data, amplitude, msg = utils.generate_calibrated_signal(self.signal_info, samplerate)

            if result is False:
                raise ValueError(f"{msg}")

            input_channels = sd.query_devices(sd.default.device[0], 'input')['max_input_channels']
            output_channels = sd.query_devices(sd.default.device[1], 'output')['max_output_channels']
            print(f"最大输入通道数: {input_channels},输出通道: {output_channels}")
            self.logger.info(f"最大输入通道数: {input_channels},输出通道: {output_channels}")

            if input_channels < 2:
                raise ValueError(f"检测到的输入通道数为 {input_channels},不足 2 个，无法进行完整校准，请选择别的设备！")

            # 把激励信号data转化成 多维数组
            stimulus_data = np.asarray(data, dtype=np.float32)
            if output_channels > 1:
                stimulus_data = np.column_stack([stimulus_data] * output_channels)

            self.logger.info("开始播放并录音...")
            # stimulus_data代表的是输出的2维激励信号, channels代表的是录音时的输入通道数,这里是 6 个通道
            recording = sd.playrec(stimulus_data, samplerate=samplerate, channels=input_channels, blocking=True)
            self.logger.info("录音完成")

            # 处理录音数据
            mic1_data, mic2_data, real_spl1, real_spl2, scale1, scale2 = self.process_mic_channels_data(recording)
            self.mic1_pa = mic1_data * scale1
            self.mic2_pa = mic2_data * scale2
            if real_spl1 > 120:
                self.logger.error(f"麦克风1实际声压级 {real_spl1:.2f} dB, 超过阈值 {120} dB，已饱和")
                QMessageBox.warning(self, "警告", f"麦克风1实际声压级 {real_spl1:.2f} dB, 超过阈值 {120} dB，已饱和")
            if real_spl2 > 120:
                self.logger.error(f"麦克风2实际声压级 {real_spl2:.2f} dB, 超过阈值 {120} dB，已饱和")
                QMessageBox.warning(self, "警告", f"麦克风2实际声压级 {real_spl2:.2f} dB, 超过阈值 {120} dB，已饱和")

            # 创建校准保存目录
            cal_dir = os.path.join(os.getcwd(), "！校准")
            os.makedirs(cal_dir, exist_ok=True)
            mic1_txt = os.path.join(cal_dir, "MIC1.txt")
            mic2_txt = os.path.join(cal_dir, "MIC2.txt")
            np.savetxt(mic1_txt, mic1_data)
            np.savetxt(mic2_txt, mic2_data)
            self.logger.info(f"MIC1 校准已保存: {mic1_txt}")
            self.logger.info(f"MIC2 校准已保存: {mic2_txt}")

            time_axis = np.linspace(0, duration, len(self.mic1_pa))
            sign.update_plot_sign.emit(time_axis, self.mic1_pa, self.mic2_pa, samplerate)
            self.logger.info(f"画图完成！")
            utils.set_adjust_button_enabled(self.start_adjust_button, True)
        except Exception as e:
            self.logger.error(f"录音或保存失败: {e}")
            sign.error_message_signal.emit(f"录音或保存失败：{e}", self)
            utils.set_adjust_button_enabled(self.start_adjust_button, True)

    @staticmethod
    def process_mic_channels_data(recording):
        """
        处理麦克风通道数据
        """
        channel_config_path = utils.get_config_path("mic_calibration.json")
        try:
            with open(channel_config_path, "r", encoding="utf-8") as f:
                config = json.load(f)
                Mic1_deviation_value = float(config.get("In1", {}).get("deviation_value", {}))
                Mic2_deviation_value = float(config.get("In2", {}).get("deviation_value", {}))
                # 获取通道对应的数组下标，可能因为选择通道顺序而改变原来的  比如 原来的1,2,3,4  变成 2,1,3,4
                binding_index_1 = int(config.get("In1", {}).get("binding_index", 999))
                binding_index_2 = int(config.get("In2", {}).get("binding_index", 999))
                if binding_index_1 == 999:
                    raise ValueError("In1 没有绑定数据下标")
                if binding_index_2 == 999:
                    raise ValueError("In2 没有绑定数据下标")
        except Exception as e:
            raise ValueError(f"获取通道对应的位置index失败,原因：{e}")

        mic1_data = recording[:, binding_index_1]
        mic2_data = recording[:, binding_index_2]
        spl_smooth1 = AudioThdFrequencyResponseAnalysis.spl_calculation(recording[:, binding_index_1])
        real_1 = np.max(spl_smooth1)
        spl_smooth2 = AudioThdFrequencyResponseAnalysis.spl_calculation(recording[:, binding_index_2])
        real_2 = np.max(spl_smooth2)
        real_spl1 = real_1 + Mic1_deviation_value
        real_spl2 = real_2 + Mic2_deviation_value
        rms1 = utils.calculate_rms(recording[:, binding_index_1])
        rms2 = utils.calculate_rms(recording[:, binding_index_2])
        scale1 = utils.calculate_scale(real_spl1, rms1)
        scale2 = utils.calculate_scale(real_spl2, rms2)
        print(f"RMS: {rms1:.8f}, {rms2:.8f}")
        print(f"Scale: {scale1:.2f}, {scale2:.2f}")
        return mic1_data, mic2_data, real_spl1, real_spl2, scale1, scale2

    def start_adjust(self):
        utils.set_adjust_button_enabled(self.start_adjust_button, False)
        self.init_config()
        sign.play_adjust_audio_sign.emit()

    def play_adjust_audio(self):
        self.record_and_plot()

    def update_plot(self, time, mic1_data, mic2_data, samplerate):
        self.plot3.plot(time, mic1_data, pen=mkPen(color=(200, 200, 200)))
        self.plot4.plot(time, mic2_data, pen='gray')
        if samplerate is not None:
            f1, m1 = utils.compute_fft(mic1_data, samplerate)
            f2, m2 = utils.compute_fft(mic2_data, samplerate)

            # 只显示到 Nyquist（rfft 本身就是正频段）
            self.plot3_freq.plot(f1, m1, pen=mkPen(color=(200, 200, 200)))
            self.plot4_freq.plot(f2, m2, pen='gray')

            self.plot3.plotItem.enableAutoRange(axis='xy', enable=True)  # X 和 Y 轴的自动范围调整
            self.plot4.plotItem.enableAutoRange(axis='xy', enable=True)  # X 和 Y 轴的自动范围调整
            self.plot3_freq.plotItem.enableAutoRange(axis='xy', enable=True)  # X 和 Y 轴的自动范围调整
            self.plot4_freq.plotItem.enableAutoRange(axis='xy', enable=True)  # X 和 Y 轴的自动范围调整

    def closeEvent(self, event):
        if self.parent:
            self.parent.mic_window = None
        sign.update_plot_sign.disconnect(self.update_plot)
        sign.error_message_signal.disconnect(utils.show_error_message)
        sign.play_adjust_audio_sign.disconnect(self.play_adjust_audio)
        self.logger.info("关闭麦克风校准界面")
        event.accept()