import sounddevice as sd
from PyQt5 import QtWidgets
from PyQt5.QtGui import QStandardItemModel, QStandardItem
from PyQt5.QtCore import QFile, Qt
from PyQt5.QtWidgets import QDialog, QMessageBox
from PyQt5.uic import loadUi
from control.log_manager import LogManager
from control.utils import utils


class SelectDeviceInterface(QDialog):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.parent = parent
        self.ui = None
        self.signal_info = None
        self.logger = LogManager.set_log_handler("选择输出设备")
        self.init_ui()
        self.device_list = []
        self.select_devices()
        self.save_all_button.clicked.connect(self.save_all_device_settings)
        self.comboBox.setMinimumWidth(300)
        self.comboBox_2.setMinimumWidth(300)

    def init_ui(self):
        ui_file = QFile(":ui/select_device.ui")
        if not ui_file.exists():
            self.logger.error("未找到资源文件 select_device.ui")
            raise FileNotFoundError("未找到资源文件 select_device.ui")
        ui_file.open(QFile.ReadOnly)
        self.ui = loadUi(ui_file, self)
        ui_file.close()
        self.setWindowTitle("选择输出设备")
        # 最小化
        self.setWindowFlags(self.windowFlags() | Qt.WindowMinimizeButtonHint)
        self.show()
        self.logger.info("打开选择输出设备界面")

    def select_devices(self):
        self.device_list = sd.query_devices()
        hostapis = sd.query_hostapis()

        # 输出设备
        self.comboBox.clear()
        default_output_index = None
        try:
            default_output_index = sd.default.device[1]
        except Exception as e:
            self.logger.warning(f"无法获取默认输出设备: {e}")

        # 输入设备
        self.comboBox_2.clear()
        default_input_index = None
        try:
            default_input_index = sd.default.device[0]
        except Exception as e:
            self.logger.warning(f"无法获取默认输入设备: {e}")

        model = QStandardItemModel()
        model.setHorizontalHeaderLabels(["序号", "名称", "输入通道", "输出通道", "驱动类型", "采样率"])

        for index, device in enumerate(self.device_list):
            hostapi_index = device['hostapi']
            hostapi_name = hostapis[hostapi_index]['name']
            display_name = f"{device['name']} ({hostapi_name})"
            samplerate = device.get("default_samplerate", "N/A")
            if isinstance(samplerate, float):
                samplerate = f"{int(samplerate)} Hz"

            # 表格显示所有设备
            row_items = [
                QStandardItem(str(index)),
                QStandardItem(display_name),
                QStandardItem(str(device["max_input_channels"])),
                QStandardItem(str(device["max_output_channels"])),
                QStandardItem(hostapi_name),
                QStandardItem(str(samplerate))
            ]
            for item in row_items:
                item.setEditable(False)
                item.setTextAlignment(Qt.AlignCenter)
            model.appendRow(row_items)

            # 输出设备填充
            if device["max_output_channels"] > 0:
                self.comboBox.addItem(f"{index}: {display_name}", index)

            # 输入设备填充
            if device["max_input_channels"] > 0:
                self.comboBox_2.addItem(f"{index}: {display_name}", index)

        # 设置默认选中
        for i in range(self.comboBox.count()):
            if self.comboBox.itemData(i) == default_output_index:
                self.comboBox.setCurrentIndex(i)
                break

        for i in range(self.comboBox_2.count()):
            if self.comboBox_2.itemData(i) == default_input_index:
                self.comboBox_2.setCurrentIndex(i)
                break

        self.tableView.setModel(model)
        self.tableView.resizeColumnsToContents()
        self.tableView.setSelectionBehavior(QtWidgets.QAbstractItemView.SelectRows)
        self.tableView.setEditTriggers(QtWidgets.QAbstractItemView.NoEditTriggers)
        self.tableView.horizontalHeader().setStretchLastSection(True)
        self.tableView.verticalHeader().setVisible(False)

    def save_all_device_settings(self):
        output_index = self.comboBox.currentData()
        input_index = self.comboBox_2.currentData()

        if input_index is None or output_index is None:
            QMessageBox.warning(self, "警告", "请选择输入和输出设备")
            return

        try:
            sd.default.device = (input_index, output_index)
            self.logger.info(f"设置默认输入设备为: {self.device_list[input_index]['name']}")
            self.logger.info(f"设置默认输出设备为: {self.device_list[output_index]['name']}")

            # 保存驱动信息至config
            hostapis = sd.query_hostapis()
            input_device = self.device_list[input_index]
            output_device = self.device_list[output_index]

            input_device_index = input_device['index']
            output_device_index = output_device['index']
            print(input_device_index, output_device_index)

            input_device_name = input_device['name']
            output_device_name = output_device['name']

            input_type = hostapis[input_device['hostapi']]['name']
            output_type = hostapis[output_device['hostapi']]['name']

            input_samplerate = str(int(input_device.get('default_samplerate', 0)))
            output_samplerate = str(int(output_device.get('default_samplerate', 0)))

            # 获取输入设备的通道数
            input_channel_count = input_device.get('max_input_channels', 0)

            utils.save_selected_devices(input_device_index, output_device_index, input_device_name, output_device_name,
                                        input_type, output_type, input_samplerate, output_samplerate,
                                        input_channel_count)
            QMessageBox.information(self, "成功", "输入输出设备保存成功！")
        except Exception as e:
            self.logger.error(f"设置默认输入输出设备失败: {e}")
            QMessageBox.critical(self, "错误", f"设置默认输入输出设备失败: {e}")

    def closeEvent(self, event):
        if self.parent:
            self.parent.device_window = None
        self.logger.info("关闭选择输出设备界面")
        event.accept()