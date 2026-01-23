import os
import sys
import threading
from typing import Dict, Any, Tuple, Optional

import yaml
import numpy as np
import sounddevice as sd
from PyQt6.QtCore import Qt, QObject, pyqtSignal
from PyQt6.QtWidgets import (
    QApplication,
    QWidget,
    QMainWindow,
    QFileDialog,
    QVBoxLayout,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QPushButton,
    QComboBox,
    QDoubleSpinBox,
    QGroupBox,
    QFormLayout,
    QMessageBox,
    QScrollArea,
    QGridLayout,
)

from cfg_filter_interface import (
    generate_log_chirp,
    apply_custom_weighting,
    play_and_record,
    design_custom_weighting_fir,
)
from scipy.signal import spectrogram, freqz
import matplotlib.pyplot as plt


def load_yaml_file(path: str) -> Dict[str, Any]:
    if not path or not os.path.exists(path):
        return {}
    with open(path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
    return data or {}


def save_yaml_file(path: str, data: Dict[str, Any]) -> None:
    with open(path, "w", encoding="utf-8") as f:
        yaml.safe_dump(data, f, allow_unicode=True, sort_keys=False)


PSEUDO_ASIO_INDEX = -1
PSEUDO_ASIO_NAME = "ASIO (detected)"


def _refresh_portaudio() -> None:
    try:
        sd._terminate()  # type: ignore[attr-defined]
        sd._initialize()  # type: ignore[attr-defined]
    except Exception:
        pass


def get_hostapis() -> list:
    # 尝试刷新，部分环境中可让 ASIO 正常枚举
    _refresh_portaudio()
    try:
        info = sd.query_hostapis()
    except Exception:
        info = []
    # list of (hostapi_index, hostapi_name, hostapi_info)
    result = []
    for idx, h in enumerate(info):
        name = h.get("name", f"hostapi-{idx}")
        result.append((idx, name, h))
    # 如果没有发现 ASIO，但设备名中包含 ASIO，增加一个伪 HostAPI 选项
    try:
        devices = sd.query_devices()
        has_asio_api = any("ASIO" in name.upper() for _, name, _ in result)
        has_asio_device = any("ASIO" in str(d.get("name", "")).upper() for d in devices)
        if (not has_asio_api) and has_asio_device:
            result.append((PSEUDO_ASIO_INDEX, PSEUDO_ASIO_NAME, {}))
    except Exception:
        pass
    return result


def get_device_list(hostapi_index: Optional[int] = None) -> Tuple[list, list]:
    devices = sd.query_devices()
    out_list = []
    in_list = []
    for idx, d in enumerate(devices):
        if hostapi_index is not None:
            if hostapi_index == PSEUDO_ASIO_INDEX:
                if "ASIO" not in str(d.get("name", "")).upper():
                    continue
            else:
                if d.get("hostapi") != hostapi_index:
                    continue
        name = f"{idx}: {d.get('name','Unknown')}"
        max_in = int(d.get("max_input_channels", 0) or 0)
        max_out = int(d.get("max_output_channels", 0) or 0)
        if max_out > 0:
            out_list.append((idx, name, d))
        if max_in > 0:
            in_list.append((idx, name, d))
    return out_list, in_list


class Worker(QObject):
    finished = pyqtSignal()
    error = pyqtSignal(str)
    result = pyqtSignal(object)

    def __init__(self, cfg_path: str, fs: float, duration: float, f0: float, f1: float, channels: int,
                 out_dev: Optional[int], in_dev: Optional[int]) -> None:
        super().__init__()
        self.cfg_path = cfg_path
        self.fs = fs
        self.duration = duration
        self.f0 = f0
        self.f1 = f1
        self.channels = channels
        self.out_dev = out_dev
        self.in_dev = in_dev

    def run(self) -> None:
        try:
            if self.out_dev is not None or self.in_dev is not None:
                # sounddevice expects (input, output)
                sd.default.device = (self.in_dev if self.in_dev is not None else None,
                                     self.out_dev if self.out_dev is not None else None)
            sd.default.samplerate = self.fs

            # 加载 CFG
            with open(self.cfg_path, "r", encoding="utf-8") as f:
                cfg = yaml.safe_load(f) or {}

            # 生成 sweep 与 custom
            sweep, _ = generate_log_chirp(self.f0, self.f1, self.duration, self.fs, amplitude=0.3)
            sweep_custom = apply_custom_weighting(
                sweep,
                self.fs,
                low_boost_db=cfg.get("low_boost_db", 0.0),
                mid_cut_db=cfg.get("mid_cut_db", 0.0),
                high_boost_db=cfg.get("high_boost_db", 0.0),
                f_low1=cfg.get("f_low1", 20.0),
                f_high1=cfg.get("f_high1", 600.0),
                f_low2=cfg.get("f_low2", 600.0),
                f_high2=cfg.get("f_high2", 1000.0),
                f_low3=cfg.get("f_low3", 1000.0),
                f_high3=cfg.get("f_high3", 10000.0),
                zero_phase=True,
            )

            # 两次录音
            recorded_raw = play_and_record(sweep, int(self.fs), channels=int(self.channels))
            sd.wait()
            recorded_custom = play_and_record(sweep_custom, int(self.fs), channels=int(self.channels))
            sd.wait()

            # 传回主线程绘图的数据（单通道在主线程取）
            self.result.emit(
                {
                    "fs": float(self.fs),
                    "f0": float(self.f0),
                    "f1": float(self.f1),
                    "duration": float(self.duration),
                    "raw": recorded_raw,
                    "custom": recorded_custom,
                    "sweep_raw": sweep,              # 计算机端生成的原始扫频信号
                    "sweep_custom": sweep_custom,   # 计算机端生成的调制后信号
                }
            )
            self.finished.emit()
        except Exception as e:
            self.error.emit(str(e))


class NoWheelDoubleSpinBox(QDoubleSpinBox):
    def wheelEvent(self, event) -> None:
        event.ignore()


class NoWheelComboBox(QComboBox):
    def wheelEvent(self, event) -> None:
        event.ignore()


class CfgEditor(QWidget):
    def __init__(self, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self._fields: Dict[str, QLineEdit] = {}
        self._data: Dict[str, Any] = {}

        self.grid = QGridLayout()
        self.grid.setAlignment(Qt.AlignmentFlag.AlignTop | Qt.AlignmentFlag.AlignLeft)
        self.grid.setHorizontalSpacing(14)
        self.grid.setVerticalSpacing(10)

        container = QWidget()
        container.setLayout(self.grid)

        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setWidget(container)

        layout = QVBoxLayout()
        layout.setContentsMargins(8, 8, 8, 8)
        layout.addWidget(scroll)
        self.setLayout(layout)

    def load_cfg(self, data: Dict[str, Any]) -> None:
        # reset
        self._fields.clear()
        self._data = dict(data) if data else {}
        # clear layout
        while self.grid.count():
            item = self.grid.takeAt(0)
            w = item.widget()
            if w is not None:
                w.deleteLater()
        # only show flat key-values (ignore nested objects), two columns per row
        flat_items = [(k, v) for k, v in self._data.items() if not isinstance(v, (dict, list))]
        row = 0
        col_pair = 0  # 0 or 1
        for key, value in flat_items:
            if isinstance(value, (dict, list)):
                continue
            edit = QLineEdit(self)
            edit.setText(str(value))
            edit.setMaximumWidth(140)
            self._fields[key] = edit
            label = QLabel(key)
            # place: (row, col*2)=label, (row, col*2+1)=editor
            start_col = 2 * col_pair
            self.grid.addWidget(label, row, start_col)
            self.grid.addWidget(edit, row, start_col + 1)
            # next slot
            if col_pair == 0:
                col_pair = 1
            else:
                col_pair = 0
                row += 1

    def dump_cfg(self) -> Dict[str, Any]:
        result: Dict[str, Any] = {}
        # keep only edited flat keys, preserve original nested data
        for key, edit in self._fields.items():
            text = edit.text().strip()
            parsed: Any = text
            # try cast to int/float/bool
            for caster in (self._to_int, self._to_float, self._to_bool):
                v = caster(text)
                if v is not None:
                    parsed = v
                    break
            result[key] = parsed
        # preserve untouched nested entries
        for key, value in self._data.items():
            if key not in result:
                result[key] = value
        return result

    @staticmethod
    def _to_int(s: str) -> Optional[int]:
        try:
            if s.lower().startswith("0x"):
                return int(s, 16)
            return int(s)
        except Exception:
            return None

    @staticmethod
    def _to_float(s: str) -> Optional[float]:
        try:
            return float(s)
        except Exception:
            return None

    @staticmethod
    def _to_bool(s: str) -> Optional[bool]:
        v = s.lower()
        if v in ("true", "false"):
            return v == "true"
        return None


class MainWindow(QMainWindow):
    def __init__(self) -> None:
        super().__init__()
        self.setWindowTitle("Speaker Sound Shaper")
        self.setMinimumSize(760, 560)

        self.cfg_path_edit = QLineEdit(self)
        self.btn_browse_cfg = QPushButton("选择...", self)
        self.btn_load_cfg = QPushButton("加载配置", self)
        self.btn_save_cfg = QPushButton("保存配置(覆盖)", self)

        # cfg editor
        self.cfg_editor = CfgEditor(self)

        # test signal params
        self.f0_spin = NoWheelDoubleSpinBox(self)
        self.f1_spin = NoWheelDoubleSpinBox(self)
        self.duration_spin = NoWheelDoubleSpinBox(self)
        for sp in (self.f0_spin, self.f1_spin):
            sp.setRange(1.0, 192000.0)
            sp.setDecimals(2)
        self.duration_spin.setRange(0.1, 1200.0)
        self.duration_spin.setDecimals(2)
        self.f0_spin.setValue(20.0)
        self.f1_spin.setValue(15000.0)
        self.duration_spin.setValue(5.0)
        # 精简输入宽度并增加单位
        for sp in (self.f0_spin, self.f1_spin):
            sp.setMaximumWidth(100)
            sp.setSuffix(" Hz")
        self.duration_spin.setMaximumWidth(100)
        self.duration_spin.setSuffix(" s")

        # devices & hostapi
        self.api_combo = NoWheelComboBox(self)
        self.out_combo = NoWheelComboBox(self)
        self.in_combo = NoWheelComboBox(self)
        # no explicit fs/channels controls per requirement

        self.btn_run = QPushButton("开始测试", self)
        self.btn_save_test_params = QPushButton("保存测试参数", self)

        central = QWidget(self)
        self.setCentralWidget(central)
        root_layout = QVBoxLayout()
        root_layout.setContentsMargins(12, 12, 12, 12)
        root_layout.setSpacing(10)
        central.setLayout(root_layout)

        # cfg group
        cfg_group = QGroupBox("配置文件", self)
        cfg_layout = QVBoxLayout()
        cfg_layout.setContentsMargins(10, 10, 10, 10)
        cfg_layout.setSpacing(8)
        path_row = QHBoxLayout()
        path_row.setSpacing(8)
        path_row.addWidget(QLabel("路径:", self))
        path_row.addWidget(self.cfg_path_edit, 1)
        path_row.addWidget(self.btn_browse_cfg)
        path_row.addWidget(self.btn_load_cfg)
        path_row.addWidget(self.btn_save_cfg)
        cfg_layout.addLayout(path_row)
        cfg_layout.addWidget(self.cfg_editor, 1)
        cfg_group.setLayout(cfg_layout)

        # test signal group
        sig_group = QGroupBox("测试信号", self)
        # 横向紧凑排列
        sig_row = QHBoxLayout()
        sig_row.setSpacing(12)
        # helper to add label+spin inline
        def add_pair(text: str, spin: NoWheelDoubleSpinBox) -> None:
            sub = QHBoxLayout()
            sub.setSpacing(6)
            lab = QLabel(text, self)
            sub.addWidget(lab)
            sub.addWidget(spin)
            sig_row.addLayout(sub)
        add_pair("f0 (Hz):", self.f0_spin)
        add_pair("f1 (Hz):", self.f1_spin)
        add_pair("duration (s):", self.duration_spin)
        sig_layout = QVBoxLayout()
        sig_layout.setContentsMargins(10, 10, 10, 10)
        sig_layout.setSpacing(8)
        sig_layout.addLayout(sig_row)
        sig_group.setLayout(sig_layout)

        # device group
        dev_group = QGroupBox("音频设备", self)
        dev_form = QFormLayout()
        dev_form.setLabelAlignment(Qt.AlignmentFlag.AlignRight)
        dev_form.setFormAlignment(Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignTop)
        dev_form.setHorizontalSpacing(12)
        dev_form.setVerticalSpacing(8)
        dev_form.addRow("音频API:", self.api_combo)
        dev_form.addRow("输出设备:", self.out_combo)
        dev_form.addRow("输入设备:", self.in_combo)
        dev_layout = QVBoxLayout()
        dev_layout.setContentsMargins(10, 10, 10, 10)
        dev_layout.setSpacing(8)
        dev_layout.addLayout(dev_form)
        dev_btn_row = QHBoxLayout()
        dev_btn_row.addStretch(1)
        dev_btn_row.addWidget(self.btn_save_test_params)
        dev_layout.addLayout(dev_btn_row)
        dev_group.setLayout(dev_layout)

        # actions
        action_row = QHBoxLayout()
        action_row.addStretch(1)
        action_row.addWidget(self.btn_run)

        root_layout.addWidget(cfg_group, 3)
        root_layout.addWidget(sig_group, 0)
        root_layout.addWidget(dev_group, 0)
        root_layout.addLayout(action_row)

        self._out_devices: list = []
        self._in_devices: list = []
        self._hostapis: list = []
        self._current_hostapi_index: Optional[int] = None
        self._load_hostapis()
        self._load_devices()
        self._wire_events()
        self._restore_from_defaults()

        # 轻量配色样式（整体美化）
        self.setStyleSheet("""
        QMainWindow { background: #f7f8fa; }
        QGroupBox {
            border: 1px solid #d0d5dd;
            border-radius: 8px;
            margin-top: 12px;
            background: #ffffff;
        }
        QGroupBox::title {
            subcontrol-origin: margin;
            subcontrol-position: top left;
            padding: 0 6px;
            color: #34495e;
            font-weight: 600;
        }
        QLabel { color: #34495e; }
        QLineEdit, QComboBox, QDoubleSpinBox {
            background: #fbfcfe;
            border: 1px solid #d0d5dd;
            border-radius: 6px;
            padding: 4px 6px;
            min-height: 24px;
        }
        QPushButton {
            background: #2d7ff9;
            color: white;
            border: none;
            border-radius: 6px;
            padding: 6px 14px;
        }
        QPushButton:hover { background: #1f6fe0; }
        QPushButton:disabled { background: #b7c3d0; }
        """)

    def _wire_events(self) -> None:
        self.btn_browse_cfg.clicked.connect(self._on_browse_cfg)
        self.btn_load_cfg.clicked.connect(self._on_load_cfg)
        self.btn_save_cfg.clicked.connect(self._on_save_cfg)
        self.btn_run.clicked.connect(self._on_run)
        self.btn_save_test_params.clicked.connect(self._on_save_test_params)
        self.api_combo.currentIndexChanged.connect(self._on_api_changed)
        self.out_combo.currentIndexChanged.connect(self._on_device_changed)
        self.in_combo.currentIndexChanged.connect(self._on_device_changed)

    def _restore_from_defaults(self) -> None:
        # try to load last cfg path from a small sidecar file
        try:
            sidecar_last = os.path.join(os.path.dirname(__file__), ".last_cfg_path.txt")
            if os.path.exists(sidecar_last):
                with open(sidecar_last, "r", encoding="utf-8") as f:
                    last_path = f.read().strip()
                if last_path and os.path.exists(last_path):
                    self.cfg_path_edit.setText(last_path)
                    self._on_load_cfg()
        except Exception:
            pass
        # refresh derived selections
        self._update_device_derived_params()

    def _load_hostapis(self) -> None:
        self.api_combo.blockSignals(True)
        self.api_combo.clear()
        self._hostapis = get_hostapis()
        # Heuristic priority: prefer ASIO, then WASAPI, else first
        default_api_row = 0
        for i, (_, name, _) in enumerate(self._hostapis):
            self.api_combo.addItem(name)
            if name.upper().find("ASIO") >= 0:
                default_api_row = i
            elif name.upper().find("WASAPI") >= 0 and default_api_row == 0:
                default_api_row = i
        if self._hostapis:
            self.api_combo.setCurrentIndex(default_api_row)
            self._current_hostapi_index = self._hostapis[default_api_row][0]
        else:
            self._current_hostapi_index = None
        self.api_combo.blockSignals(False)

    def _load_devices(self) -> None:
        self.out_combo.blockSignals(True)
        self.in_combo.blockSignals(True)
        self.out_combo.clear()
        self.in_combo.clear()
        outs, ins = get_device_list(self._current_hostapi_index)
        self._out_devices = outs
        self._in_devices = ins
        for _, name, _ in outs:
            self.out_combo.addItem(name)
        for _, name, _ in ins:
            self.in_combo.addItem(name)
        # try select defaults
        try:
            cur_in, cur_out = sd.default.device
        except Exception:
            cur_in, cur_out = (None, None)
        if cur_out is not None:
            for i, (idx, _, _) in enumerate(self._out_devices):
                if idx == cur_out:
                    self.out_combo.setCurrentIndex(i)
                    break
        if cur_in is not None:
            for i, (idx, _, _) in enumerate(self._in_devices):
                if idx == cur_in:
                    self.in_combo.setCurrentIndex(i)
                    break
        self.out_combo.blockSignals(False)
        self.in_combo.blockSignals(False)
        self._update_device_derived_params()

    def _on_api_changed(self) -> None:
        row = self.api_combo.currentIndex()
        if 0 <= row < len(self._hostapis):
            self._current_hostapi_index = self._hostapis[row][0]
        else:
            self._current_hostapi_index = None
        self._load_devices()

    def _on_device_changed(self) -> None:
        self._update_device_derived_params()

    def _get_selected_devices_info(self) -> Tuple[Optional[int], Optional[int], Optional[float], int]:
        out_idx = None
        in_idx = None
        fs = None
        in_channels = 1
        if self.out_combo.currentIndex() >= 0 and self.out_combo.currentIndex() < len(self._out_devices):
            out_idx = self._out_devices[self.out_combo.currentIndex()][0]
            out_dev = self._out_devices[self.out_combo.currentIndex()][2]
            fs = float(out_dev.get("default_samplerate") or 0) or None
        if self.in_combo.currentIndex() >= 0 and self.in_combo.currentIndex() < len(self._in_devices):
            in_idx = self._in_devices[self.in_combo.currentIndex()][0]
            in_dev = self._in_devices[self.in_combo.currentIndex()][2]
            # prefer input default fs if output not selected
            if fs is None:
                fs = float(in_dev.get("default_samplerate") or 0) or None
            max_in = int(in_dev.get("max_input_channels", 1) or 1)
            in_channels = max(1, max_in)
        if fs is None or fs <= 0:
            fs = 48000.0
        return out_idx, in_idx, fs, in_channels

    def _update_device_derived_params(self) -> None:
        # Nothing to display explicitly per requirement; keep computed values for run/save only
        _ = self._get_selected_devices_info()

    def _on_browse_cfg(self) -> None:
        path, _ = QFileDialog.getOpenFileName(self, "选择配置文件", ".", "YAML Files (*.yaml *.yml);;All Files (*)")
        if path:
            self.cfg_path_edit.setText(path)

    def _on_load_cfg(self) -> None:
        path = self.cfg_path_edit.text().strip()
        if not path:
            QMessageBox.warning(self, "提示", "请先选择配置文件路径")
            return
        data = load_yaml_file(path)
        self.cfg_editor.load_cfg(data)
        # persist last path
        try:
            sidecar_last = os.path.join(os.path.dirname(__file__), ".last_cfg_path.txt")
            with open(sidecar_last, "w", encoding="utf-8") as f:
                f.write(path)
        except Exception:
            pass
        # try load test params sidecar
        sidecar = f"{path}.test_params.yaml"
        if os.path.exists(sidecar):
            try:
                tdata = load_yaml_file(sidecar)
                self.f0_spin.setValue(float(tdata.get("f0", self.f0_spin.value())))
                self.f1_spin.setValue(float(tdata.get("f1", self.f1_spin.value())))
                self.duration_spin.setValue(float(tdata.get("duration", self.duration_spin.value())))
                # restore audio api and devices if present
                api_name = tdata.get("audio_api_name")
                api_index_saved = tdata.get("audio_api_index")
                if self._hostapis:
                    # match by index first, else by name
                    target_row = None
                    if isinstance(api_index_saved, int):
                        for i, (hidx, _, _) in enumerate(self._hostapis):
                            if hidx == api_index_saved:
                                target_row = i
                                break
                    if target_row is None and isinstance(api_name, str):
                        for i, (_, name, _) in enumerate(self._hostapis):
                            if name == api_name:
                                target_row = i
                                break
                    if target_row is not None:
                        self.api_combo.setCurrentIndex(target_row)
                        # _on_api_changed will reload devices
                    # set device selections
                    out_idx_saved = tdata.get("output_device_index")
                    in_idx_saved = tdata.get("input_device_index")
                    if isinstance(out_idx_saved, int):
                        for i, (idx, _, _) in enumerate(self._out_devices):
                            if idx == out_idx_saved:
                                self.out_combo.setCurrentIndex(i)
                                break
                    if isinstance(in_idx_saved, int):
                        for i, (idx, _, _) in enumerate(self._in_devices):
                            if idx == in_idx_saved:
                                self.in_combo.setCurrentIndex(i)
                                break
            except Exception:
                pass

    def _on_save_cfg(self) -> None:
        path = self.cfg_path_edit.text().strip()
        if not path:
            QMessageBox.warning(self, "提示", "请先选择配置文件路径")
            return
        data = self.cfg_editor.dump_cfg()
        try:
            save_yaml_file(path, data)
            QMessageBox.information(self, "成功", f"已保存配置到: {path}")
        except Exception as e:
            QMessageBox.critical(self, "错误", f"保存失败: {e}")

    def _on_save_test_params(self) -> None:
        path = self.cfg_path_edit.text().strip()
        if not path:
            QMessageBox.warning(self, "提示", "请先选择配置文件路径")
            return
        sidecar = f"{path}.test_params.yaml"
        out_idx, in_idx, fs, in_channels = self._get_selected_devices_info()
        # audio api current
        api_row = self.api_combo.currentIndex()
        api_name = None
        api_index = None
        if 0 <= api_row < len(self._hostapis):
            api_index, api_name, _ = self._hostapis[api_row]
        data = {
            "f0": float(self.f0_spin.value()),
            "f1": float(self.f1_spin.value()),
            "duration": float(self.duration_spin.value()),
            "audio_api_name": api_name,
            "audio_api_index": api_index,
            "output_device_index": out_idx,
            "output_device_name": next((n for (i, n, _) in self._out_devices if i == out_idx), None),
            "input_device_index": in_idx,
            "input_device_name": next((n for (i, n, _) in self._in_devices if i == in_idx), None),
            "samplerate": float(fs),
            "channels": int(in_channels),
        }
        try:
            save_yaml_file(sidecar, data)
            QMessageBox.information(self, "成功", f"已保存测试参数到: {sidecar}")
        except Exception as e:
            QMessageBox.critical(self, "错误", f"保存失败: {e}")

    def _on_run(self) -> None:
        path = self.cfg_path_edit.text().strip()
        if not path or not os.path.exists(path):
            QMessageBox.warning(self, "提示", "请先选择并加载有效的配置文件")
            return
        out_idx, in_idx, fs, in_channels_auto = self._get_selected_devices_info()
        channels = int(in_channels_auto)
        f0 = float(self.f0_spin.value())
        f1 = float(self.f1_spin.value())
        duration = float(self.duration_spin.value())
        self.btn_run.setEnabled(False)
        self.statusBar().showMessage("正在测试中，请稍候...")
        worker = Worker(
            cfg_path=path,
            fs=float(fs),
            duration=duration,
            f0=f0,
            f1=f1,
            channels=channels,
            out_dev=out_idx,
            in_dev=in_idx,
        )
        # run in background thread
        def _on_finish() -> None:
            self.btn_run.setEnabled(True)
            self.statusBar().showMessage("测试完成", 5000)
        def _on_error(msg: str) -> None:
            self.btn_run.setEnabled(True)
            self.statusBar().clearMessage()
            QMessageBox.critical(self, "错误", f"测试失败: {msg}")
        worker.finished.connect(_on_finish)
        worker.error.connect(_on_error)
        worker.result.connect(self._on_result_plot)
        threading.Thread(target=worker.run, daemon=True).start()

    def _on_result_plot(self, data: dict) -> None:
        try:
            fs = float(data.get("fs", 48000.0))
            f0_chirp = float(data.get("f0", 20.0))
            f1_chirp = float(data.get("f1", 15000.0))
            duration_chirp = float(data.get("duration", 5.0))
            rec_raw = data.get("raw", None)
            rec_custom = data.get("custom", None)
            sweep_raw = data.get("sweep_raw", None)
            sweep_custom = data.get("sweep_custom", None)
            if rec_raw is None or rec_custom is None:
                return
            # 单通道
            def _mono(x: np.ndarray) -> np.ndarray:
                return x[:, 0] if getattr(x, "ndim", 1) == 2 else x
            y1 = _mono(rec_raw)
            y2 = _mono(rec_custom)
            n = min(len(y1), len(y2))
            y1 = y1[:n]
            y2 = y2[:n]
            t_plot = np.arange(n) / fs

            # 对数扫频信号的瞬时频率计算函数
            # f(t) = f0 * exp(t * R / L), 其中 R = ln(f1/f0), L = duration
            R_chirp = np.log(f1_chirp / f0_chirp)

            def time_to_freq(t: float) -> float:
                """将时间转换为对数扫频信号的瞬时频率"""
                t_clipped = np.clip(t, 0, duration_chirp)
                return f0_chirp * np.exp(t_clipped * R_chirp / duration_chirp)

            def freq_to_time(f: float) -> float:
                """将频率转换为对数扫频信号的对应时间"""
                f_clipped = np.clip(f, f0_chirp, f1_chirp)
                return duration_chirp * np.log(f_clipped / f0_chirp) / R_chirp

            # 计算 STFT
            nperseg = 1024
            noverlap = 768
            f1v, t1, S1 = spectrogram(y1, fs=fs, nperseg=nperseg, noverlap=noverlap, scaling="spectrum", mode="magnitude")
            f2v, t2, S2 = spectrogram(y2, fs=fs, nperseg=nperseg, noverlap=noverlap, scaling="spectrum", mode="magnitude")
            eps = 1e-12
            S1_db = 20.0 * np.log10(S1 + eps)
            S2_db = 20.0 * np.log10(S2 + eps)
            vmin = float(min(np.nanmin(S1_db), np.nanmin(S2_db)))
            vmax = float(max(np.nanmax(S1_db), np.nanmax(S2_db)))

            # 计算每个采样点对应的瞬时频率（用于以频率为主轴绑图）
            f_plot = np.array([time_to_freq(t) for t in t_plot])

            # 辅助函数：设置合适的 y 轴范围（留出 20% 余量）
            def calc_ylim(data_arrays, margin=0.2):
                all_max = max(np.max(np.abs(arr)) for arr in data_arrays if arr is not None and len(arr) > 0)
                ylim = all_max * (1 + margin)
                return (-ylim, ylim)

            # 辅助函数：设置频率轴刻度
            def setup_freq_axis(ax, f0, f1):
                freq_ticks_target = [20, 50, 100, 200, 500, 1000, 2000, 5000, 10000, 15000, 20000]
                freq_ticks = [f for f in freq_ticks_target if f0 <= f <= f1]
                if f0 not in freq_ticks and f0 >= 10:
                    freq_ticks.insert(0, f0)
                if f1 not in freq_ticks:
                    freq_ticks.append(f1)
                freq_ticks = sorted(set(freq_ticks))
                ax.set_xticks(freq_ticks)
                freq_labels = []
                for f in freq_ticks:
                    if f >= 1000:
                        freq_labels.append(f"{f/1000:.1f}k" if f % 1000 != 0 else f"{int(f/1000)}k")
                    else:
                        freq_labels.append(f"{int(f)}")
                ax.set_xticklabels(freq_labels)
                return freq_ticks

            # 辅助函数：添加时间副轴
            def add_time_axis(ax, f0, f1, duration, time_to_freq_func):
                ax_time = ax.twiny()
                ax_time.set_xscale('log')
                ax_time.set_xlim(f0, f1)
                time_ticks_target = [0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 4.0, 5.0, 6.0, 8.0, 10.0]
                time_ticks = [t for t in time_ticks_target if 0 <= t <= duration]
                if duration not in time_ticks:
                    time_ticks.append(duration)
                time_ticks = sorted(set(time_ticks))
                time_ticks = [t for t in time_ticks if t > 0.01]
                freq_positions = [time_to_freq_func(t) for t in time_ticks]
                ax_time.set_xticks(freq_positions)
                time_labels = [f"{t:.1f}s" if t < 10 else f"{int(t)}s" for t in time_ticks]
                ax_time.set_xticklabels(time_labels)
                ax_time.set_xlabel("时间 (s)", color='#2980B9')
                ax_time.tick_params(axis='x', colors='#2980B9')
                ax_time.spines['top'].set_color('#2980B9')
                return ax_time

            # ==================== 图1：计算机端生成的时域波形 ====================
            if sweep_raw is not None and sweep_custom is not None:
                n_sweep = min(len(sweep_raw), len(sweep_custom))
                t_sweep = np.arange(n_sweep) / fs
                f_sweep = np.array([time_to_freq(t) for t in t_sweep])
                
                fig1 = plt.figure(figsize=(14, 7), constrained_layout=True)
                gs1 = fig1.add_gridspec(2, 1, height_ratios=[1, 1])
                ax1_raw = fig1.add_subplot(gs1[0, 0])
                ax1_mod = fig1.add_subplot(gs1[1, 0])
                
                # 计算合适的 y 轴范围
                ylim_raw = calc_ylim([sweep_raw])
                ylim_mod = calc_ylim([sweep_custom])
                
                # 子图1：原始扫频信号
                ax1_raw.plot(f_sweep, sweep_raw, 'b-', linewidth=0.5, alpha=0.8)
                ax1_raw.set_xscale('log')
                ax1_raw.set_xlim(f0_chirp, f1_chirp)
                ax1_raw.set_ylim(ylim_raw)
                ax1_raw.set_xlabel("瞬时频率 (Hz)")
                ax1_raw.set_ylabel("幅度")
                ax1_raw.set_title("(a) 计算机端生成：原始扫频信号 x(t)")
                ax1_raw.grid(True, alpha=0.3, which='both')
                ax1_raw.axhline(y=0, color='gray', linewidth=0.5)
                setup_freq_axis(ax1_raw, f0_chirp, f1_chirp)
                add_time_axis(ax1_raw, f0_chirp, f1_chirp, duration_chirp, time_to_freq)
                
                # 子图2：调制后扫频信号
                ax1_mod.plot(f_sweep, sweep_custom, 'r-', linewidth=0.5, alpha=0.8)
                ax1_mod.set_xscale('log')
                ax1_mod.set_xlim(f0_chirp, f1_chirp)
                ax1_mod.set_ylim(ylim_mod)
                ax1_mod.set_xlabel("瞬时频率 (Hz)")
                ax1_mod.set_ylabel("幅度")
                ax1_mod.set_title("(b) 计算机端生成：幅度调制后扫频信号 X_mod(t) = M(f)·x(t)")
                ax1_mod.grid(True, alpha=0.3, which='both')
                ax1_mod.axhline(y=0, color='gray', linewidth=0.5)
                setup_freq_axis(ax1_mod, f0_chirp, f1_chirp)
                add_time_axis(ax1_mod, f0_chirp, f1_chirp, duration_chirp, time_to_freq)
                
                fig1.suptitle("计算机端生成的激励信号对比", fontsize=14, fontweight='bold')

            # ==================== 图2：扬声器输出波形 + STFT ====================
            fig2 = plt.figure(figsize=(12, 8.5), constrained_layout=True)
            gs2 = fig2.add_gridspec(3, 1, height_ratios=[1.1, 1.0, 1.0])
            ax_top = fig2.add_subplot(gs2[0, 0])
            ax_mid = fig2.add_subplot(gs2[1, 0])
            ax_bot = fig2.add_subplot(gs2[2, 0])

            # 计算录音信号的合适 y 轴范围
            ylim_rec = calc_ylim([y1, y2])

            # 以频率为主横轴绑图
            ax_top.plot(f_plot, y1, color='#1f77b4', alpha=1, linewidth=1.2, label="未调制时域波形")
            ax_top.plot(f_plot, y2, color='#ff7f0e', alpha=0.7, linewidth=1.2, label="调制后时域波形") 
            ax_top.set_xlabel("瞬时频率 (Hz)")
            ax_top.set_ylabel("幅度")
            ax_top.set_title("扬声器输出波形（主轴：频率 / 辅轴：时间）")
            ax_top.grid(True, alpha=0.3, which='both')
            ax_top.legend(loc='upper right')
            ax_top.set_xscale('log')
            ax_top.set_xlim(f0_chirp, f1_chirp)
            ax_top.set_ylim(ylim_rec)
            setup_freq_axis(ax_top, f0_chirp, f1_chirp)
            add_time_axis(ax_top, f0_chirp, f1_chirp, duration_chirp, time_to_freq)

            im1 = ax_mid.pcolormesh(t1, f1v, S1_db, shading="gouraud", cmap="inferno", vmin=vmin, vmax=vmax)
            ax_mid.set_title("Raw STFT (dB)")
            ax_mid.set_xlabel("时间 (s)")
            ax_mid.set_ylabel("频率 (Hz)")
            ax_mid.set_yscale("log")
            y_min1 = 20.0 if len(f1v) < 2 else max(20.0, float(f1v[1]))
            ax_mid.set_ylim(y_min1, float(f1v[-1]))
            fig2.colorbar(im1, ax=ax_mid, fraction=0.035, pad=0.02)

            im2 = ax_bot.pcolormesh(t2, f2v, S2_db, shading="gouraud", cmap="inferno", vmin=vmin, vmax=vmax)
            ax_bot.set_title("Custom-weighted STFT (dB)")
            ax_bot.set_xlabel("时间 (s)")
            ax_bot.set_ylabel("频率 (Hz)")
            ax_bot.set_yscale("log")
            y_min2 = 20.0 if len(f2v) < 2 else max(20.0, float(f2v[1]))
            ax_bot.set_ylim(y_min2, float(f2v[-1]))
            fig2.colorbar(im2, ax=ax_bot, fraction=0.035, pad=0.02)

            # ==================== 图3：设计的滤波器频率响应 ====================
            # 获取当前的配置文件参数
            cfg = load_yaml_file(self.cfg_path_edit.text()) if hasattr(self, 'cfg_path_edit') and self.cfg_path_edit.text() else {}

            # 设计滤波器
            b = design_custom_weighting_fir(
                fs,
                low_boost_db=cfg.get("low_boost_db", 0.0),
                mid_cut_db=cfg.get("mid_cut_db", 0.0),
                high_boost_db=cfg.get("high_boost_db", 0.0),
                f_low1=cfg.get("f_low1", 20.0),
                f_high1=cfg.get("f_high1", 600.0),
                f_low2=cfg.get("f_low2", 600.0),
                f_high2=cfg.get("f_high2", 1000.0),
                f_low3=cfg.get("f_low3", 1000.0),
                f_high3=cfg.get("f_high3", 10000.0),
            )

            # 计算频率响应
            w, h = freqz(b, worN=8192, fs=fs)
            freq_resp = 20 * np.log10(np.abs(h) + 1e-12)  # 转换为dB

            fig3 = plt.figure(figsize=(10, 6))
            ax_filter = fig3.add_subplot(111)

            # 绘制滤波器频率响应
            ax_filter.plot(w, freq_resp, 'b-', linewidth=2, alpha=0.8)
            ax_filter.set_xlabel("频率 (Hz)")
            ax_filter.set_ylabel("相对增益 (dB)")
            ax_filter.set_title("设计的滤波器频率响应")
            ax_filter.grid(True, alpha=0.3, which='both')
            ax_filter.set_xscale('log')
            ax_filter.set_xlim(20, fs/2)
            ax_filter.set_ylim(-15, 15)  # 通常滤波器增益在-15dB到+15dB范围内

            # 添加参考线
            ax_filter.axhline(y=0, color='black', linestyle='--', alpha=0.5, linewidth=1)
            ax_filter.axvline(x=1000, color='gray', linestyle=':', alpha=0.5, linewidth=1)

            # 添加频段标注
            if cfg.get("f_high1", 600) > cfg.get("f_low1", 20):
                ax_filter.axvspan(cfg.get("f_low1", 20), cfg.get("f_high1", 600),
                                 alpha=0.1, color='red', label=f'低频段增益: {cfg.get("low_boost_db", 0.0):.1f}dB')
            if cfg.get("f_high2", 1000) > cfg.get("f_low2", 600):
                ax_filter.axvspan(cfg.get("f_low2", 600), cfg.get("f_high2", 1000),
                                 alpha=0.1, color='blue', label=f'中频段增益: {cfg.get("mid_cut_db", 0.0):.1f}dB')
            if cfg.get("f_high3", 10000) > cfg.get("f_low3", 1000):
                ax_filter.axvspan(cfg.get("f_low3", 1000), cfg.get("f_high3", 10000),
                                 alpha=0.1, color='green', label=f'高频段增益: {cfg.get("high_boost_db", 0.0):.1f}dB')

            ax_filter.legend(loc='upper right')

            plt.show()
        except Exception as e:
            QMessageBox.critical(self, "错误", f"绘图失败: {e}")

def main() -> None:
    app = QApplication(sys.argv)
    # 统一现代外观
    try:
        app.setStyle("Fusion")
    except Exception:
        pass
    win = MainWindow()
    win.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()


