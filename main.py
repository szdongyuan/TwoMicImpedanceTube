import os
# 必须在导入 sounddevice 之前设置这个环境变量
os.environ["SD_ENABLE_ASIO"] = "1"
import sys
from PyQt5.QtWidgets import QApplication

from control.utils import utils
from control.window import MainWindow


if __name__ == '__main__':
    utils.copy_all_configs_to_base_dir()
    utils.init_selected_devide()
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec_())

