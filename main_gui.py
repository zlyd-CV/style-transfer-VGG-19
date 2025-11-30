import sys
import os
from PyQt6.QtWidgets import QApplication
from gui.app import MainWindow

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


def main():
    app = QApplication(sys.argv)
    app.setApplicationName("Neural Style Transfer")
    window = MainWindow()
    window.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
    # 貌似风格只能选style的文件夹里的文件,样式只能选content的文件夹里的文件(不知什么原因)
