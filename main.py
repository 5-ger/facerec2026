import sys
import os

sys.path.insert(0, os.path.dirname(__file__))

from PyQt5.QtWidgets import QApplication
from PyQt5.QtCore import Qt

from gui.main_window import MainWindow


def main():
    QApplication.setAttribute(Qt.AA_EnableHighDpiScaling, True)
    QApplication.setAttribute(Qt.AA_UseHighDpiPixmaps, True)

    app = QApplication(sys.argv)
    app.setApplicationName("智能门禁系统")

    window = MainWindow()
    window.show()
 
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()
