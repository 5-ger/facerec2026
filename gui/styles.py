DARK_STYLE = """
QMainWindow {
    background-color: #1a1a2e;
}
QWidget {
    color: #e0e0e0;
    font-family: "Microsoft YaHei", "SimHei", sans-serif;
}
QLabel#title {
    font-size: 18px;
    font-weight: bold;
    color: #00d4ff;
    padding: 8px;
}
QLabel#status_granted {
    font-size: 16px;
    font-weight: bold;
    color: #00ff88;
    padding: 4px;
}
QLabel#status_denied {
    font-size: 16px;
    font-weight: bold;
    color: #ff5555;
    padding: 4px;
}
QLabel#info_label {
    font-size: 13px;
    color: #aaaacc;
    padding: 2px;
}
QPushButton {
    background-color: #16213e;
    color: #00d4ff;
    border: 1px solid #00d4ff;
    border-radius: 4px;
    padding: 8px 16px;
    font-size: 13px;
    font-weight: bold;
}
QPushButton:hover {
    background-color: #1a3a5e;
}
QPushButton:pressed {
    background-color: #0f3460;
}
QPushButton#btn_register {
    background-color: #00d4ff;
    color: #1a1a2e;
    border: none;
    font-size: 14px;
    padding: 10px 24px;
}
QPushButton#btn_register:hover {
    background-color: #33ddff;
}
QTableWidget {
    background-color: #16213e;
    alternate-background-color: #1a2748;
    border: 1px solid #2a2a4e;
    gridline-color: #2a2a4e;
    font-size: 12px;
}
QTableWidget::item {
    padding: 4px;
}
QHeaderView::section {
    background-color: #0f3460;
    color: #00d4ff;
    padding: 4px;
    border: none;
    font-weight: bold;
}
QGroupBox {
    border: 1px solid #2a2a4e;
    border-radius: 6px;
    margin-top: 12px;
    padding-top: 16px;
    font-size: 14px;
    font-weight: bold;
    color: #00d4ff;
}
QGroupBox::title {
    subcontrol-origin: margin;
    left: 12px;
    padding: 0 6px;
}
QLineEdit {
    background-color: #16213e;
    border: 1px solid #2a2a4e;
    border-radius: 4px;
    padding: 6px 10px;
    font-size: 13px;
    color: #e0e0e0;
}
QLineEdit:focus {
    border-color: #00d4ff;
}
"""
