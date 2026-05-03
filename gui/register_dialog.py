from PyQt5.QtWidgets import (QDialog, QVBoxLayout, QHBoxLayout, QLabel,
                             QLineEdit, QPushButton, QMessageBox)
from PyQt5.QtCore import Qt, QTimer
from PyQt5.QtGui import QPixmap, QImage
import cv2

from core.face_detector import detect_faces, validate_face, draw_roi_zone
from core.face_recognizer import FaceRecognizer
from core.liveness import LivenessDetector


class RegisterDialog(QDialog):
    def __init__(self, recognizer: FaceRecognizer, parent=None):
        super().__init__(parent)
        self.recognizer = recognizer
        self._liveness = LivenessDetector(warmup_time=2.0)
        self.cap = None
        self.timer = QTimer()
        self.timer.timeout.connect(self._update_frame)
        self._last_frame = None
        self._face_valid = False
        self._status_msg = ""
        self._init_ui()
        self._open_camera()

    def _init_ui(self):
        self.setWindowTitle("人脸注册")
        self.setFixedSize(520, 600)
        self.setStyleSheet("""
            QDialog { background-color: #1a1a2e; }
            QLabel { color: #e0e0e0; font-family: "Microsoft YaHei"; }
            QLabel#preview { border: 2px solid #2a2a4e; border-radius: 4px; }
            QLabel#hint_good { color: #00ff88; font-size: 13px; font-weight: bold; }
            QLabel#hint_bad { color: #ffaa00; font-size: 13px; font-weight: bold; }
            QLineEdit {
                background-color: #16213e; border: 1px solid #2a2a4e;
                border-radius: 4px; padding: 6px 10px; font-size: 13px; color: #e0e0e0;
            }
            QLineEdit:focus { border-color: #00d4ff; }
            QPushButton {
                background-color: #16213e; color: #00d4ff;
                border: 1px solid #00d4ff; border-radius: 4px;
                padding: 8px 16px; font-size: 13px; font-weight: bold;
            }
            QPushButton:hover { background-color: #1a3a5e; }
            QPushButton#btn_capture {
                background-color: #00d4ff; color: #1a1a2e; border: none;
                font-size: 14px; padding: 10px 24px;
            }
            QPushButton#btn_capture:disabled {
                background-color: #333; color: #666; border: 1px solid #444;
            }
        """)

        layout = QVBoxLayout()
        layout.setSpacing(10)
        layout.setContentsMargins(24, 16, 24, 16)

        title = QLabel("人脸信息注册")
        title.setAlignment(Qt.AlignCenter)
        title.setStyleSheet("font-size: 18px; font-weight: bold; color: #00d4ff;")
        layout.addWidget(title)

        self.preview_label = QLabel()
        self.preview_label.setObjectName("preview")
        self.preview_label.setFixedSize(400, 300)
        self.preview_label.setAlignment(Qt.AlignCenter)
        self.preview_label.setText("摄像头加载中...")
        self.preview_label.setStyleSheet("color: #888;")
        layout.addWidget(self.preview_label, alignment=Qt.AlignCenter)

        self.hint_label = QLabel("请站在画面中央黄色框内，正对摄像头")
        self.hint_label.setObjectName("hint_bad")
        self.hint_label.setAlignment(Qt.AlignCenter)
        self.hint_label.setWordWrap(True)
        layout.addWidget(self.hint_label)

        form = QHBoxLayout()
        form.addWidget(QLabel("姓名:"))
        self.name_input = QLineEdit()
        self.name_input.setPlaceholderText("请输入姓名")
        form.addWidget(self.name_input)
        form.addWidget(QLabel("部门:"))
        self.dept_input = QLineEdit()
        self.dept_input.setPlaceholderText("（可选）")
        form.addWidget(self.dept_input)
        layout.addLayout(form)

        btn_layout = QHBoxLayout()
        btn_layout.addStretch()
        self.btn_capture = QPushButton("采集并注册")
        self.btn_capture.setObjectName("btn_capture")
        self.btn_capture.clicked.connect(self._on_capture)
        self.btn_capture.setEnabled(False)
        btn_layout.addWidget(self.btn_capture)

        btn_cancel = QPushButton("取消")
        btn_cancel.clicked.connect(self.reject)
        btn_layout.addWidget(btn_cancel)
        btn_layout.addStretch()
        layout.addLayout(btn_layout)

        self.setLayout(layout)

    def _open_camera(self):
        self.cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
        if not self.cap.isOpened():
            self.preview_label.setText("无法打开摄像头")
            return
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        self.timer.start(33)

    def _update_frame(self):
        if self.cap is None:
            return
        ret, frame = self.cap.read()
        if not ret:
            return
        frame = cv2.flip(frame, 1)
        self._last_frame = frame.copy()

        faces = detect_faces(frame)
        self._face_valid = False
        live_results = self._liveness.update(frame, faces) if faces else {}

        for i, (top, right, bottom, left) in enumerate(faces):
            valid, status = validate_face(frame, (top, right, bottom, left))
            live_info = live_results.get(i, {})

            if valid and live_info.get("is_live") is False:
                status = "not_live"
                valid = False
            elif valid and live_info.get("is_live") is None:
                status = "warming"
                valid = False

            if valid:
                color = (0, 255, 136)
                self._face_valid = True
                self._status_msg = "人脸姿态正确，可以注册"
                self.hint_label.setObjectName("hint_good")
            elif status == "not_live":
                color = (255, 80, 80)
                self._status_msg = "检测到非活体，请使用真实人脸"
                self.hint_label.setObjectName("hint_bad")
            elif status == "warming":
                color = (0, 140, 255)
                self._status_msg = "活体检测初始化中，请稍候..."
                self.hint_label.setObjectName("hint_bad")
            elif status == "outside_roi":
                color = (0, 140, 255)
                self._status_msg = "请将人脸移入中央检测区域"
                self.hint_label.setObjectName("hint_bad")
            elif status == "too_far":
                color = (0, 140, 255)
                self._status_msg = "人脸距离过远，请靠近摄像头"
                self.hint_label.setObjectName("hint_bad")
            elif status == "too_close":
                color = (0, 140, 255)
                self._status_msg = "人脸距离过近，请稍微后退"
                self.hint_label.setObjectName("hint_bad")
            elif status == "not_frontal":
                color = (0, 140, 255)
                self._status_msg = "请正对摄像头，不要侧脸或低头"
                self.hint_label.setObjectName("hint_bad")
            else:
                color = (0, 140, 255)
                self._status_msg = "请调整姿态"
                self.hint_label.setObjectName("hint_bad")

            cv2.rectangle(frame, (left, top), (right, bottom), color, 2)
            self.hint_label.setText(self._status_msg)
            self.hint_label.style().unpolish(self.hint_label)
            self.hint_label.style().polish(self.hint_label)

        if not faces:
            self._status_msg = "未检测到人脸，请正对摄像头"
            self.hint_label.setObjectName("hint_bad")
            self.hint_label.setText(self._status_msg)
            self.hint_label.style().unpolish(self.hint_label)
            self.hint_label.style().polish(self.hint_label)

        self.btn_capture.setEnabled(self._face_valid)

        draw_roi_zone(frame)

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        h, w, ch = rgb.shape
        qimg = QImage(rgb.data, w, h, ch * w, QImage.Format_RGB888)
        pixmap = QPixmap.fromImage(qimg).scaled(400, 300, Qt.KeepAspectRatio, Qt.SmoothTransformation)
        self.preview_label.setPixmap(pixmap)

    def _on_capture(self):
        name = self.name_input.text().strip()
        if not name:
            QMessageBox.warning(self, "提示", "请输入姓名")
            return

        if self._last_frame is None:
            QMessageBox.warning(self, "提示", "摄像头未就绪，请稍后重试")
            return

        if not self._face_valid:
            QMessageBox.warning(self, "提示", f"人脸验证未通过: {self._status_msg}")
            return

        user_id, error = self.recognizer.register_face(
            self._last_frame, name, self.dept_input.text().strip()
        )

        if error:
            QMessageBox.warning(self, "注册失败", error)
            return

        QMessageBox.information(self, "注册成功", f"用户 {name} 已成功注册！")
        self.accept()

    def closeEvent(self, event):
        self.timer.stop()
        if self.cap:
            self.cap.release()
        super().closeEvent(event)
