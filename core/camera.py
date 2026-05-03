import cv2
from PyQt5.QtCore import QThread, pyqtSignal
from PyQt5.QtGui import QImage
import numpy as np


class CameraThread(QThread):
    frame_ready = pyqtSignal(np.ndarray)
    camera_error = pyqtSignal(str)

    def __init__(self, camera_id=0):
        super().__init__()
        self.camera_id = camera_id
        self.running = False
        self.cap = None

    def run(self):
        self.running = True
        self.cap = cv2.VideoCapture(self.camera_id)
        if not self.cap.isOpened():
            self.camera_error.emit(f"无法打开摄像头 (ID: {self.camera_id})")
            return

        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        self.cap.set(cv2.CAP_PROP_FPS, 30)

        while self.running:
            ret, frame = self.cap.read()
            if not ret:
                self.camera_error.emit("摄像头读取帧失败")
                break
            frame = cv2.flip(frame, 1)
            self.frame_ready.emit(frame)
            self.msleep(10)

        if self.cap:
            self.cap.release()

    def stop(self):
        self.running = False
        self.wait(3000)


def frame_to_qimage(frame):
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    h, w, ch = rgb.shape
    bytes_per_line = ch * w
    return QImage(rgb.data, w, h, bytes_per_line, QImage.Format_RGB888)
