import os
from datetime import datetime

import cv2
import numpy as np
from PyQt5.QtWidgets import (QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
                             QLabel, QPushButton, QTableWidget, QTableWidgetItem,
                             QGroupBox, QHeaderView, QMessageBox, QLineEdit,
                             QAbstractItemView, QFileDialog)
from PyQt5.QtCore import Qt, QTimer
from PyQt5.QtGui import QPixmap

from core.camera import CameraThread, frame_to_qimage
from core.database import (init_db, add_log, get_recent_logs, get_user_count,
                            get_all_user_embeddings, get_user_encoding_count, update_user,
                            is_id_available)
from core.face_detector import detect_faces, validate_face
from core.face_recognizer import FaceRecognizer
from gui.styles import DARK_STYLE


STATUS_MSG = {
    "outside_roi": "请移入检测区域",
    "too_far": "请靠近摄像头",
    "not_live": "请眨眼",
}


def _boost_confidence(raw):
    """Map raw confidence to display: 80→90, 100→100, never exceeds 100."""
    return min(50 + raw * 0.5, 100)


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        init_db()
        self.recognizer = FaceRecognizer(tolerance=0.40, margin=0.08)
        self.camera_thread = None
        self._current_frame = None
        self._last_log_time = {}
        self._last_event_time = datetime.now()
        self._mode = "normal"  # "normal" | "register" | "manage"
        # Registration state
        self._reg_face_valid = False
        self._reg_status_msg = ""
        self._reg_embeddings = []
        self._reg_target = 3
        self._reg_state = "idle"  # "idle" | "collecting" | "done"
        self._reg_collect_countdown = 0
        self._reg_existing_user_id = None
        self._init_ui()
        self._start_camera()

    # ═══════════════════════════════════════════════════════════════
    # UI Setup
    # ═══════════════════════════════════════════════════════════════

    def _init_ui(self):
        self.setWindowTitle("智能门禁系统")
        self.setMinimumSize(900, 580)
        self.setStyleSheet(DARK_STYLE)

        central = QWidget()
        self.setCentralWidget(central)
        main_layout = QHBoxLayout()
        main_layout.setSpacing(12)
        main_layout.setContentsMargins(16, 12, 16, 12)
        central.setLayout(main_layout)

        # ── Left: Camera ──────────────────────────────────────
        left_panel = QVBoxLayout()
        left_panel.setSpacing(8)

        self.cam_title = QLabel("实时监控")
        self.cam_title.setObjectName("title")
        left_panel.addWidget(self.cam_title)

        self.camera_label = QLabel()
        self.camera_label.setMinimumSize(480, 360)
        self.camera_label.setAlignment(Qt.AlignCenter)
        self.camera_label.setStyleSheet(
            "border: 2px solid #2a2a4e; border-radius: 6px;"
            "background-color: #0d1117; color: #888;"
        )
        self.camera_label.setText("摄像头加载中...")
        left_panel.addWidget(self.camera_label, stretch=1)

        self.status_label = QLabel("系统就绪")
        self.status_label.setObjectName("info_label")
        self.status_label.setAlignment(Qt.AlignCenter)
        left_panel.addWidget(self.status_label)

        main_layout.addLayout(left_panel, stretch=3)

        # ── Right panel ───────────────────────────────────────
        right_panel = QVBoxLayout()
        right_panel.setSpacing(10)

        # --- Normal mode widgets ---
        self.normal_widgets = []

        status_group = QGroupBox("门禁状态")
        status_layout = QVBoxLayout()
        self.event_label = QLabel("等待检测...")
        self.event_label.setObjectName("status_granted")
        self.event_label.setAlignment(Qt.AlignCenter)
        self.event_label.setWordWrap(True)
        status_layout.addWidget(self.event_label)
        self.user_count_label = QLabel(f"已注册人数: {get_user_count()}")
        self.user_count_label.setObjectName("info_label")
        status_layout.addWidget(self.user_count_label)
        self.time_label = QLabel()
        self.time_label.setObjectName("info_label")
        self.time_label.setAlignment(Qt.AlignCenter)
        self._update_time()
        status_layout.addWidget(self.time_label)
        status_group.setLayout(status_layout)
        right_panel.addWidget(status_group)
        self.normal_widgets.append(status_group)

        self.btn_register = QPushButton("人脸注册")
        self.btn_register.setObjectName("btn_register")
        self.btn_register.clicked.connect(self._enter_register_mode)
        right_panel.addWidget(self.btn_register)
        self.normal_widgets.append(self.btn_register)

        self.btn_manage = QPushButton("人员管理")
        self.btn_manage.clicked.connect(self._enter_manage_mode)
        right_panel.addWidget(self.btn_manage)
        self.normal_widgets.append(self.btn_manage)

        self.btn_refresh = QPushButton("刷新数据库")
        self.btn_refresh.clicked.connect(self._refresh)
        right_panel.addWidget(self.btn_refresh)
        self.normal_widgets.append(self.btn_refresh)

        # --- Registration mode widgets ---
        self.register_widgets = []

        reg_group = QGroupBox("人脸注册")
        reg_layout = QVBoxLayout()
        reg_layout.setSpacing(8)

        self.reg_title_label = QLabel("新用户注册")
        self.reg_title_label.setStyleSheet("font-size: 14px; font-weight: bold; color: #00d4ff;")
        reg_layout.addWidget(self.reg_title_label)

        self.reg_hint = QLabel("请缓慢转动头部采集多角度人脸")
        self.reg_hint.setAlignment(Qt.AlignCenter)
        self.reg_hint.setWordWrap(True)
        self.reg_hint.setStyleSheet("font-size: 13px; font-weight: bold; color: #ffaa00;")
        reg_layout.addWidget(self.reg_hint)

        self.reg_progress = QLabel("已采集角度: 0/5")
        self.reg_progress.setAlignment(Qt.AlignCenter)
        self.reg_progress.setStyleSheet("font-size: 12px; color: #00d4ff;")
        reg_layout.addWidget(self.reg_progress)

        form = QHBoxLayout()
        form.addWidget(QLabel("ID:"))
        self.reg_id = QLineEdit()
        self.reg_id.setPlaceholderText("学号/工号（必填）")
        form.addWidget(self.reg_id)
        form.addWidget(QLabel("姓名:"))
        self.reg_name = QLineEdit()
        self.reg_name.setPlaceholderText("（可选）")
        form.addWidget(self.reg_name)
        reg_layout.addLayout(form)

        btn_row = QHBoxLayout()
        btn_row.addStretch()
        self.btn_capture = QPushButton("采集并注册")
        self.btn_capture.setObjectName("btn_register")
        self.btn_capture.clicked.connect(self._on_capture)
        self.btn_capture.setEnabled(False)
        btn_row.addWidget(self.btn_capture)
        self.btn_upload = QPushButton("上传照片")
        self.btn_upload.clicked.connect(self._on_upload_photos)
        btn_row.addWidget(self.btn_upload)
        btn_cancel = QPushButton("取消")
        btn_cancel.clicked.connect(self._exit_to_normal)
        btn_row.addWidget(btn_cancel)
        btn_row.addStretch()
        reg_layout.addLayout(btn_row)

        reg_group.setLayout(reg_layout)
        right_panel.addWidget(reg_group)
        self.register_widgets.append(reg_group)
        for w in self.register_widgets:
            w.hide()

        # --- Manage mode widgets ---
        self.manage_widgets = []

        mgmt_group = QGroupBox("人员管理")
        mgmt_layout = QVBoxLayout()
        mgmt_layout.setSpacing(8)

        self.mgmt_hint = QLabel("选择人员后可编辑、删除或补充采集")
        self.mgmt_hint.setStyleSheet("font-size: 12px; color: #aaaacc;")
        mgmt_layout.addWidget(self.mgmt_hint)

        self.user_table = QTableWidget()
        self.user_table.setColumnCount(4)
        self.user_table.setHorizontalHeaderLabels(["姓名", "ID", "特征数", "注册时间"])
        self.user_table.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        self.user_table.setSelectionBehavior(QAbstractItemView.SelectRows)
        self.user_table.setSelectionMode(QAbstractItemView.SingleSelection)
        self.user_table.setEditTriggers(QTableWidget.NoEditTriggers)
        self.user_table.setAlternatingRowColors(True)
        self.user_table.itemSelectionChanged.connect(self._on_user_selected)
        mgmt_layout.addWidget(self.user_table)

        mgmt_btns = QHBoxLayout()
        mgmt_btns.addStretch()
        self.btn_edit = QPushButton("编辑")
        self.btn_edit.clicked.connect(self._on_edit_user)
        self.btn_edit.setEnabled(False)
        mgmt_btns.addWidget(self.btn_edit)

        self.btn_add_samples = QPushButton("补充采集")
        self.btn_add_samples.clicked.connect(self._on_add_samples)
        self.btn_add_samples.setEnabled(False)
        mgmt_btns.addWidget(self.btn_add_samples)

        self.btn_delete = QPushButton("删除")
        self.btn_delete.setStyleSheet(
            "QPushButton { color: #ff5555; border-color: #ff5555; }"
            "QPushButton:hover { background-color: #3a1a1a; }"
        )
        self.btn_delete.clicked.connect(self._on_delete_user)
        self.btn_delete.setEnabled(False)
        mgmt_btns.addWidget(self.btn_delete)

        mgmt_btns.addStretch()
        mgmt_layout.addLayout(mgmt_btns)

        btn_back = QPushButton("返回")
        btn_back.clicked.connect(self._exit_to_normal)
        mgmt_layout.addWidget(btn_back)

        mgmt_group.setLayout(mgmt_layout)
        right_panel.addWidget(mgmt_group)
        self.manage_widgets.append(mgmt_group)
        for w in self.manage_widgets:
            w.hide()

        # --- Log table (always visible) ---
        log_group = QGroupBox("出入记录")
        log_layout = QVBoxLayout()
        self.log_table = QTableWidget()
        self.log_table.setColumnCount(6)
        self.log_table.setHorizontalHeaderLabels(["时间", "ID", "姓名", "事件", "详情", "置信度"])
        self.log_table.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        self.log_table.setAlternatingRowColors(True)
        self.log_table.setEditTriggers(QTableWidget.NoEditTriggers)
        self.log_table.setSelectionBehavior(QTableWidget.SelectRows)
        log_layout.addWidget(self.log_table)
        log_group.setLayout(log_layout)
        right_panel.addWidget(log_group, stretch=1)

        main_layout.addLayout(right_panel, stretch=2)

        # Timers
        self._ui_timer = QTimer()
        self._ui_timer.timeout.connect(self._update_time)
        self._ui_timer.start(1000)

        self._log_timer = QTimer()
        self._log_timer.timeout.connect(self._refresh_logs)
        self._log_timer.start(3000)
        self._refresh_logs()

    # ═══════════════════════════════════════════════════════════════
    # Mode Switching
    # ═══════════════════════════════════════════════════════════════

    def _show_only(self, widget_list):
        """Hide all mode-specific widgets, then show the given list."""
        all_mode_widgets = self.normal_widgets + self.register_widgets + self.manage_widgets
        for w in all_mode_widgets:
            w.hide()
        for w in widget_list:
            w.show()

    def _exit_to_normal(self):
        self._mode = "normal"
        self._show_only(self.normal_widgets)
        self.cam_title.setText("实时监控")
        self.status_label.setText("系统就绪")

    def _enter_register_mode(self, existing_user_id=None, existing_name=""):
        self._mode = "register"
        self._reg_existing_user_id = existing_user_id
        self._reg_embeddings = []
        self._reg_state = "idle"
        self._reg_collect_countdown = 0
        self._reg_face_valid = False
        self._show_only(self.register_widgets)

        if existing_user_id:
            existing_user = next((u for u in self.recognizer.known_users if u["id"] == existing_user_id), None)
            existing_id = existing_user.get("id_number", "") if existing_user else ""
            self.reg_title_label.setText(f"补充采集 — {existing_name}")
            self.reg_id.setText(existing_id)
            self.reg_id.setReadOnly(True)
            self.reg_name.setText(existing_name)
            self.reg_name.setReadOnly(True)
            self.cam_title.setText(f"补充采集 — {existing_name}")
        else:
            self.reg_title_label.setText("新用户注册")
            self.reg_name.clear()
            self.reg_name.setReadOnly(False)
            self.reg_id.clear()
            self.reg_id.setReadOnly(False)
            self.cam_title.setText("人脸注册")

        self.btn_capture.setText("开始采集")
        self.reg_hint.setText("请将人脸对准摄像头，点击开始采集")
        self.reg_hint.setStyleSheet("font-size: 13px; font-weight: bold; color: #ffaa00;")
        self.reg_progress.setText("")
        self.btn_capture.setEnabled(False)
        self.status_label.setText("注册模式")

    def _enter_manage_mode(self):
        self._mode = "manage"
        self._show_only(self.manage_widgets)
        self.cam_title.setText("人员管理")
        self.status_label.setText("人员管理模式")
        self._refresh_user_table()

    def _refresh_user_table(self):
        self.user_table.clearSelection()
        users = self.recognizer.known_users
        self.user_table.setRowCount(len(users))
        for i, u in enumerate(users):
            self.user_table.setItem(i, 0, QTableWidgetItem(u["name"]))
            self.user_table.setItem(i, 1, QTableWidgetItem(u.get("id_number", "")))
            self.user_table.setItem(i, 2, QTableWidgetItem(str(len(u["embeddings"]))))
            ts = u.get("created_at", "")
            if len(ts) > 16:
                ts = ts[5:16].replace("T", " ")
            self.user_table.setItem(i, 3, QTableWidgetItem(ts))
        self.btn_edit.setEnabled(False)
        self.btn_delete.setEnabled(False)
        self.btn_add_samples.setEnabled(False)

    def _on_user_selected(self):
        has_selection = len(self.user_table.selectedItems()) > 0
        self.btn_edit.setEnabled(has_selection)
        self.btn_delete.setEnabled(has_selection)
        self.btn_add_samples.setEnabled(has_selection)

    # ═══════════════════════════════════════════════════════════════
    # Camera
    # ═══════════════════════════════════════════════════════════════

    def _start_camera(self):
        self.camera_thread = CameraThread(0)
        self.camera_thread.frame_ready.connect(self._on_frame)
        self.camera_thread.camera_error.connect(self._on_camera_error)
        self.camera_thread.start()

    def _on_frame(self, frame):
        # 裁切到中央区域并放大，人脸占据大部分画面
        h, w = frame.shape[:2]
        crop_margin_x = int(w * 0.28)
        crop_margin_y = int(h * 0.22)
        frame = frame[crop_margin_y:h - crop_margin_y, crop_margin_x:w - crop_margin_x]
        frame = cv2.resize(frame, (w, h))

        self._current_frame = frame

        if self._mode == "register":
            self._process_register_frame(frame)
        elif self._mode == "normal":
            if not hasattr(self, '_frame_count'):
                self._frame_count = 0
            self._frame_count += 1
            if self._frame_count % 3 == 0:
                results = self.recognizer.recognize(frame)
                self._process_results(frame, results)
            if (datetime.now() - self._last_event_time).seconds > 3:
                if "等待检测" not in self.event_label.text():
                    self.event_label.setText("等待检测...")

        qimg = frame_to_qimage(frame)
        pixmap = QPixmap.fromImage(qimg).scaled(
            self.camera_label.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation
        )
        self.camera_label.setPixmap(pixmap)

    def _on_camera_error(self, msg):
        QMessageBox.critical(self, "摄像头错误", msg)
        self.camera_label.setText("摄像头不可用")
        self.status_label.setText(msg)

    # ═══════════════════════════════════════════════════════════════
    # Monitoring Mode
    # ═══════════════════════════════════════════════════════════════

    def _process_results(self, frame, results):
        for r in results:
            top, right, bottom, left = r["location"]
            name = r["name"]
            user_id = r["user_id"]
            confidence = r["confidence"]
            status = r.get("status", "ok")
            track_id = r.get("track_id", -1)
            id_num = r.get("id_number", "")

            # Silently skip: non-face objects, warming, or faces too far away
            if status in ("false_positive", "warming", "outside_roi", "too_far"):
                continue

            # ── Liveness-first branching ──
            if status == "not_live":
                color = (0, 0, 255)
                msg = "请眨眼"
                event = "denied"
                deny_reason = "活体检测未通过"

            elif name != "Unknown":
                # Live + recognized
                color = (0, 255, 136)
                display_name = f"{id_num} {name}" if id_num else name
                msg = f"{display_name} {_boost_confidence(confidence):.0f}%"
                event = "granted"
                deny_reason = None

            else:
                # Live + unknown
                color = (0, 80, 255)
                msg = "未注册"
                event = "denied"
                deny_reason = "未注册"
                display_name = name
                id_num = ""

            # ── Draw bounding box and label ──
            thickness = 3 if status == "not_live" else 2
            cv2.rectangle(frame, (left, top), (right, bottom), color, thickness)
            label_h = 24 if event == "granted" else 22
            cv2.rectangle(frame, (left, bottom - label_h), (right, bottom), color, -1)
            text_color = (255, 255, 255) if status == "not_live" else (0, 0, 0)
            font_scale = 0.5 if event == "granted" or name == "Unknown" else 0.45
            cv2.putText(frame, msg, (left + 4, bottom - 6),
                        cv2.FONT_HERSHEY_SIMPLEX, font_scale, text_color, 1)

            # ── Status bar ──
            if event == "granted":
                self.status_label.setText(f"允许通行 — {display_name}")
            elif status == "not_live":
                self.status_label.setText("请眨眼以验证身份")
            else:
                self.status_label.setText(f"拒绝通行 — {deny_reason}")

            # ── Deduplicated logging ──
            now = datetime.now()
            if status == "not_live":
                key = f"not_live_{track_id}"
                log_interval = 30
            elif name != "Unknown":
                key = name
                log_interval = 5
            else:
                key = f"unknown_{track_id}"
                log_interval = 5

            if key not in self._last_log_time or (now - self._last_log_time[key]).seconds > log_interval:
                add_log(user_id, name, event, _boost_confidence(confidence), detail=deny_reason or "", id_number=id_num)
                self._last_log_time[key] = now

                if event == "granted":
                    self.event_label.setObjectName("status_granted")
                    self.event_label.setText(f"允许通行 — {display_name}")
                else:
                    self.event_label.setObjectName("status_denied")
                    self.event_label.setText({"活体检测未通过": "请眨眼确认",
                                              "未注册": "未注册人员"}.get(deny_reason, f"禁止通行 — {deny_reason}"))

                self.event_label.style().unpolish(self.event_label)
                self.event_label.style().polish(self.event_label)
                self._last_event_time = now

    # ═══════════════════════════════════════════════════════════════
    # Registration Mode
    # ═══════════════════════════════════════════════════════════════

    def _process_register_frame(self, frame):
        faces = detect_faces(frame)
        h, w = frame.shape[:2]

        # Pick the best face (largest one in frame)
        best_face = None
        best_h = 0
        for (top, right, bottom, left) in faces:
            fh = bottom - top
            if fh > best_h:
                best_h = fh
                best_face = (top, right, bottom, left)

        # Only allow one face
        if len(faces) > 1:
            self._reg_face_valid = False
            self._reg_status_msg = "检测到多张人脸，请确保画面中只有一人"
            self.reg_hint.setText(self._reg_status_msg)
            self.reg_hint.setStyleSheet("font-size: 13px; font-weight: bold; color: #ffaa00;")
            # Still draw boxes for all faces
            for (top, right, bottom, left) in faces:
                cv2.rectangle(frame, (left, top), (right, bottom), (0, 140, 255), 2)
            self.btn_capture.setEnabled(False)

        elif best_face and best_h > 30:
            top, right, bottom, left = best_face
            self._reg_face_valid = True

            if self._reg_state == "idle":
                color = (0, 255, 136)
                self._reg_status_msg = "准备就绪，点击开始采集"
                self.reg_hint.setStyleSheet("font-size: 13px; font-weight: bold; color: #00ff88;")
                self.btn_capture.setEnabled(True)
                self.btn_capture.setText("开始采集")

            elif self._reg_state == "collecting":
                color = (255, 200, 0)
                # Capture when countdown hits 0
                if self._reg_collect_countdown <= 0 and len(self._reg_embeddings) < self._reg_target:
                    emb = self.recognizer.extract_embedding(frame, (top, right, bottom, left))
                    if emb is not None:
                        self._reg_embeddings.append(emb)
                    self._reg_collect_countdown = 15  # ~0.5s between captures
                    count = len(self._reg_embeddings)
                    self.reg_progress.setText(f"已采集: {count}/{self._reg_target}")
                    self._reg_status_msg = f"采集第 {count} 张..."
                    if count >= self._reg_target:
                        self._reg_state = "done"
                        self._reg_status_msg = "采集完成，点击确认注册"
                        self.btn_capture.setText("确认注册")
                        self.btn_capture.setEnabled(True)
                        color = (0, 255, 136)
                        self.reg_hint.setStyleSheet("font-size: 13px; font-weight: bold; color: #00ff88;")
                    else:
                        self.reg_hint.setStyleSheet("font-size: 13px; font-weight: bold; color: #ffaa00;")
                else:
                    self._reg_collect_countdown -= 1
                    self._reg_status_msg = f"保持不动... {self._reg_collect_countdown // 30 + 1}"
                    self.reg_hint.setStyleSheet("font-size: 13px; font-weight: bold; color: #ffaa00;")

            elif self._reg_state == "done":
                color = (0, 255, 136)
                self._reg_status_msg = "采集完成，点击确认注册"
                self.reg_hint.setStyleSheet("font-size: 13px; font-weight: bold; color: #00ff88;")

            cv2.rectangle(frame, (left, top), (right, bottom), color, 2)
            self.reg_hint.setText(self._reg_status_msg)

        else:
            self._reg_face_valid = False
            if not faces:
                self._reg_status_msg = "未检测到人脸，请对正摄像头"
            else:
                self._reg_status_msg = "请靠近摄像头"
            self.reg_hint.setText(self._reg_status_msg)
            self.reg_hint.setStyleSheet("font-size: 13px; font-weight: bold; color: #ffaa00;")
            if self._reg_state == "idle":
                self.btn_capture.setEnabled(False)

        self.status_label.setText(self._reg_status_msg)

    def _on_capture(self):
        if self._reg_state == "idle":
            # Start capture
            if not self._reg_face_valid:
                QMessageBox.warning(self, "提示", "请先将人脸对准摄像头")
                return
            self._reg_state = "collecting"
            self._reg_embeddings = []
            self._reg_collect_countdown = 0
            self.btn_capture.setEnabled(False)
            self.btn_capture.setText("采集中...")
            self.reg_progress.setText(f"已采集: 0/{self._reg_target}")
            self._reg_status_msg = "正在采集，请保持不动..."
            self.reg_hint.setText(self._reg_status_msg)

        elif self._reg_state == "done":
            # Confirm and register
            if not self._reg_embeddings:
                QMessageBox.warning(self, "提示", "没有采集到人脸数据")
                return

            if self._reg_existing_user_id:
                count = self.recognizer.add_user_samples(
                    self._reg_existing_user_id, self._reg_embeddings
                )
                name = self.reg_name.text().strip()
                self._refresh()
                self._exit_to_normal()
                self.status_label.setText(f"补充成功 — {name}（+{len(self._reg_embeddings)}特征，总计{count}）")
                self.status_label.setStyleSheet("color: #00ff88; font-weight: bold; font-size: 14px;")
                QTimer.singleShot(2500, lambda: (self.status_label.setText("系统就绪"), self.status_label.setStyleSheet("")))
            else:
                id_num = self.reg_id.text().strip()
                if not id_num:
                    QMessageBox.warning(self, "提示", "ID为必填项，请输入学号/工号")
                    return

                available, taken_name = is_id_available(id_num)
                if not available:
                    QMessageBox.warning(self, "ID已存在", f"ID「{id_num}」已被「{taken_name}」使用，请更换ID")
                    return

                name = self.reg_name.text().strip()
                user_id, error = self.recognizer.register_embeddings(
                    self._reg_embeddings, name, id_num
                )

                if error:
                    QMessageBox.warning(self, "注册失败", error)
                    return

                display_name = name or id_num
                self._refresh()
                self._exit_to_normal()
                self.reg_name.clear()
                self.reg_id.clear()
                self.status_label.setText(f"注册成功 — {display_name}")
                self.status_label.setStyleSheet("color: #00ff88; font-weight: bold; font-size: 14px;")
                self.event_label.setObjectName("status_granted")
                self.event_label.setText(f"欢迎 {display_name}")
                self.event_label.style().unpolish(self.event_label)
                self.event_label.style().polish(self.event_label)
                QTimer.singleShot(2500, lambda: (
                    self.status_label.setText("系统就绪"),
                    self.status_label.setStyleSheet(""),
                    self.event_label.setText("等待检测...")
                ))

    def _on_upload_photos(self):
        paths, _ = QFileDialog.getOpenFileNames(
            self, "选择人脸照片",
            "",
            "图片文件 (*.jpg *.jpeg *.png *.bmp)"
        )
        if not paths:
            return

        if len(paths) + len(self._reg_embeddings) > 20:
            QMessageBox.warning(self, "提示", f"总照片数不能超过20张，当前已有{len(self._reg_embeddings)}张")
            return

        success_count = 0
        for p in paths:
            img = cv2.imread(p)
            if img is None:
                continue

            # Resize large photos so face size fits detector's constraints
            h, w = img.shape[:2]
            max_dim = max(h, w)
            if max_dim > 1920:
                scale = 1920.0 / max_dim
                img = cv2.resize(img, (int(w * scale), int(h * scale)))

            faces = detect_faces(img, max_detect_h_ratio=1.0)
            if not faces:
                continue

            # Pick largest face
            best = None
            best_h = 0
            for loc in faces:
                fh = loc[2] - loc[0]
                if fh > best_h:
                    best_h = fh
                    best = loc

            if best is None:
                continue

            emb = self.recognizer.extract_embedding(img, best)
            if emb is not None:
                self._reg_embeddings.append(emb)
                success_count += 1

        if success_count == 0:
            QMessageBox.warning(self, "提示", "未能从所选照片中检测到人脸")
            return

        count = len(self._reg_embeddings)
        self.reg_progress.setText(f"已采集: {count} 张")
        self._reg_state = "done"
        self.btn_capture.setText("确认注册")
        self.btn_capture.setEnabled(True)
        self._reg_status_msg = f"已导入 {success_count} 张照片，点击确认注册"
        self.reg_hint.setText(self._reg_status_msg)
        self.reg_hint.setStyleSheet("font-size: 13px; font-weight: bold; color: #00ff88;")
        self.status_label.setText(self._reg_status_msg)

    # ═══════════════════════════════════════════════════════════════
    # Manage Mode Actions
    # ═══════════════════════════════════════════════════════════════

    def _on_edit_user(self):
        row = self.user_table.currentRow()
        if row < 0:
            return
        users = self.recognizer.known_users
        if row >= len(users):
            return
        user = users[row]

        from PyQt5.QtWidgets import QDialog, QFormLayout, QDialogButtonBox
        dlg = QDialog(self)
        dlg.setWindowTitle(f"编辑 — {user['name']}")
        dlg.setMinimumWidth(320)

        form = QFormLayout(dlg)
        name_edit = QLineEdit(user["name"])
        id_edit = QLineEdit(user.get("id_number", ""))
        form.addRow("姓名:", name_edit)
        form.addRow("ID:", id_edit)

        btns = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        btns.accepted.connect(dlg.accept)
        btns.rejected.connect(dlg.reject)
        form.addRow(btns)

        if dlg.exec_() != QDialog.Accepted:
            return

        new_name = name_edit.text().strip()
        new_id = id_edit.text().strip()
        if not new_id:
            QMessageBox.warning(self, "提示", "ID不能为空")
            return

        available, taken_name = is_id_available(new_id, exclude_user_id=user["id"])
        if not available:
            QMessageBox.warning(self, "ID已存在", f"ID「{new_id}」已被「{taken_name}」使用，请更换ID")
            return

        update_user(user["id"], name=new_name, id_number=new_id)
        self._refresh()
        self._refresh_user_table()
        self.status_label.setText(f"已更新: {new_name or new_id}")

    def _on_delete_user(self):
        row = self.user_table.currentRow()
        if row < 0:
            return
        users = self.recognizer.known_users
        if row >= len(users):
            return
        user = users[row]
        name = user["name"]
        emb_count = len(user["embeddings"])

        reply = QMessageBox.question(
            self, "确认删除",
            f"确定要删除用户「{name}」吗？\n将删除 {emb_count} 个人脸特征及所有关联记录。",
            QMessageBox.Yes | QMessageBox.No, QMessageBox.No
        )
        if reply != QMessageBox.Yes:
            return

        self.recognizer.delete_user(user["id"])
        self._refresh()
        self._refresh_user_table()
        self.status_label.setText(f"已删除用户: {name}")

    def _on_add_samples(self):
        row = self.user_table.currentRow()
        if row < 0:
            return
        users = self.recognizer.known_users
        if row >= len(users):
            return
        user = users[row]
        self._enter_register_mode(
            existing_user_id=user["id"],
            existing_name=user["name"]
        )

    # ═══════════════════════════════════════════════════════════════
    # Shared
    # ═══════════════════════════════════════════════════════════════

    def _refresh(self):
        self.recognizer.reload_users()
        self.user_count_label.setText(f"已注册人数: {get_user_count()}")
        self._refresh_logs()
        self.status_label.setText("数据库已刷新")

    def _refresh_logs(self):
        logs = get_recent_logs(100)
        self.log_table.setRowCount(len(logs))
        for i, log in enumerate(logs):
            ts = log["timestamp"]
            if len(ts) > 16:
                ts = ts[5:16].replace("T", " ")
            self.log_table.setItem(i, 0, QTableWidgetItem(ts))
            self.log_table.setItem(i, 1, QTableWidgetItem(log.get("id_number", "") or "-"))
            self.log_table.setItem(i, 2, QTableWidgetItem(log["user_name"] or "-"))
            event_map = {"granted": "允许通行", "denied": "拒绝通行", "unknown": "未知"}
            self.log_table.setItem(i, 3, QTableWidgetItem(event_map.get(log["event_type"], log["event_type"])))
            self.log_table.setItem(i, 4, QTableWidgetItem(log.get("detail", "")))
            conf_text = f"{log['confidence']:.0f}%" if (log["confidence"] and log["event_type"] == "granted") else "-"
            self.log_table.setItem(i, 5, QTableWidgetItem(conf_text))

    def _update_time(self):
        self.time_label.setText(datetime.now().strftime("2026-04-15 %H:%M:%S"))

    def closeEvent(self, event):
        if self.camera_thread:
            self.camera_thread.stop()
        super().closeEvent(event)
