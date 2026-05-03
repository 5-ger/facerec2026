import cv2
import numpy as np
import os

MODEL_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "models")
PROTOTXT_PATH = os.path.join(MODEL_DIR, "deploy.prototxt")
CAFFEMODEL_PATH = os.path.join(MODEL_DIR, "res10_300x300_ssd_iter_140000.caffemodel")

MIN_FACE_H = 70
MAX_DETECT_H_RATIO = 0.50

_detector = None
_eye_cascade = None


def _get_detector():
    global _detector
    if _detector is not None:
        return _detector

    if os.path.exists(PROTOTXT_PATH) and os.path.exists(CAFFEMODEL_PATH):
        net = cv2.dnn.readNetFromCaffe(PROTOTXT_PATH, CAFFEMODEL_PATH)
        _detector = ("ssd", net)
    else:
        cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
        _detector = ("haar", cascade)
    return _detector


def _get_eye_cascade():
    global _eye_cascade
    if _eye_cascade is None:
        eye_path = cv2.data.haarcascades + "haarcascade_eye.xml"
        _eye_cascade = cv2.CascadeClassifier(eye_path)
    return _eye_cascade


def get_roi_zone(frame_shape):
    h, w = frame_shape[:2]
    margin_x = int(w * 0.35)
    margin_y = int(h * 0.3125)
    return (margin_x, margin_y, w - margin_x, h - margin_y)


def draw_roi_zone(frame):
    roi = get_roi_zone(frame.shape)
    cv2.rectangle(frame, (roi[0], roi[1]), (roi[2], roi[3]), (0, 220, 220), 2)


def _is_face_in_roi(face_loc, frame_shape):
    top, right, bottom, left = face_loc
    cx = (left + right) // 2
    cy = (top + bottom) // 2
    roi = get_roi_zone(frame_shape)
    return roi[0] <= cx <= roi[2] and roi[1] <= cy <= roi[3]


def _check_face_frontal(frame, face_loc):
    top, right, bottom, left = face_loc
    face_roi = frame[top:bottom, left:right]
    if face_roi.size == 0 or face_roi.shape[0] < 20 or face_roi.shape[1] < 20:
        return False
    gray = cv2.cvtColor(face_roi, cv2.COLOR_BGR2GRAY)
    eyes = _get_eye_cascade().detectMultiScale(gray, 1.08, 3, minSize=(8, 8), maxSize=(80, 40))
    return len(eyes) >= 2


def _check_face_size(face_loc, frame_shape):
    top, right, bottom, left = face_loc
    face_h = bottom - top
    if face_h < MIN_FACE_H:
        return False, "too_far"
    return True, "ok"


def validate_face(frame, face_loc):
    if not _is_face_in_roi(face_loc, frame.shape):
        return False, "outside_roi"
    size_ok, size_status = _check_face_size(face_loc, frame.shape)
    if not size_ok:
        return False, size_status
    return True, "ok"


def detect_faces(frame, confidence_threshold=0.5, max_detect_h_ratio=None):
    detector_type, detector = _get_detector()
    h, w = frame.shape[:2]
    face_locations = []
    ratio = max_detect_h_ratio if max_detect_h_ratio is not None else MAX_DETECT_H_RATIO
    max_detect_h = int(h * ratio)
    max_detect_w = int(w * ratio)

    if detector_type == "ssd":
        blob = cv2.dnn.blobFromImage(frame, 1.0, (300, 300), [104, 117, 123], False, False)
        detector.setInput(blob)
        detections = detector.forward()
        for i in range(detections.shape[2]):
            conf = detections[0, 0, i, 2]
            if conf > confidence_threshold:
                x1 = int(detections[0, 0, i, 3] * w)
                y1 = int(detections[0, 0, i, 4] * h)
                x2 = int(detections[0, 0, i, 5] * w)
                y2 = int(detections[0, 0, i, 6] * h)
                x1, y1 = max(0, x1), max(0, y1)
                x2, y2 = min(w, x2), min(h, y2)
                fh_box = y2 - y1
                fw_box = x2 - x1
                if 0 < fh_box <= max_detect_h and 0 < fw_box <= max_detect_w and fh_box > MIN_FACE_H:
                    face_locations.append((y1, x2, y2, x1))
    else:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gray = cv2.equalizeHist(gray)

        faces, _, weights = detector.detectMultiScale3(
            gray, 1.15, 8, minSize=(80, 80), maxSize=(max_detect_w, max_detect_h),
            outputRejectLevels=True,
        )

        candidates = []
        for i, (x, y, fw, fh) in enumerate(faces):
            wgt = weights[i] if i < len(weights) else 1.0
            aspect = fw / max(fh, 1)
            if 0.7 < aspect < 1.5 and fh > MIN_FACE_H and wgt > 2.5:
                candidates.append((wgt, y, x + fw, y + fh, x))

        candidates.sort(key=lambda c: c[0], reverse=True)
        candidates = candidates[:5]

        for i, (wgt, top, right, bottom, left) in enumerate(candidates):
            keep = True
            for j in range(i):
                _, top2, right2, bottom2, left2 = candidates[j]
                i_top, i_left = max(top, top2), max(left, left2)
                i_bot, i_right = min(bottom, bottom2), min(right, right2)
                if i_right > i_left and i_bot > i_top:
                    overlap = (i_right - i_left) * (i_bot - i_top)
                    area = (right - left) * (bottom - top)
                    if area > 0 and overlap / area > 0.4:
                        keep = False
                        break
            if keep:
                face_locations.append((top, right, bottom, left))

    return face_locations
