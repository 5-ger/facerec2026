"""Blink-based liveness detection using eye-region intensity variance.
When eyes are open → high variance (dark pupil + bright sclera).
When eyes close (blink) → variance drops sharply (uniform skin).
A real person blinks naturally; a printed photo shows no such drop."""
import time
import numpy as np
import cv2


class LivenessDetector:
    def __init__(self, warmup_time=1.5, max_track_age=5.0):
        self.warmup_time = warmup_time
        self.max_track_age = max_track_age
        self._tracks = {}
        self._next_id = 0

    @staticmethod
    def _iou(loc1, loc2):
        top1, right1, bottom1, left1 = loc1
        top2, right2, bottom2, left2 = loc2
        i_top = max(top1, top2)
        i_left = max(left1, left2)
        i_bot = min(bottom1, bottom2)
        i_right = min(right1, right2)
        if i_right <= i_left or i_bot <= i_top:
            return 0.0
        inter = (i_right - i_left) * (i_bot - i_top)
        area1 = (right1 - left1) * (bottom1 - top1)
        area2 = (right2 - left2) * (bottom2 - top2)
        return inter / (area1 + area2 - inter + 1e-8)

    def _eye_variance(self, frame, face_loc):
        top, right, bottom, left = face_loc
        if top < 0:
            top = 0
        if left < 0:
            left = 0
        face_roi = frame[top:bottom, left:right]
        h, w = face_roi.shape[:2]
        if h < 20 or w < 20:
            return 0.0

        eye_zone = face_roi[int(h * 0.15):int(h * 0.50), :]
        if eye_zone.size == 0:
            return 0.0

        gray = cv2.cvtColor(eye_zone, cv2.COLOR_BGR2GRAY)
        return float(np.var(gray.astype(np.float32)))

    def update(self, frame, face_locations):
        now = time.time()
        matched_ids = set()
        results = {}

        for idx, loc in enumerate(face_locations):
            best_tid = None
            best_iou = 0.25
            for tid, track in self._tracks.items():
                if tid in matched_ids:
                    continue
                iou = self._iou(loc, track["last_loc"])
                if iou > best_iou:
                    best_iou = iou
                    best_tid = tid

            if best_tid is not None:
                matched_ids.add(best_tid)
                track = self._tracks[best_tid]
                track["last_loc"] = loc
            else:
                best_tid = self._next_id
                self._next_id += 1
                self._tracks[best_tid] = {
                    "last_loc": loc,
                    "start_time": now,
                    "var_history": [],
                }

            var_val = self._eye_variance(frame, loc)
            track = self._tracks[best_tid]
            track["var_history"].append((now, var_val))

            cutoff = now - self.max_track_age
            track["var_history"] = [
                (ts, v) for ts, v in track["var_history"] if ts > cutoff
            ]

            elapsed = now - track["start_time"]
            history = track["var_history"]
            max_var = max(v for _, v in history) if history else 0

            if elapsed < self.warmup_time or len(history) < 4:
                is_live = None
            elif elapsed > 5.0 and self._count_blinks(history) == 0:
                is_live = False  # still no blink after 5s → suspected spoof
            else:
                is_live = self._count_blinks(history) >= 1

            results[idx] = {
                "is_live": is_live,
                "var_value": round(var_val, 1),
                "track_time": elapsed,
                "ever_had_eyes": max_var >= 10,
                "track_id": best_tid,
            }

        stale = [
            tid
            for tid, t in self._tracks.items()
            if t["var_history"]
            and now - t["var_history"][-1][0] > self.max_track_age
        ]
        for tid in stale:
            del self._tracks[tid]

        return results

    @staticmethod
    def _count_blinks(history):
        """Count blinks: variance drops >30% below recent peak, then recovers.
        A gentle blink is enough — the threshold is lenient."""
        if len(history) < 3:
            return 0

        blinks = 0
        state = "open"
        recent = []

        for _ts, var in history:
            recent.append(var)
            if len(recent) > 6:
                recent.pop(0)
            peak = max(recent) if recent else var
            # Blink: variance drops below 70% of recent peak
            threshold = peak * 0.70

            if state == "open" and var < threshold and peak > 5:
                state = "closed"
            elif state == "closed" and var >= threshold:
                blinks += 1
                state = "open"

        return blinks
