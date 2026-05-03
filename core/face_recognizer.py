import os

import numpy as np
import cv2
import mindspore as ms
from mindspore import Tensor

from core.database import (get_all_user_embeddings, add_user, add_user_embedding,
                            delete_user, get_user_encoding_count)
from core.face_detector import detect_faces, validate_face
from core.liveness import LivenessDetector

MODEL_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "models")
MODEL_PATH = os.path.join(MODEL_DIR, "facenet_final.ckpt")
IMG_SIZE = 112
EMBEDDING_SIZE = 128


class FaceRecognizer:
    def __init__(self, tolerance=0.65, margin=0.15):
        self.tolerance = tolerance
        self.margin = margin
        self.known_users = []  # [{id, name, embeddings: [{encoding_id, embedding}, ...]}, ...]
        self._model = None
        self.liveness = LivenessDetector(warmup_time=1.5)
        self._load_model()
        self.reload_users()

    def _load_model(self):
        from train import FaceNet
        if not os.path.exists(MODEL_PATH):
            print(f"警告: 模型文件 {MODEL_PATH} 不存在，请先运行 train.py 训练模型")
            self._model = None
            return
        net = FaceNet(num_classes=10, embedding_size=EMBEDDING_SIZE)
        param_dict = ms.load_checkpoint(MODEL_PATH)
        filtered = {k: v for k, v in param_dict.items()
                    if not k.startswith("classifier.")}
        ms.load_param_into_net(net, filtered, strict_load=False)
        net.set_train(False)
        self._model = net
        print("MindSpore face model loaded.")

    def _preprocess_face(self, frame, location):
        top, right, bottom, left = location
        face = frame[top:bottom, left:right]
        if face.size == 0:
            return None
        face = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)
        face = cv2.resize(face, (IMG_SIZE, IMG_SIZE))
        face = face.astype(np.float32) / 255.0
        face = (face - 0.5) / 0.5
        face = np.transpose(face, (2, 0, 1))
        return np.expand_dims(face, 0)

    def extract_embedding(self, frame, location):
        blob = self._preprocess_face(frame, location)
        if blob is None or self._model is None:
            return None
        tensor = Tensor(blob, ms.float32)
        embedding = self._model(tensor, return_embedding=True)
        return embedding.asnumpy().flatten()

    def reload_users(self):
        self.known_users = get_all_user_embeddings()

    def recognize(self, frame):
        face_locations = detect_faces(frame)
        results = []

        if not face_locations or self._model is None:
            return results

        live_results = self.liveness.update(frame, face_locations)

        for i, loc in enumerate(face_locations):
            valid, status = validate_face(frame, loc)
            live_info = live_results.get(i, {})

            result = {
                "location": loc,
                "name": "Unknown",
                "user_id": None,
                "confidence": 0.0,
                "status": status,
                "is_live": live_info.get("is_live"),
                "track_id": live_info.get("track_id", -1),
            }

            if not valid:
                results.append(result)
                continue

            if live_info.get("is_live") is False:
                if live_info.get("ever_had_eyes"):
                    result["status"] = "not_live"  # photo/video spoof → red box
                else:
                    result["status"] = "false_positive"  # non-face object → silent skip
                results.append(result)
                continue

            if live_info.get("is_live") is None:
                result["status"] = "warming"
                results.append(result)
                continue

            embedding = self.extract_embedding(frame, loc)
            if embedding is None:
                results.append(result)
                continue

            # Per-user best similarity, then margin check
            best_user = None
            best_sim = 0.0
            second_best_sim = 0.0
            for user in self.known_users:
                user_best = 0.0
                for emb_info in user["embeddings"]:
                    sim = self._cosine_sim(embedding, emb_info["embedding"])
                    if sim > user_best:
                        user_best = sim
                if user_best > best_sim:
                    second_best_sim = best_sim
                    best_sim = user_best
                    best_user = user
                elif user_best > second_best_sim:
                    second_best_sim = user_best

            if (best_user
                    and best_sim >= self.tolerance
                    and (best_sim - second_best_sim) >= self.margin):
                result["name"] = best_user["name"]
                result["user_id"] = best_user["id"]
                result["id_number"] = best_user.get("id_number", "")
                result["confidence"] = round(best_sim * 100, 1)

            results.append(result)

        return results

    def register_face(self, frame, name, id_number=""):
        """Register a new user with a single face embedding."""
        face_locations = detect_faces(frame)

        if not face_locations:
            return None, "未检测到人脸，请对正摄像头并保持在检测区域内"

        if len(face_locations) > 1:
            return None, "检测到多张人脸，请确保画面中只有一人"

        loc = face_locations[0]
        valid, status = validate_face(frame, loc)

        if not valid:
            if status == "outside_roi":
                return None, "请将人脸移入中央黄色检测框内"
            if status == "too_far":
                return None, "请靠近摄像头"
            if status == "too_close":
                return None, "请稍微后退"
            return None, "请调整人脸位置和距离"

        embedding = self.extract_embedding(frame, loc)
        if embedding is None:
            return None, "无法提取人脸特征，请调整光线或角度"

        user_id = add_user(name, id_number=id_number)
        add_user_embedding(user_id, embedding)
        self.reload_users()
        return user_id, None

    def register_embeddings(self, embeddings, name, id_number=""):
        """Register a new user with averaged embeddings from multiple angles."""
        if not embeddings:
            return None, "没有可用的人脸特征数据"
        avg_embedding = np.mean(embeddings, axis=0)
        norm = np.linalg.norm(avg_embedding)
        if norm > 0:
            avg_embedding = avg_embedding / norm
        user_id = add_user(name, id_number=id_number)
        add_user_embedding(user_id, avg_embedding)
        # Also add each individual embedding for better matching
        for emb in embeddings:
            add_user_embedding(user_id, emb)
        self.reload_users()
        return user_id, None

    def add_user_samples(self, user_id, embeddings):
        """Add more face samples to an existing user for improved accuracy."""
        for emb in embeddings:
            add_user_embedding(user_id, emb)
        self.reload_users()
        return get_user_encoding_count(user_id)

    def delete_user(self, user_id):
        delete_user(user_id)
        self.reload_users()

    @staticmethod
    def _cosine_sim(a, b):
        return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-8)
