import time
import cv2
from collections import deque, Counter
from deepface import DeepFace

# OpenCV face detector
face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
)

EMOTION_MAP = {
    "angry": "anger",
    "disgust": "disgust",
    "fear": "fear",
    "happy": "joy",
    "sad": "sadness",
    "surprise": "surprise",
    "neutral": "neutral"
}

_HISTORY = deque(maxlen=20)
_LAST_RUN = 0.0
_COOLDOWN = 0.6   # run model only every 0.6 sec (smooth + faster)


def detect_face_emotion(frame_bgr):
    global _LAST_RUN

    now = time.time()
    if now - _LAST_RUN < _COOLDOWN and len(_HISTORY) > 0:
        # return smoothed result
        most_common = Counter([e for e, _ in _HISTORY]).most_common(1)[0][0]
        avg_conf = sum([c for _, c in _HISTORY]) / len(_HISTORY)
        return most_common, float(avg_conf)

    _LAST_RUN = now

    # ✅ 1) detect face first
    gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.2, 5, minSize=(90, 90))

    if len(faces) == 0:
        return "No Face", 0.0

    # pick largest face
    x, y, w, h = sorted(faces, key=lambda f: f[2] * f[3], reverse=True)[0]

    # expand a bit for better emotion detection
    pad = int(0.15 * w)
    x1 = max(0, x - pad)
    y1 = max(0, y - pad)
    x2 = min(frame_bgr.shape[1], x + w + pad)
    y2 = min(frame_bgr.shape[0], y + h + pad)

    face_crop = frame_bgr[y1:y2, x1:x2]

    # ✅ 2) DeepFace only on face crop
    try:
        result = DeepFace.analyze(
            img_path=face_crop,
            actions=["emotion"],
            enforce_detection=True  # IMPORTANT: no guessing
        )

        if isinstance(result, list):
            result = result[0]

        dominant = result.get("dominant_emotion", "neutral")
        emotions = result.get("emotion", {})

        mapped = EMOTION_MAP.get(dominant, "neutral")
        conf = float(emotions.get(dominant, 0.0)) / 100.0

        # ✅ smoothing
        _HISTORY.append((mapped, conf))

        most_common = Counter([e for e, _ in _HISTORY]).most_common(1)[0][0]
        avg_conf = sum([c for _, c in _HISTORY]) / len(_HISTORY)

        return most_common, float(avg_conf)

    except Exception:
        return "No Face", 0.0
