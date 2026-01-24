import io
import os
import json
from collections import deque, Counter
from typing import List, Dict, Optional

import numpy as np
import cv2
from PIL import Image

import torch
from transformers import DistilBertTokenizerFast, DistilBertForSequenceClassification

from fastapi import FastAPI, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware

from dotenv import load_dotenv

# Gemini
from google import genai
from google.genai.errors import ClientError

# AU Detector (py-feat)
from feat import Detector


# =================== APP ===================

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],  # Next frontend
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# =================== LABELS ===================

emotion_labels = [
    "admiration", "amusement", "anger", "annoyance", "approval",
    "caring", "confusion", "curiosity", "desire", "disappointment",
    "disapproval", "disgust", "embarrassment", "excitement", "fear",
    "gratitude", "grief", "joy", "love", "nervousness",
    "optimism", "pride", "realization", "relief", "remorse",
    "sadness", "surprise", "neutral"
]

HIGH_RISK_EMOTIONS = {
    "sadness", "grief", "fear", "remorse",
    "nervousness", "disappointment"
}

# We will output these emotions from camera (AU-based)
CAM_EMOTIONS = ["joy", "sadness", "anger", "fear", "surprise", "disgust", "neutral"]


# =================== LOAD TRANSFORMERS MODELS ===================

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
EMOTION_MODEL_PATH = os.path.join(BASE_DIR, "models", "emotion_model")
CRISIS_MODEL_PATH = os.path.join(BASE_DIR, "models", "crisis_model")

emotion_tokenizer = DistilBertTokenizerFast.from_pretrained(
    EMOTION_MODEL_PATH, local_files_only=True
)
emotion_model = DistilBertForSequenceClassification.from_pretrained(
    EMOTION_MODEL_PATH, local_files_only=True
).to(DEVICE)
emotion_model.eval()

crisis_tokenizer = DistilBertTokenizerFast.from_pretrained(
    CRISIS_MODEL_PATH, local_files_only=True
)
crisis_model = DistilBertForSequenceClassification.from_pretrained(
    CRISIS_MODEL_PATH, local_files_only=True
).to(DEVICE)
crisis_model.eval()


# =================== GEMINI ===================

load_dotenv()
API_KEY = os.getenv("GOOGLE_API_KEY", "")

print("✅ GOOGLE_API_KEY loaded:", "YES" if API_KEY else "NO")

client = genai.Client(api_key=API_KEY)


# =================== AU DETECTOR (py-feat) ===================

# ✅ Build once (very important)
au_detector = Detector(
    face_model="retinaface",
    landmark_model="mobilenet",
    au_model="xgb",
    emotion_model="resmasknet",   # ✅ correct
    facepose_model="img2pose"
)


# ✅ smoothing memory
EMO_HISTORY = deque(maxlen=8)


# =================== TEXT MODELS ===================

def predict_emotion_text(text: str) -> str:
    inputs = emotion_tokenizer(
        text,
        return_tensors="pt",
        truncation=True,
        padding=True,
        max_length=128
    ).to(DEVICE)

    with torch.no_grad():
        logits = emotion_model(**inputs).logits
        probs = torch.softmax(logits, dim=1)
        label_id = torch.argmax(probs, dim=1).item()

    return emotion_labels[label_id]


def crisis_score(text: str) -> float:
    text_lower = text.lower()

    hard_triggers = [
        "kill myself",
        "end my life",
        "suicide",
        "i want to die",
        "i dont want to live",
        "i don't want to live",
        "life is not worth living",
        "no reason to live",
        "i wish i wasn't alive",
        "i wish i were dead",
        "i want to disappear",
        "i'm done with life",
        "i cant live anymore",
        "i can't live anymore"
    ]

    for phrase in hard_triggers:
        if phrase in text_lower:
            return 0.95

    inputs = crisis_tokenizer(
        text,
        return_tensors="pt",
        truncation=True,
        padding=True,
        max_length=128
    ).to(DEVICE)

    with torch.no_grad():
        logits = crisis_model(**inputs).logits
        probs = torch.softmax(logits, dim=1)

    return float(probs[0][1].item())


def is_crisis(text: str, emotion: str):
    score = crisis_score(text)
    crisis = (score >= 0.6) or (score >= 0.3 and emotion in HIGH_RISK_EMOTIONS)
    return crisis, score


# =================== GEMINI RESPONSE ===================

def gemini_response(
    user_input: str,
    primary_emotion: str,
    text_emotion: str,
    cam_emotion: str,
    cam_conf: float,
    history: List[Dict[str, str]]
):
    history_text = "\n".join(
        [f"User: {h['user']}\nBot: {h['bot']}" for h in history[-5:]]
    )

    # ✅ Decide emotional support style (NO emotion labels shown to user)
    face_reliable = (cam_emotion != "Not Used" and cam_conf >= 0.55)
    conflict = face_reliable and cam_emotion != "neutral" and cam_emotion != text_emotion

    # ✅ Therapist behaviour rules based on combined emotion
    support_mode = "normal"
    if primary_emotion in ["sadness", "grief", "remorse", "disappointment"]:
        support_mode = "deep_support"
    elif primary_emotion in ["fear", "nervousness"]:
        support_mode = "anxiety_support"
    elif primary_emotion in ["anger", "annoyance", "disgust"]:
        support_mode = "calm_deescalation"

    # ✅ Extra caution if mismatch
    if conflict:
        support_mode = "gentle_checkin"

    # ✅ Instructions for Gemini based on support mode
    MODE_INSTRUCTIONS = {
        "normal": """
Tone: warm, friendly, supportive.
Goal: listen + respond naturally.
""",
        "deep_support": """
Tone: very gentle, slow, caring, empathetic.
Goal: emotional validation + supportive reassurance.
Do: 1 small coping step (breathing, journaling, talk to someone).
Avoid: being too cheerful or too fast.
""",
        "anxiety_support": """
Tone: calm and grounding.
Goal: reduce overwhelm.
Do: give 1 grounding technique (5-4-3-2-1 or breathing).
Avoid: long paragraphs.
""",
        "calm_deescalation": """
Tone: calm, non-judgmental, stable.
Goal: make user feel heard without arguing.
Do: reflect feelings + ask what's triggering this.
Avoid: sounding defensive.
""",
        "gentle_checkin": """
Tone: gentle, curious, safe.
Goal: don't assume feelings; check softly.
Do: ask 1 short clarifying question.
Avoid: mentioning emotion labels or camera.
"""
    }

    mode_text = MODE_INSTRUCTIONS.get(support_mode, MODE_INSTRUCTIONS["normal"])

    prompt = f"""
You are a calm, empathetic mental health support chatbot acting like a therapist.

Conversation so far:
{history_text}

{mode_text}

Rules:
- NEVER mention camera, facial emotion detection, AI detection, confidence scores
- NEVER say "I detected you look ___"
- Always respond like a supportive therapist
- Validate feelings first, then ask 1 helpful question or give 1 small next step
- Keep response short (3–5 lines)
- No medical advice, no diagnosis

User message:
"{user_input}"
"""

    try:
        res = client.models.generate_content(
            model="gemini-2.5-flash",
            contents=prompt
        )
        return res.text.strip()

    except ClientError:
        return (
            "⚠️ I'm receiving too many requests right now.\n\n"
            "Please wait a minute and try again."
        )

# =================== CAMERA UTILS ===================

def read_upload_to_rgb(image: UploadFile) -> np.ndarray:
    """
    Reads uploaded image into RGB numpy (H,W,3)
    """
    data = image.file.read()
    pil = Image.open(io.BytesIO(data)).convert("RGB")
    rgb = np.array(pil)
    return rgb


def get_au(au_map: dict, key: str) -> float:
    return float(au_map.get(key, 0.0))


def aus_to_emotion(au_map: dict):
    """
    FACS-inspired rule scoring from AUs -> emotion
    Returns: (emotion_label, confidence 0..1)
    """

    # main AUs
    AU01 = get_au(au_map, "AU01")
    AU02 = get_au(au_map, "AU02")
    AU04 = get_au(au_map, "AU04")
    AU05 = get_au(au_map, "AU05")
    AU06 = get_au(au_map, "AU06")
    AU07 = get_au(au_map, "AU07")
    AU09 = get_au(au_map, "AU09")
    AU10 = get_au(au_map, "AU10")
    AU12 = get_au(au_map, "AU12")
    AU14 = get_au(au_map, "AU14")
    AU15 = get_au(au_map, "AU15")
    AU17 = get_au(au_map, "AU17")
    AU20 = get_au(au_map, "AU20")
    AU23 = get_au(au_map, "AU23")
    AU24 = get_au(au_map, "AU24")
    AU25 = get_au(au_map, "AU25")
    AU26 = get_au(au_map, "AU26")

    scores = {
        # ✅ joy = smile + cheek raiser
        "joy": (AU12 * 1.3) + (AU06 * 1.0),

        # ✅ sadness = inner brow raise + brow lower + lip depress
        "sadness": (AU01 * 0.8) + (AU04 * 0.9) + (AU15 * 1.2) + (AU17 * 0.3),

        # ✅ anger = brow lower + lid tighten + lip press/tight
        "anger": (AU04 * 1.2) + (AU07 * 0.8) + (AU23 * 0.8) + (AU24 * 0.8),

        # ✅ fear = brows up + brows down + eyes wide + mouth stretch
        "fear": (AU01 * 0.7) + (AU02 * 0.7) + (AU04 * 0.7) + (AU05 * 0.9) + (AU20 * 0.9) + (AU26 * 0.4),

        # ✅ surprise = brows up + eyes wide + jaw drop
        "surprise": (AU01 * 0.6) + (AU02 * 1.0) + (AU05 * 1.1) + (AU26 * 1.0),

        # ✅ disgust = nose wrinkle + upper lip raise
        "disgust": (AU09 * 1.2) + (AU10 * 1.0) + (AU14 * 0.2),

        # baseline neutral
        "neutral": 0.35,
    }

    best_emotion = max(scores, key=scores.get)
    best_score = float(scores[best_emotion])

    # normalize confidence (simple)
    total = float(sum(scores.values())) + 1e-6
    conf = best_score / total

    # ✅ thresholding: prevent random anger/fear if weak
    if best_emotion != "neutral":
        # require some strength
        if best_score < 0.85:
            return "neutral", 0.50

    return best_emotion, float(conf)


# =================== API ===================

@app.get("/")
def home():
    return {"status": "ok"}


@app.post("/live-emotion")
async def live_emotion(image: UploadFile = File(...)):
    """
    Receives ONE frame from frontend
    Uses py-feat Emotion Model (resmasknet) + smoothing
    """

    rgb = read_upload_to_rgb(image)
    rgb_small = cv2.resize(rgb, (256, 256))

    detections = au_detector.detect_image([rgb_small])

    if detections is None or len(detections) == 0:
        EMO_HISTORY.append("neutral")
        return {"emotion": "neutral", "conf": 0.50, "raw": {}, "aus": {}}

    row = detections.iloc[0]

    # ✅ get emotion probabilities directly from py-feat
    FEAT_EMO_COLS = ["anger", "disgust", "fear", "happiness", "sadness", "surprise", "neutral"]
    emo_probs = {}

    for c in FEAT_EMO_COLS:
        if c in detections.columns:
            v = row.get(c)
            try:
                if v is not None and not np.isnan(v):
                    emo_probs[c] = float(v)
            except Exception:
                pass

    if len(emo_probs) == 0:
        EMO_HISTORY.append("neutral")
        return {"emotion": "neutral", "conf": 0.50, "raw": {}, "aus": {}}

    # ✅ choose best
    best_raw = max(emo_probs, key=emo_probs.get)
    best_conf = float(emo_probs[best_raw])

    # ✅ map to your output labels
    MAP = {
        "happiness": "joy",
        "sadness": "sadness",
        "anger": "anger",
        "fear": "fear",
        "surprise": "surprise",
        "disgust": "disgust",
        "neutral": "neutral",
    }

    emotion = MAP.get(best_raw, "neutral")

    # ✅ smoothing (majority vote)
    EMO_HISTORY.append(emotion)
    stable_emotion = Counter(EMO_HISTORY).most_common(1)[0][0]

    # ✅ stable confidence = avg of recent frames with same emotion
    confs = [best_conf for _ in EMO_HISTORY]
    stable_conf = float(sum(confs) / len(confs))

    # ✅ also send AUs for debug
    au_cols = [c for c in detections.columns if c.startswith("AU")]
    au_map = {}
    for c in au_cols:
        v = row.get(c)
        try:
            if v is not None and not np.isnan(v):
                au_map[c] = float(v)
        except Exception:
            pass

    return {
        "emotion": stable_emotion,
        "conf": round(stable_conf, 2),
        "raw_emotion": emotion,
        "raw_conf": round(best_conf, 2),
        "raw": {k: round(v, 3) for k, v in emo_probs.items()},
        "aus": {k: round(v, 2) for k, v in au_map.items()},
    }

@app.post("/chat")
async def chat(
    message: str = Form(...),
    history: str = Form("[]"),
    live_emotion: str = Form("Not Used"),
    live_conf: float = Form(0.0),
):
    """
    Chat endpoint:
    - text emotion (transformer)
    - facial emotion (py-feat / AU based)
    - BOTH influence response generation
    - crisis detection
    """

    # ✅ parse history safely
    try:
        history_list = json.loads(history)
    except Exception:
        history_list = []

    # ✅ 1) TEXT emotion
    text_emotion = predict_emotion_text(message)

    # ✅ 2) FACE emotion (from frontend live detection)
    cam_emotion = live_emotion if live_emotion in CAM_EMOTIONS else "Not Used"
    cam_conf = float(live_conf)

    # ✅ reliability of face signal
    face_reliable = (cam_emotion != "Not Used" and cam_conf >= 0.55)

    # ✅ 3) Emotion Fusion Logic (USE BOTH)
    # -----------------------------------
    # We create a "primary emotion" mainly for crisis logic,
    # but response will use both signals anyway.
    if face_reliable and cam_emotion != "neutral":
        primary_emotion = cam_emotion
    else:
        primary_emotion = text_emotion

    # ✅ detect mismatch case
    emotion_conflict = False
    if face_reliable and cam_emotion != "neutral" and cam_emotion != text_emotion:
        emotion_conflict = True

    # ✅ emotion summary for Gemini
    if face_reliable:
        emotion_summary = f"""
TEXT emotion suggests: {text_emotion}
FACE emotion suggests: {cam_emotion} (confidence {cam_conf:.2f})
"""
    else:
        emotion_summary = f"""
TEXT emotion suggests: {text_emotion}
FACE emotion not reliable / not used
"""

    # ✅ 4) Crisis detection still uses primary emotion
    crisis, score = is_crisis(message, primary_emotion)

    # ✅ 5) Reply
    if crisis:
        reply = (
            "I'm really sorry you're feeling this way. "
            "You're not alone, and your feelings are valid. "
            "Please consider reaching out to a trusted person or a helpline.\n\n"
            "**🇮🇳 India Crisis Helplines:**\n"
            "- AASRA: 91-9820466726\n"
            "- iCall: 022-25521111\n"
            "- Vandrevala Foundation: 1860-2662-345\n\n"
            "If you're in immediate danger, please contact emergency services (100)."
        )
    else:
        # ✅ send both signals into Gemini prompt
        reply = gemini_response(
            user_input=message,
            primary_emotion=primary_emotion,
            text_emotion=text_emotion,
            cam_emotion=cam_emotion,
            cam_conf=cam_conf,
            history=history_list
        )

    return {
        "reply": reply,

        # ✅ primary emotion (used for crisis + debug)
        "emotion": primary_emotion,

        # ✅ show both
        "text_emotion": text_emotion,
        "camera_emotion": cam_emotion,
        "camera_conf": cam_conf,

        # ✅ extra debug
        "face_reliable": face_reliable,
        "emotion_conflict": emotion_conflict,

        "crisis": crisis,
        "score": float(score),
    }
