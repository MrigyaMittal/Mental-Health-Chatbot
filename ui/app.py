import streamlit as st
import torch
from transformers import (
    DistilBertTokenizerFast,
    DistilBertForSequenceClassification
)
from google import genai
from google.genai.errors import ClientError

# ================= PAGE CONFIG ================= #

st.set_page_config(
    page_title="Mental Health Chatbot",
    page_icon="🧠",
    layout="centered"
)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ================= LOAD MODELS ================= #

@st.cache_resource
def load_emotion_model():
    tokenizer = DistilBertTokenizerFast.from_pretrained("models/emotion_model")
    model = DistilBertForSequenceClassification.from_pretrained(
        "models/emotion_model"
    ).to(DEVICE)
    model.eval()
    return tokenizer, model

@st.cache_resource
def load_crisis_model():
    tokenizer = DistilBertTokenizerFast.from_pretrained("models/crisis_model")
    model = DistilBertForSequenceClassification.from_pretrained(
        "models/crisis_model"
    ).to(DEVICE)
    model.eval()
    return tokenizer, model

emotion_tokenizer, emotion_model = load_emotion_model()
crisis_tokenizer, crisis_model = load_crisis_model()

# ================= GEMINI ================= #

API_KEY = st.secrets["GOOGLE_API_KEY"]
client = genai.Client(api_key=API_KEY)

# ================= LABELS ================= #

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

# ================= ML FUNCTIONS ================= #

def predict_emotion(text):
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

def crisis_score(text):
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

    return probs[0][1].item()

def is_crisis(text, emotion):
    score = crisis_score(text)
    return (
        score >= 0.6 or
        (score >= 0.3 and emotion in HIGH_RISK_EMOTIONS)
    ), score

# ================= GEMINI RESPONSE ================= #

def gemini_response(user_input, emotion, history):
    history_text = "\n".join(
        [f"User: {h['user']}\nBot: {h['bot']}" for h in history[-5:]]
    )

    prompt = f"""
You are a calm, empathetic mental health support chatbot.

Conversation so far:
{history_text}

User emotion: {emotion}
User message: "{user_input}"

Rules:
- Be empathetic and supportive
- No medical advice
- No judgement
- Short, human-like response (3–4 lines)
"""

    try:
        response = client.models.generate_content(
            model="gemini-2.5-flash",
            contents=prompt
        )
        return response.text.strip()

    except ClientError:
        return (
            "⚠️ I'm receiving too many requests right now.\n\n"
            "Please wait a minute and try again."
        )

# ================= SESSION STATE ================= #

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

if "last_user_input" not in st.session_state:
    st.session_state.last_user_input = None

# ================= UI ================= #

st.title("🧠 Mental Health Support Chatbot")
st.caption("A safe space to talk. Not a replacement for professional care.")

show_debug = st.toggle("Show emotion & crisis info", False)

# Chat history display
for chat in st.session_state.chat_history:
    with st.chat_message("user"):
        st.markdown(chat["user"])
    with st.chat_message("assistant"):
        st.markdown(chat["bot"])
        if show_debug:
            st.caption(
                f"Emotion: {chat['emotion']} | "
                f"Crisis: {chat['crisis']} ({chat['score']:.2f})"
            )

# Chat input
user_input = st.chat_input("Type how you're feeling...")

# ================= HANDLE INPUT (CRITICAL FIX) ================= #

if user_input and user_input != st.session_state.last_user_input:
    st.session_state.last_user_input = user_input

    emotion = predict_emotion(user_input)
    crisis, score = is_crisis(user_input, emotion)

    if crisis:
        bot_reply = (
            "I'm really sorry you're feeling this way. "
            "You're not alone. Please consider reaching out to a trusted person or a helpline.\n\n"
            "🇮🇳 AASRA: 91-9820466726\n"
            "If you're in immediate danger, please contact local emergency services."
        )
    else:
        bot_reply = gemini_response(
            user_input,
            emotion,
            st.session_state.chat_history
        )

    st.session_state.chat_history.append({
        "user": user_input,
        "bot": bot_reply,
        "emotion": emotion,
        "crisis": crisis,
        "score": score
    })

    st.rerun()
