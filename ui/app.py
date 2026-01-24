import os
import sys

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

import streamlit as st
import torch
from transformers import (
    DistilBertTokenizerFast,
    DistilBertForSequenceClassification
)
from google import genai
from google.genai.errors import ClientError
from datetime import datetime
import uuid
import cv2
import av
from streamlit_webrtc import webrtc_streamer, VideoProcessorBase
from src.face_emotion import detect_face_emotion


# ================= PAGE CONFIG ================= #

st.set_page_config(
    page_title="Mental Health Chatbot",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ================= CUSTOM CSS ================= #

st.markdown("""
<style>
    /* Hide streamlit branding */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    
    /* Keep sidebar toggle button visible */
    button[kind="header"] {
        visibility: visible !important;
    }
    
    [data-testid="collapsedControl"] {
        visibility: visible !important;
        display: block !important;
    }
    
    /* Professional color palette */
    :root {
        --primary: #2563eb;
        --primary-dark: #1e40af;
        --secondary: #475569;
        --accent: #0ea5e9;
        --bg-main: #ffffff;
        --bg-secondary: #f8fafc;
        --border: #e2e8f0;
        --text-primary: #0f172a;
        --text-secondary: #64748b;
    }
    
    /* Sidebar styling - Professional and clean */
    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #1e293b 0%, #334155 100%);
        border-right: 1px solid rgba(255, 255, 255, 0.1);
    }
    
    [data-testid="stSidebar"] * {
        color: #e2e8f0 !important;
    }
    
    [data-testid="stSidebar"] h1,
    [data-testid="stSidebar"] h2,
    [data-testid="stSidebar"] h3 {
        color: #ffffff !important;
        font-weight: 600;
    }
    
    /* Sidebar buttons - Professional style */
    [data-testid="stSidebar"] button {
        background-color: rgba(255, 255, 255, 0.08) !important;
        border: 1px solid rgba(255, 255, 255, 0.12) !important;
        color: #e2e8f0 !important;
        transition: all 0.2s ease;
        font-weight: 500;
    }
    
    [data-testid="stSidebar"] button:hover {
        background-color: rgba(255, 255, 255, 0.12) !important;
        border-color: rgba(255, 255, 255, 0.2) !important;
    }
    
    [data-testid="stSidebar"] button[kind="primary"] {
        background-color: #2563eb !important;
        color: white !important;
        font-weight: 600;
        border: none !important;
    }
    
    [data-testid="stSidebar"] button[kind="primary"]:hover {
        background-color: #1e40af !important;
    }
    
    /* Main area - Clean professional background */
    .main {
        background-color: #f8fafc;
    }
    
    /* Chat container - Minimal and professional */
    .chat-container {
        max-width: 900px;
        margin: 0 auto;
        background: white;
        border-radius: 8px;
        padding: 2rem;
        box-shadow: 0 1px 3px rgba(0, 0, 0, 0.1);
        border: 1px solid #e2e8f0;
    }
    
    /* Chat messages - Professional spacing */
    .stChatMessage {
        background-color: transparent !important;
        padding: 0.75rem 0 !important;
    }
    
    [data-testid="stChatMessageContent"] {
        border-radius: 8px;
        padding: 1rem 1.25rem;
        line-height: 1.6;
        font-size: 0.95rem;
    }
    
    /* User message - Professional blue */
    [data-testid="stChatMessage"]:has([data-testid="chatAvatarIcon-user"]) [data-testid="stChatMessageContent"] {
        background-color: #2563eb;
        color: white !important;
        margin-left: 10%;
        border: none;
    }
    
    [data-testid="stChatMessage"]:has([data-testid="chatAvatarIcon-user"]) [data-testid="stChatMessageContent"] p {
        color: white !important;
    }
    
    /* Assistant message - Clean white background */
    [data-testid="stChatMessage"]:has([data-testid="chatAvatarIcon-assistant"]) [data-testid="stChatMessageContent"] {
        background-color: #ffffff;
        color: #0f172a !important;
        margin-right: 10%;
        border: 1px solid #e2e8f0;
        box-shadow: 0 1px 2px rgba(0, 0, 0, 0.05);
    }
    
    [data-testid="stChatMessage"]:has([data-testid="chatAvatarIcon-assistant"]) [data-testid="stChatMessageContent"] p {
        color: #0f172a !important;
    }
    
    /* Chat input - Professional and minimal */
    .stChatInputContainer {
        max-width: 900px;
        margin: 0 auto;
        border-top: none !important;
        padding-top: 1rem;
    }
    
    [data-testid="stChatInput"] {
        border-radius: 8px !important;
        border: 1px solid #cbd5e1 !important;
        background-color: white !important;
    }
    
    [data-testid="stChatInput"]:focus-within {
        border-color: #2563eb !important;
        box-shadow: 0 0 0 2px rgba(37, 99, 235, 0.1) !important;
    }
    
    /* Title area - Professional typography */
    .title-container {
        max-width: 900px;
        margin: 0 auto 2rem auto;
        text-align: center;
    }
    
    .main-title {
        font-size: 2rem;
        font-weight: 700;
        color: #5477c8;
        margin-bottom: 0.5rem;
        letter-spacing: -0.025em;
    }
    
    .subtitle {
        color: #64748b;
        font-size: 1rem;
        font-weight: 400;
    }
    
    /* Debug info - Professional error styling */
    .debug-info {
        background-color: #fef2f2;
        border-left: 3px solid #ef4444;
        padding: 0.875rem;
        border-radius: 6px;
        margin-top: 0.75rem;
        font-size: 0.875rem;
        color: #7f1d1d;
    }
    
    /* Empty state - Professional and minimal */
    .empty-state {
        text-align: center;
        padding: 4rem 2rem;
        color: #94a3b8;
    }
    
    .empty-state-icon {
        font-size: 3rem;
        margin-bottom: 1rem;
        opacity: 0.4;
    }
    
    .empty-state h3 {
        font-size: 1.25rem;
        color: #475569;
        margin-bottom: 0.5rem;
        font-weight: 600;
    }
    
    /* Divider - Subtle and professional */
    hr {
        border: none;
        height: 1px;
        background-color: rgba(255, 255, 255, 0.2);
        margin: 1.5rem 0;
    }
    
    /* Chat list item - Professional hover state */
    .chat-list-item {
        padding: 0.75rem;
        border-radius: 6px;
        margin-bottom: 0.5rem;
        cursor: pointer;
        background-color: rgba(255, 255, 255, 0.05);
        transition: background-color 0.2s ease;
        border: 1px solid rgba(255, 255, 255, 0.08);
    }
    
    .chat-list-item:hover {
        background-color: rgba(255, 255, 255, 0.1);
    }
    
    /* Checkbox styling - Professional */
    [data-testid="stSidebar"] .stCheckbox {
        background-color: rgba(255, 255, 255, 0.05);
        padding: 0.625rem;
        border-radius: 6px;
        border: 1px solid rgba(255, 255, 255, 0.08);
    }
    
    /* Avatar styling - Clean circles */
    [data-testid="chatAvatarIcon-user"],
    [data-testid="chatAvatarIcon-assistant"] {
        border-radius: 50%;
    }
    
    /* Scrollbar - Professional minimal style */
    ::-webkit-scrollbar {
        width: 6px;
        height: 6px;
    }
    
    ::-webkit-scrollbar-track {
        background: #f1f5f9;
    }
    
    ::-webkit-scrollbar-thumb {
        background: #cbd5e1;
        border-radius: 3px;
    }
    
    ::-webkit-scrollbar-thumb:hover {
        background: #94a3b8;
    }
    
    /* Professional typography */
    .main {
        font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', 'Roboto', 'Oxygen', 'Ubuntu', 'Cantarell', sans-serif;
    }
    
    /* Remove excessive animations */
    * {
        animation: none !important;
    }
    
    /* Code blocks - Professional styling */
    code {
        background-color: #f1f5f9;
        color: #1e293b;
        padding: 0.2rem 0.4rem;
        border-radius: 4px;
        font-size: 0.875em;
    }
    
    /* Links - Professional blue */
    a {
        color: #2563eb;
        text-decoration: none;
    }
    
    a:hover {
        text-decoration: underline;
    }
</style>
""", unsafe_allow_html=True)

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
    text_lower = text.lower()

    # 🚨 Strong suicidal intent phrases (instant high risk)
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
            return 0.95   # force high crisis risk

    # ---- fallback to your trained model ----
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

    crisis = (
        score >= 0.6
        or (score >= 0.3 and emotion in HIGH_RISK_EMOTIONS)
    )

    return crisis, score

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

if "chats" not in st.session_state:
    chat_id = str(uuid.uuid4())
    st.session_state.chats = {
        chat_id: {
            "id": chat_id,
            "title": "New Chat",
            "created_at": datetime.now(),
            "messages": []
        }
    }
    st.session_state.current_chat_id = chat_id

if "current_chat_id" not in st.session_state:
    st.session_state.current_chat_id = list(st.session_state.chats.keys())[0]

if "show_debug" not in st.session_state:
    st.session_state.show_debug = False

if "camera_emotion" not in st.session_state:
    st.session_state.camera_emotion = "Not Used"
if "camera_conf" not in st.session_state:
    st.session_state.camera_conf = 0.0

# ================= HELPER FUNCTIONS ================= #

def create_new_chat():
    chat_id = str(uuid.uuid4())
    st.session_state.chats[chat_id] = {
        "id": chat_id,
        "title": "New Chat",
        "created_at": datetime.now(),
        "messages": []
    }
    st.session_state.current_chat_id = chat_id

def delete_chat(chat_id):
    if len(st.session_state.chats) > 1:
        del st.session_state.chats[chat_id]
        st.session_state.current_chat_id = list(st.session_state.chats.keys())[0]
    else:
        st.session_state.chats[chat_id]["messages"] = []
        st.session_state.chats[chat_id]["title"] = "New Chat"

def update_chat_title(chat_id, first_message):
    title = first_message[:35] + "..." if len(first_message) > 35 else first_message
    st.session_state.chats[chat_id]["title"] = title

def get_current_chat():
    return st.session_state.chats[st.session_state.current_chat_id]

# ================= SIDEBAR ================= #

with st.sidebar:
    st.markdown("## 🧠 Mental Health Support")
    st.markdown("*Your safe space to talk*")
    
    st.markdown("<br>", unsafe_allow_html=True)
    
    # New Chat Button
    if st.button("✨ New Chat", use_container_width=True, type="primary"):
        create_new_chat()
        st.rerun()
    
    st.markdown("<hr>", unsafe_allow_html=True)
    
    # Settings
    st.markdown("### ⚙️ Settings")
    st.session_state.show_debug = st.checkbox(
        "Show emotion & crisis info",
        value=st.session_state.show_debug
    )
    
    st.markdown("<hr>", unsafe_allow_html=True)
    
    st.markdown("<hr>", unsafe_allow_html=True)
    st.markdown("### 🎥 Facial Emotion (Camera)")

    use_camera = st.checkbox("Enable camera emotion detection", value=False)

    if use_camera:

        class CamProcessor(VideoProcessorBase):
            def __init__(self):
                self.last_emotion = "No Face"
                self.last_conf = 0.0

            def recv(self, frame):
                img = frame.to_ndarray(format="bgr24")

                emotion, conf, extra = detect_face_emotion(img)
                stress = extra.get("stress_score", 0.0)
                engage = extra.get("engagement_score", 0.0)

                self.last_emotion = emotion
                self.last_conf = conf

                # draw info on frame
                label = f"{emotion} ({conf:.2f} | stress:{stress:.2f} | engage:{engage:.2f})"
                cv2.putText(
                    img,
                    label,
                    (20, 40),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1,
                    (0, 255, 0),
                    2
                )

                return av.VideoFrame.from_ndarray(img, format="bgr24")

        ctx = webrtc_streamer(
            key="camera-emotion",
            video_processor_factory=CamProcessor,
            media_stream_constraints={"video": True, "audio": False},
            async_processing=True,
        )

        if ctx.video_processor:
            st.session_state.camera_emotion = ctx.video_processor.last_emotion
            st.session_state.camera_conf = ctx.video_processor.last_conf

            st.info(
                f"Camera Emotion: **{st.session_state.camera_emotion}** "
                f"(conf: {st.session_state.camera_conf:.2f})"
            )
    else:
        st.session_state.camera_emotion = "Not Used"
        st.session_state.camera_conf = 0.0

    
    # Chat History
    st.markdown("### 💬 Chat History")
    
    sorted_chats = sorted(
        st.session_state.chats.values(),
        key=lambda x: x["created_at"],
        reverse=True
    )
    
    for chat in sorted_chats:
        col1, col2 = st.columns([5, 1])
        
        with col1:
            is_active = chat["id"] == st.session_state.current_chat_id
            
            if st.button(
                f"{'📍' if is_active else '💭'} {chat['title']}",
                key=f"chat_{chat['id']}",
                use_container_width=True,
                type="primary" if is_active else "secondary"
            ):
                st.session_state.current_chat_id = chat["id"]
                st.rerun()
        
        with col2:
            if st.button("🗑️", key=f"delete_{chat['id']}", use_container_width=True):
                delete_chat(chat["id"])
                st.rerun()
    
    st.markdown("<hr>", unsafe_allow_html=True)
    
    # Crisis Resources
    st.markdown("### 🆘 Crisis Support")
    st.markdown("""
    **India Helplines:**
    - AASRA: 91-9820466726
    - iCall: 022-25521111
    - Vandrevala: 1860-2662-345
    
    **Emergency:**
    - Police: 100
    - Ambulance: 102
    """)

# ================= MAIN CHAT AREA ================= #

current_chat = get_current_chat()

# Header
st.markdown("""
<div class="title-container">
    <div class="main-title">🧠 Mental Health Support</div>
    <div class="subtitle"> • A safe space to talk • </div>
</div>
""", unsafe_allow_html=True)

# Chat container
if len(current_chat["messages"]) == 0:
    st.markdown("""
    <div class="empty-state">
        <div class="empty-state-icon">💭</div>
        <h2>Welcome to your safe space</h2>
        <p>Share what's on your mind. I'm here to listen and support you.</p>
        <p style="margin-top: 2rem; font-size: 0.9rem;"><em>Your conversations are private and secure.</em></p>
    </div>
    """, unsafe_allow_html=True)
else:
    # Display messages
    for msg in current_chat["messages"]:
        with st.chat_message("user"):
            st.markdown(msg["user"])
        
        with st.chat_message("assistant", avatar="🧠"):
            st.markdown(msg["bot"])
            
            if st.session_state.show_debug:
                emotion_emoji = "😢" if msg["emotion"] in HIGH_RISK_EMOTIONS else "😊"
                crisis_emoji = "🚨" if msg["crisis"] else "✅"
                
                st.markdown(f"""
                <div class="debug-info">
                    {emotion_emoji} <strong>Emotion:</strong> {msg['emotion']} | 
                    {crisis_emoji} <strong>Crisis:</strong> {'Yes' if msg['crisis'] else 'No'} 
                    (Score: {msg['score']:.2f})
                </div>
                """, unsafe_allow_html=True)

# Chat input
user_input = st.chat_input("💭 Type how you're feeling...")

# ================= HANDLE INPUT ================= #

# ================= HANDLE INPUT ================= #

if user_input:
    # Display user message
    with st.chat_message("user"):
        st.markdown(user_input)

    # ✅ 1) emotion from text model
    text_emotion = predict_emotion(user_input)

    # ✅ 2) emotion from camera (if enabled)
    camera_emotion = st.session_state.camera_emotion

    # ✅ 3) decide final emotion
    if camera_emotion in ["Not Used", "No Face"]:
        emotion = text_emotion
    else:
        emotion = camera_emotion

    # ✅ 4) crisis check
    crisis, score = is_crisis(user_input, emotion)

    # Generate response
    with st.chat_message("assistant", avatar="🧠"):
        with st.spinner("💭 Listening..."):
            if crisis:
                bot_reply = (
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
                bot_reply = gemini_response(
                    user_input,
                    emotion,
                    current_chat["messages"]
                )

        st.markdown(bot_reply)

        if st.session_state.show_debug:
            emotion_emoji = "😢" if emotion in HIGH_RISK_EMOTIONS else "😊"
            crisis_emoji = "🚨" if crisis else "✅"

            st.markdown(f"""
            <div class="debug-info">
                {emotion_emoji} <strong>Emotion:</strong> {emotion} | 
                {crisis_emoji} <strong>Crisis:</strong> {'Yes' if crisis else 'No'} 
                (Score: {score:.2f})
            </div>
            """, unsafe_allow_html=True)

            # ✅ optional: show both emotions in debug
            st.markdown(f"""
            <div class="debug-info">
                📝 <strong>Text Emotion:</strong> {text_emotion}<br>
                🎥 <strong>Camera Emotion:</strong> {camera_emotion}
            </div>
            """, unsafe_allow_html=True)

    # Save to history
    current_chat["messages"].append({
        "user": user_input,
        "bot": bot_reply,
        "emotion": emotion,
        "crisis": crisis,
        "score": score
    })

    # Update title on first message
    if len(current_chat["messages"]) == 1:
        update_chat_title(current_chat["id"], user_input)

    st.rerun()
