import torch
import sys
from collections import deque

from transformers import (
    DistilBertTokenizerFast,
    DistilBertForSequenceClassification
)

from google import genai

# ===================== CONFIG ===================== #

API_KEY = ""

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

MEMORY_SIZE = 5   # last 5 turns

# ===================== LOAD GEMINI ===================== #

client = genai.Client(api_key=API_KEY)

# ===================== LOAD EMOTION MODEL ===================== #

emotion_tokenizer = DistilBertTokenizerFast.from_pretrained(
    "models/emotion_model"
)
emotion_model = DistilBertForSequenceClassification.from_pretrained(
    "models/emotion_model"
).to(DEVICE)
emotion_model.eval()

emotion_labels = [
    "admiration", "amusement", "anger", "annoyance", "approval",
    "caring", "confusion", "curiosity", "desire", "disappointment",
    "disapproval", "disgust", "embarrassment", "excitement", "fear",
    "gratitude", "grief", "joy", "love", "nervousness",
    "optimism", "pride", "realization", "relief", "remorse",
    "sadness", "surprise", "neutral"
]

# ===================== LOAD CRISIS MODEL ===================== #

crisis_tokenizer = DistilBertTokenizerFast.from_pretrained(
    "models/crisis_model"
)
crisis_model = DistilBertForSequenceClassification.from_pretrained(
    "models/crisis_model"
).to(DEVICE)
crisis_model.eval()

# ===================== MEMORY ===================== #

conversation_memory = deque(maxlen=MEMORY_SIZE)

# ===================== PREDICTION FUNCTIONS ===================== #

def predict_emotion(text):
    inputs = emotion_tokenizer(
        text,
        return_tensors="pt",
        truncation=True,
        padding=True,
        max_length=128
    ).to(DEVICE)

    with torch.no_grad():
        outputs = emotion_model(**inputs)
        probs = torch.softmax(outputs.logits, dim=1)
        label_id = torch.argmax(probs, dim=1).item()

    return emotion_labels[label_id]


def predict_crisis(text):
    inputs = crisis_tokenizer(
        text,
        return_tensors="pt",
        truncation=True,
        padding=True,
        max_length=128
    ).to(DEVICE)

    with torch.no_grad():
        outputs = crisis_model(**inputs)
        probs = torch.softmax(outputs.logits, dim=1)

    crisis_score = probs[0][1].item()  # class 1 = crisis
    return crisis_score, crisis_score >= 0.6


# ===================== GEMINI RESPONSE ===================== #

def gemini_response(user_text, emotion):
    memory_text = ""
    for turn in conversation_memory:
        memory_text += f"User: {turn['user']}\nBot: {turn['bot']}\n"

    prompt = f"""
You are a calm, empathetic mental health support chatbot.

Conversation so far:
{memory_text}

User emotion: {emotion}
User message: "{user_text}"

Rules:
- Be empathetic and supportive
- Do NOT give medical advice
- Do NOT encourage self-harm
- Keep response short (1–4 lines)
"""

    try:
        response = client.models.generate_content(
            model="gemini-2.5-flash",
            contents=prompt
        )
        return response.text.strip()
    except Exception:
        return "I'm here with you. You can talk to me. What's been on your mind?"


# ===================== MAIN LOOP ===================== #

def main():
    print("\n🧠 Mental Health Chatbot (type 'exit' to quit)\n")

    while True:
        user_input = input("You: ").strip()

        if user_input.lower() == "exit":
            print("Bot: Take care. You're not alone 💙")
            sys.exit()

        emotion = predict_emotion(user_input)
        crisis_score, crisis = predict_crisis(user_input)

        print(f"Emotion: {emotion}")
        print(f"Crisis detected: {crisis} (score={crisis_score:.2f})")

        if crisis:
            print(
                "\nBot: I'm really sorry you're feeling this way.\n"
                "You deserve care and support.\n"
                "Please reach out to a trusted person or a mental health professional.\n"
                "If you're in immediate danger, contact local emergency services.\n"
            )
            continue

        reply = gemini_response(user_input, emotion)
        print(f"\nBot: {reply}\n")

        conversation_memory.append({
            "user": user_input,
            "bot": reply
        })


if __name__ == "__main__":
    main()
