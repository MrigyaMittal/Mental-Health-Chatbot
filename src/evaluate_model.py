from datasets import load_dataset
from transformers import DistilBertTokenizerFast, DistilBertForSequenceClassification
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import torch
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

# -------------------------------------------------
# 1️⃣ Load Official GoEmotions Test Split
# -------------------------------------------------

dataset = load_dataset("go_emotions")
test_dataset = dataset["test"]

# Keep only single-label samples
def filter_single_label(example):
    return len(example["labels"]) == 1

test_dataset = test_dataset.filter(filter_single_label)

texts = test_dataset["text"]
true_label_ids = [label[0] for label in test_dataset["labels"]]

emotion_names = [
    "admiration","amusement","anger","annoyance","approval","caring",
    "confusion","curiosity","desire","disappointment","disapproval","disgust",
    "embarrassment","excitement","fear","gratitude","grief","joy","love",
    "nervousness","optimism","pride","realization","relief","remorse",
    "sadness","surprise","neutral"
]

true_labels = [emotion_names[i] for i in true_label_ids]

# -------------------------------------------------
# 2️⃣ Load Trained Model
# -------------------------------------------------

MODEL_PATH = "models/emotion_model"

tokenizer = DistilBertTokenizerFast.from_pretrained(MODEL_PATH)
model = DistilBertForSequenceClassification.from_pretrained(MODEL_PATH)
model.eval()

# -------------------------------------------------
# 3️⃣ Predict
# -------------------------------------------------

predicted_labels = []

for text in texts:
    inputs = tokenizer(
        text,
        return_tensors="pt",
        truncation=True,
        padding=True,
        max_length=128
    )
    with torch.no_grad():
        outputs = model(**inputs)
        pred_id = torch.argmax(outputs.logits, dim=1).item()

    predicted_labels.append(emotion_names[pred_id])

# -------------------------------------------------
# 4️⃣ Metrics
# -------------------------------------------------

accuracy = accuracy_score(true_labels, predicted_labels)

print("\nTest Accuracy:", round(accuracy * 100, 2), "%\n")

print("Classification Report:\n")
print(classification_report(true_labels, predicted_labels, digits=4))

# -------------------------------------------------
# 5️⃣ Confusion Matrix
# -------------------------------------------------

cm = confusion_matrix(true_labels, predicted_labels, labels=emotion_names)

plt.figure(figsize=(14, 10))
sns.heatmap(cm, cmap="Blues")
plt.title("Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.show()