import pandas as pd
from transformers import DistilBertTokenizerFast, DistilBertForSequenceClassification
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import torch
import seaborn as sns
import matplotlib.pyplot as plt

# -----------------------------
# 1️⃣ Load Test Data
# -----------------------------
DATA_PATH = "data/suicide_data.csv"

df = pd.read_csv(DATA_PATH)
df = df[["text", "class"]]

label_map = {
    "suicide": 1,
    "non-suicide": 0
}

df["label"] = df["class"].map(label_map)
df = df.dropna()
df["label"] = df["label"].astype(int)

# Use same 10% test split logic
from sklearn.model_selection import train_test_split
_, test_df = train_test_split(df, test_size=0.1, random_state=42)

texts = test_df["text"].tolist()
true_labels = test_df["label"].tolist()

# -----------------------------
# 2️⃣ Load Trained Model
# -----------------------------
MODEL_PATH = "models/crisis_model"

tokenizer = DistilBertTokenizerFast.from_pretrained(MODEL_PATH)
model = DistilBertForSequenceClassification.from_pretrained(MODEL_PATH)
model.eval()

# -----------------------------
# 3️⃣ Predict
# -----------------------------
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
        pred = torch.argmax(outputs.logits, dim=1).item()

    predicted_labels.append(pred)

# -----------------------------
# 4️⃣ Metrics
# -----------------------------
accuracy = accuracy_score(true_labels, predicted_labels)

print("\nTest Accuracy:", round(accuracy * 100, 2), "%\n")

print("Classification Report:\n")
print(classification_report(
    true_labels,
    predicted_labels,
    target_names=["non-suicide", "suicide"],
    digits=4
))

# -----------------------------
# 5️⃣ Confusion Matrix
# -----------------------------
cm = confusion_matrix(true_labels, predicted_labels)

plt.figure(figsize=(6,5))
sns.heatmap(
    cm,
    annot=True,
    fmt="d",
    cmap="Reds",
    xticklabels=["non-suicide", "suicide"],
    yticklabels=["non-suicide", "suicide"]
)
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Crisis Detection Confusion Matrix")
plt.show()