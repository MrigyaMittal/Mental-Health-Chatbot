import pandas as pd
from datasets import Dataset
from transformers import (
    DistilBertTokenizerFast,
    DistilBertForSequenceClassification,
    Trainer,
    TrainingArguments
)
import torch

# -----------------------------
# 1. Load Dataset
# -----------------------------
DATA_PATH = "data/suicide_data.csv"

df = pd.read_csv(DATA_PATH)

# Keep only required columns
df = df[["text", "class"]]

# Map labels to integers
label_map = {
    "suicide": 1,
    "non-suicide": 0
}

df["label"] = df["class"].map(label_map)

# Drop rows that couldn't be mapped
df = df.dropna()

df["label"] = df["label"].astype(int)

# Convert to HuggingFace Dataset
dataset = Dataset.from_pandas(df)

# Train-test split
dataset = dataset.train_test_split(test_size=0.1)

# -----------------------------
# 2. Tokenization
# -----------------------------
tokenizer = DistilBertTokenizerFast.from_pretrained(
    "distilbert-base-uncased"
)

def tokenize(example):
    return tokenizer(
        example["text"],
        truncation=True,
        padding="max_length",
        max_length=128
    )

dataset = dataset.map(tokenize, batched=True)

dataset.set_format(
    type="torch",
    columns=["input_ids", "attention_mask", "label"]
)

# -----------------------------
# 3. Model
# -----------------------------
model = DistilBertForSequenceClassification.from_pretrained(
    "distilbert-base-uncased",
    num_labels=2
)

# -----------------------------
# 4. Training Arguments
# -----------------------------
training_args = TrainingArguments(
    output_dir="models/crisis_model",
    num_train_epochs=3,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=16,
    logging_steps=50,
    save_strategy="epoch",
    report_to="none"
)

# -----------------------------
# 5. Trainer
# -----------------------------
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=dataset["train"],
    eval_dataset=dataset["test"],
    tokenizer=tokenizer
)

# -----------------------------
# 6. Train
# -----------------------------
trainer.train()

# -----------------------------
# 7. Save Model
# -----------------------------
trainer.save_model("models/crisis_model")
tokenizer.save_pretrained("models/crisis_model")

print("✅ Crisis detection model training complete.")
