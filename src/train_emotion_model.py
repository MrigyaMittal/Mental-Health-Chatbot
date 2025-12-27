from datasets import load_dataset
from transformers import (
    DistilBertTokenizerFast,
    DistilBertForSequenceClassification,
    TrainingArguments,
    Trainer
)
import torch

# 1. Load dataset
dataset = load_dataset("go_emotions")

emotion_names = [
    "admiration","amusement","anger","annoyance","approval","caring",
    "confusion","curiosity","desire","disappointment","disapproval","disgust",
    "embarrassment","excitement","fear","gratitude","grief","joy","love",
    "nervousness","optimism","pride","realization","relief","remorse",
    "sadness","surprise","neutral"
]

# 2. Add emotion label
def add_emotion(example):
    example["label"] = example["labels"][0]
    return example

dataset = dataset.map(add_emotion)

# 3. Tokenizer
tokenizer = DistilBertTokenizerFast.from_pretrained(
    "distilbert-base-uncased"
)

def tokenize(batch):
    return tokenizer(
        batch["text"],
        truncation=True,
        padding="max_length",
        max_length=128
    )

dataset = dataset.map(tokenize, batched=True)

dataset.set_format(
    type="torch",
    columns=["input_ids", "attention_mask", "label"]
)

# 4. Model
model = DistilBertForSequenceClassification.from_pretrained(
    "distilbert-base-uncased",
    num_labels=28
)

# 5. Training arguments
training_args = TrainingArguments(
    output_dir="../models/emotion_model",
    num_train_epochs=2,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=16,
    logging_steps=200,
    save_strategy="epoch",
    report_to="none"
)

# 6. Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=dataset["train"],
    eval_dataset=dataset["validation"],
    tokenizer=tokenizer
)

# 7. Train
trainer.train()

# 8. Save model
model.save_pretrained("../models/emotion_model")
tokenizer.save_pretrained("../models/emotion_model")
