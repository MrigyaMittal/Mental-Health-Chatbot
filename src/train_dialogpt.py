import torch
import pandas as pd
import csv

from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    Trainer,
    TrainingArguments,
    DataCollatorForLanguageModeling
)

# ---------------- CONFIG ---------------- #

MODEL_NAME = "microsoft/DialoGPT-medium"
OUTPUT_DIR = "models/dialogpt_empathy"
MAX_LENGTH = 128

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ---------------- LOAD DATA ---------------- #

train_df = pd.read_csv(
    "data/empathetic_dialogues/train.csv",
    engine="python",
    quoting=csv.QUOTE_ALL,
    on_bad_lines="skip"
)

# IMPORTANT: check columns
print(train_df.columns)

# Expected columns usually include:
# 'context', 'utterance', 'emotion'

# ---------------- PREPROCESS ---------------- #

def build_text(row):
    return f"User: {row['context']}\nBot: {row['utterance']}"

train_df["text"] = train_df.apply(build_text, axis=1)
texts = train_df["text"].tolist()

# ---------------- TOKENIZER ---------------- #

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
tokenizer.pad_token = tokenizer.eos_token

def tokenize(batch):
    enc = tokenizer(
        batch,
        truncation=True,
        padding="max_length",
        max_length=MAX_LENGTH
    )
    enc["labels"] = enc["input_ids"].copy()
    return enc

tokenized_data = tokenizer(
    texts,
    truncation=True,
    padding="max_length",
    max_length=MAX_LENGTH,
    return_tensors="pt"
)

tokenized_data["labels"] = tokenized_data["input_ids"].clone()

# ---------------- MODEL ---------------- #

model = AutoModelForCausalLM.from_pretrained(MODEL_NAME)
model.to(DEVICE)

# ---------------- DATASET WRAPPER ---------------- #

class DialogueDataset(torch.utils.data.Dataset):
    def __init__(self, encodings):
        self.encodings = encodings

    def __len__(self):
        return self.encodings["input_ids"].size(0)

    def __getitem__(self, idx):
        return {key: val[idx] for key, val in self.encodings.items()}

dataset = DialogueDataset(tokenized_data)

# ---------------- TRAINING ---------------- #

training_args = TrainingArguments(
    output_dir=OUTPUT_DIR,
    per_device_train_batch_size=1,
    gradient_accumulation_steps=8,
    num_train_epochs=1,
    learning_rate=5e-5,
    fp16=torch.cuda.is_available(),
    logging_steps=100,
    save_steps=1000,
    save_total_limit=2,
    report_to="none"
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=dataset,
    data_collator=DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=False
    )
)

trainer.train()

# ---------------- SAVE ---------------- #

model.save_pretrained(OUTPUT_DIR)
tokenizer.save_pretrained(OUTPUT_DIR)

print("✅ DialoGPT fine-tuned on EmpatheticDialogues")
