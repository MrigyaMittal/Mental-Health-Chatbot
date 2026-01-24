from transformers import DistilBertForSequenceClassification, DistilBertTokenizerFast

model_path = "models/emotion_model"   # your local folder

model = DistilBertForSequenceClassification.from_pretrained(model_path)
tokenizer = DistilBertTokenizerFast.from_pretrained(model_path)

model.push_to_hub("MrigyaM/emotion_model")
tokenizer.push_to_hub("MrigyaM/emotion_model")
