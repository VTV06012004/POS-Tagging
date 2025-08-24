import json
import os
import sys
import random
import numpy as np
import torch

from torch.utils.data import Dataset
from sklearn.metrics import precision_recall_fscore_support

from transformers import (
    Trainer,
    TrainingArguments,
    EarlyStoppingCallback,
)

from preprocess import build_tag_mapping
from data_load import load_data_from_txt
from model import load_model_and_tokenizer


# ========= Utils =========
def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


# ========= Dataset cho POS =========
class POSDataset(Dataset):
    def __init__(self, sentences, tags, tokenizer, tag2id, max_length=128):
        # sentences: List[List[str]]
        # tags:      List[List[str]]
        self.encodings = tokenizer(
            sentences,
            is_split_into_words=True,
            return_offsets_mapping=True,
            padding=True,
            truncation=True,
            max_length=max_length,
        )
        self.labels = []
        for i, label_seq in enumerate(tags):
            word_ids = self.encodings.word_ids(batch_index=i)
            label_ids = []
            previous_word_idx = None
            for word_idx in word_ids:
                if word_idx is None:
                    label_ids.append(-100)
                elif word_idx != previous_word_idx:
                    label_ids.append(tag2id[label_seq[word_idx]])
                else:
                    # token con (subword) → ignore
                    label_ids.append(-100)
                previous_word_idx = word_idx
            self.labels.append(label_ids)
        # không cần nữa
        self.encodings.pop("offset_mapping")

    def __getitem__(self, idx):
        item = {k: torch.tensor(v[idx]) for k, v in self.encodings.items()}
        item["labels"] = torch.tensor(self.labels[idx])
        return item

    def __len__(self):
        return len(self.labels)


# ========= Metrics =========
def compute_metrics(p):
    # Tương thích cả EvalPrediction và tuple
    if hasattr(p, "predictions"):
        predictions, labels = p.predictions, p.label_ids
    else:
        predictions, labels = p

    preds = predictions.argmax(axis=-1)

    true_labels, true_preds = [], []
    for pred, label in zip(preds, labels):
        for p_i, l_i in zip(pred, label):
            if l_i != -100:
                true_labels.append(l_i)
                true_preds.append(p_i)

    precision, recall, f1, _ = precision_recall_fscore_support(
        true_labels, true_preds, average="micro", zero_division=0
    )
    accuracy = (np.array(true_labels) == np.array(true_preds)).mean().item()

    return {"accuracy": accuracy, "precision": precision, "recall": recall, "f1": f1}


# ========= Main train =========
def main():
    set_seed(42)

    # 1) Paths
    base_path = "/kaggle/working/POS-Tagging/data/processed"
    if not os.path.exists(base_path):
        base_path = "../data/processed"  # chạy local

    # 2) Load dữ liệu
    train_sentences, train_tags = load_data_from_txt(os.path.join(base_path, "train.txt"))
    dev_sentences, dev_tags     = load_data_from_txt(os.path.join(base_path, "dev.txt"))
    # test có thể dùng ở evaluate.py, không cần ở đây

    # 3) Tag mapping (từ train)
    tag2id, id2tag = build_tag_mapping(train_tags)
    num_labels = len(tag2id)

    # 4) Model & tokenizer
    model_name = "bert-base-cased"
    model, tokenizer = load_model_and_tokenizer(model_name, tag2id)

    # 5) Tạo datasets: train = train.txt, eval = dev.txt
    train_dataset = POSDataset(train_sentences, train_tags, tokenizer, tag2id)
    dev_dataset   = POSDataset(dev_sentences,   dev_tags,   tokenizer, tag2id)

    # 6) TrainingArguments
    training_args = TrainingArguments(
        output_dir="./results",
        num_train_epochs=5,
        per_device_train_batch_size=16,
        per_device_eval_batch_size=16,
        eval_strategy="epoch",
        save_strategy="epoch",
        save_total_limit=2,
        logging_dir="./logs",
        logging_steps=50,
        learning_rate=5e-5,
        weight_decay=0.01,
        load_best_model_at_end=True,
        metric_for_best_model="f1",
        greater_is_better=True,
        report_to=["tensorboard"],
        seed=42,
    )
    
    # 7) Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=dev_dataset,
        tokenizer=tokenizer,
        compute_metrics=compute_metrics,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=2)],
    )

    # 8) Train
    trainer.train()

    # 9) Save best model
    os.makedirs("models/bert-pos", exist_ok=True)
    model.save_pretrained("models/bert-pos")
    tokenizer.save_pretrained("models/bert-pos")

if __name__ == "__main__":
    main()
