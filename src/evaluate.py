import os
import sys
import numpy as np
import torch

from torch.utils.data import Dataset
from sklearn.metrics import precision_recall_fscore_support

from transformers import Trainer, BertForTokenClassification, BertTokenizerFast

# Cho phép import từ src nếu cần (đang đứng ở repo root trên Kaggle)
sys.path.append("/kaggle/working/POS-Tagging/src")

from preprocess import build_tag_mapping
from data_load import load_data_from_txt


# ========= Dataset cho POS (copy gọn để evaluate độc lập) =========
class POSDataset(Dataset):
    def __init__(self, sentences, tags, tokenizer, tag2id, max_length=128):
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
                    label_ids.append(-100)
                previous_word_idx = word_idx
            self.labels.append(label_ids)
        self.encodings.pop("offset_mapping")

    def __getitem__(self, idx):
        item = {k: torch.tensor(v[idx]) for k, v in self.encodings.items()}
        item["labels"] = torch.tensor(self.labels[idx])
        return item

    def __len__(self):
        return len(self.labels)


def compute_metrics(p):
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


def main():
    # Paths
    base_path = "/kaggle/working/POS-Tagging/data/processed"
    if not os.path.exists(base_path):
        base_path = "../data/processed"

    # Load data
    train_sentences, train_tags = load_data_from_txt(os.path.join(base_path, "train.txt"))
    dev_sentences, dev_tags     = load_data_from_txt(os.path.join(base_path, "dev.txt"))
    test_sentences, test_tags   = load_data_from_txt(os.path.join(base_path, "test.txt"))

    # Tag mapping theo train
    tag2id, id2tag = build_tag_mapping(train_tags)

    # Load model/tokenizer đã train
    model = BertForTokenClassification.from_pretrained("models/bert-pos")
    tokenizer = BertTokenizerFast.from_pretrained("models/bert-pos")

    # Datasets
    dev_dataset  = POSDataset(dev_sentences,  dev_tags,  tokenizer, tag2id)
    test_dataset = POSDataset(test_sentences, test_tags, tokenizer, tag2id)

    # Trainer for evaluation
    trainer = Trainer(model=model, tokenizer=tokenizer, compute_metrics=compute_metrics)

    print("Đánh giá trên DEV (dev.txt):")
    metrics_dev = trainer.evaluate(eval_dataset=dev_dataset)
    print(metrics_dev)

    print("\nĐánh giá trên TEST (test.txt):")
    metrics_test = trainer.evaluate(eval_dataset=test_dataset)
    print(metrics_test)


if __name__ == "__main__":
    main()
