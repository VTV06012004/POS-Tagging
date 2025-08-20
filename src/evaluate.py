# src/evaluate.py
import os
import json
import torch
import numpy as np
from torch.utils.data import Dataset
from sklearn.metrics import precision_recall_fscore_support

from transformers import Trainer

from preprocess import build_tag_mapping
from data_load import load_data_from_txt
from model import load_model_and_tokenizer


# ========= Dataset =========
class POSDataset(Dataset):
    def __init__(self, sentences, tags, tokenizer, tag2id, max_length=128):
        self.sentences = sentences
        self.tags = tags
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


# ========= Metrics =========
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


# ========= Main evaluate =========
def main():
    # 1) Paths
    base_path = "/kaggle/working/POS-Tagging/data/processed"
    if not os.path.exists(base_path):
        base_path = "../data/processed"

    model_dir = "/kaggle/working/POS-Tagging/models/bert-pos"
    if not os.path.exists(model_dir):
        model_dir = "models/bert-pos"

    # 2) Load test data
    test_sentences, test_tags = load_data_from_txt(os.path.join(base_path, "test.txt"))

    # 3) Tag mapping
    tag2id, id2tag = build_tag_mapping(test_tags)

    # 4) Load model + tokenizer đã train
    model, tokenizer = load_model_and_tokenizer(model_dir, tag2id)

    # 5) Dataset test
    test_dataset = POSDataset(test_sentences, test_tags, tokenizer, tag2id)

    # 6) Trainer
    trainer = Trainer(
        model=model,
        tokenizer=tokenizer,
        compute_metrics=compute_metrics,
    )

    # 7) Evaluate trên test
    results = trainer.evaluate(test_dataset)
    print("📊 Evaluation on test set:", results)

    # 8) Save metrics
    os.makedirs("/kaggle/working/POS-Tagging/results", exist_ok=True)
    out_path = "/kaggle/working/POS-Tagging/results/eval_test.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=4)
    print(f"✅ Saved evaluation results to {out_path}")

    # 9) Predict chi tiết → để tiện cho predict.py
    preds_output = trainer.predict(test_dataset)
    preds = np.argmax(preds_output.predictions, axis=-1)

    os.makedirs("/kaggle/working/POS-Tagging/results/predictions", exist_ok=True)
    pred_file = "/kaggle/working/POS-Tagging/results/predictions/test_predictions.txt"

    with open(pred_file, "w", encoding="utf-8") as f:
        for sent, tag_seq, pred_ids, word_ids in zip(
            test_sentences, test_tags, preds, test_dataset.encodings.word_ids(batch_index=0)
        ):
            # Với mỗi câu, ánh xạ token -> nhãn dự đoán
            word_ids = test_dataset.encodings.word_ids(batch_index=0)
            pred_tags = []
            used_idx = set()
            for word_idx, p_i in zip(word_ids, pred_ids):
                if word_idx is None or word_idx in used_idx:
                    continue
                pred_tags.append(id2tag.get(int(p_i), "O"))
                used_idx.add(word_idx)
            f.write(" ".join([f"{w}/{t}" for w, t in zip(sent, pred_tags)]) + "\n")

    print(f"✅ Saved predictions to {pred_file}")


if __name__ == "__main__":
    main()
