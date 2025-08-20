import os
import json
import numpy as np
import torch
from torch.utils.data import Dataset
from sklearn.metrics import precision_recall_fscore_support
from transformers import Trainer, TrainingArguments

from preprocess import build_tag_mapping
from data_load import load_data_from_txt
from model import load_model_and_tokenizer

# Tắt W&B
os.environ["WANDB_DISABLED"] = "true"

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
            prev_word_idx = None
            for word_idx in word_ids:
                if word_idx is None:
                    label_ids.append(-100)
                elif word_idx != prev_word_idx:
                    label_ids.append(tag2id[label_seq[word_idx]])
                else:
                    # subword -> ignore
                    label_ids.append(-100)
                prev_word_idx = word_idx
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
    # Tương thích EvalPrediction hoặc tuple
    if hasattr(p, "predictions"):
        predictions, labels = p.predictions, p.label_ids
    else:
        predictions, labels = p

    preds = predictions.argmax(axis=-1)

    true_labels, true_preds = [], []

    for pred, label in zip(preds, labels):
        for p_i, l_i in zip(pred, label):
            if l_i != -100:  # bỏ subword/pad
                true_labels.append(l_i)
                true_preds.append(p_i)

    precision, recall, f1, _ = precision_recall_fscore_support(
        true_labels, true_preds, average="micro", zero_division=0
    )
    accuracy = (np.array(true_labels) == np.array(true_preds)).mean().item()

    return {"accuracy": accuracy, "precision": precision, "recall": recall, "f1": f1}

# ========= Main =========
def main():
    # Paths
    base_path = "/kaggle/working/POS-Tagging/data/processed"
    if not os.path.exists(base_path):
        base_path = "../data/processed"

    model_dir = "/kaggle/working/POS-Tagging/models/bert-pos"
    if not os.path.exists(model_dir):
        model_dir = "models/bert-pos"

    eval_file = os.environ.get("EVAL_SPLIT", "test.txt")
    eval_path = os.path.join(base_path, eval_file)

    # Load train để build mapping
    train_sentences, train_tags = load_data_from_txt(os.path.join(base_path, "train.txt"))
    tag2id, id2tag = build_tag_mapping(train_tags)

    # Load eval data
    eval_sentences, eval_tags = load_data_from_txt(eval_path)

    # Load model + tokenizer
    model, tokenizer = load_model_and_tokenizer(model_dir, tag2id)

    # Dataset eval
    eval_dataset = POSDataset(eval_sentences, eval_tags, tokenizer, tag2id)

    # TrainingArguments (evaluate-only)
    args = TrainingArguments(
        output_dir="./results/eval_tmp",
        per_device_eval_batch_size=16,
        dataloader_num_workers=2,
        report_to="none",
        logging_dir="./logs",
    )

    # Trainer
    trainer = Trainer(
        model=model,
        args=args,
        eval_dataset=eval_dataset,
        tokenizer=tokenizer,
        compute_metrics=compute_metrics,
    )

    # Evaluate
    results = trainer.evaluate()
    print("📊 Evaluation on", eval_file, ":", results)

    # Save metrics
    results_dir = "/kaggle/working/POS-Tagging/results"
    os.makedirs(results_dir, exist_ok=True)
    out_metrics = os.path.join(results_dir, f"eval_{eval_file.replace('.txt','')}.json")
    with open(out_metrics, "w") as f:
        json.dump(results, f, indent=2)
    print(f"✅ Saved evaluation results to {out_metrics}")

    # Predict chi tiết
    preds_output = trainer.predict(eval_dataset)
    preds = np.argmax(preds_output.predictions, axis=-1)

    pred_dir = os.path.join(results_dir, "predictions")
    os.makedirs(pred_dir, exist_ok=True)
    out_preds = os.path.join(pred_dir, f"{eval_file.replace('.txt','')}_predictions.txt")

    with open(out_preds, "w", encoding="utf-8") as f:
        for i, (sent, pred_ids) in enumerate(zip(eval_sentences, preds)):
            word_ids = eval_dataset.encodings.word_ids(batch_index=i)
            prev_word_idx = None
            pred_tags = []
            for wid, pid in zip(word_ids, pred_ids):
                if wid is None or wid == prev_word_idx:
                    continue
                pred_tags.append(id2tag.get(int(pid), "O"))
                prev_word_idx = wid
            # đảm bảo đủ tag cho tất cả từ
            if len(pred_tags) < len(sent):
                pred_tags += ["O"] * (len(sent) - len(pred_tags))
            f.write(" ".join(f"{w}/{t}" for w, t in zip(sent, pred_tags)) + "\n")

    print(f"✅ Saved predictions to {out_preds}")

if __name__ == "__main__":
    main()