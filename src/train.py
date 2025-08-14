import os
import torch
from transformers import BertForTokenClassification, BertTokenizerFast, Trainer, TrainingArguments
from preprocess import build_tag_mapping
from data_load import load_data_from_txt
from model import load_model_and_tokenizer
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset
from sklearn.metrics import precision_recall_fscore_support
from transformers import EarlyStoppingCallback

# 1. Load dữ liệu
train_sentences, train_tags = load_data_from_txt("../data/processed/train.txt")
val_sentences, val_tags = load_data_from_txt("../data/processed/dev.txt")
test_sentences, test_tags = load_data_from_txt("../data/processed/test.txt")
tag2id, id2tag = build_tag_mapping(train_tags)
num_labels = len(tag2id)

# 2. Load model và tokenizer
model_name = "bert-base-cased"
model, tokenizer = load_model_and_tokenizer(model_name, tag2id)

# 3. Tokenize và căn chỉnh nhãn
class POSDataset(Dataset):
    def __init__(self, sentences, tags, tokenizer, tag2id, max_length=128):
        self.encodings = tokenizer(sentences,
                                   is_split_into_words=True,
                                   return_offsets_mapping=True,
                                   padding=True,
                                   truncation=True,
                                   max_length=max_length)
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
        self.encodings.pop("offset_mapping")  # không cần nữa

    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        item["labels"] = torch.tensor(self.labels[idx])
        return item

    def __len__(self):
        return len(self.labels)

# 4. Chia train/test
train_sents, val_sents, train_lbls, val_lbls = train_test_split(
    train_sentences, train_tags, test_size=0.1, random_state=42
)
train_dataset = POSDataset(train_sents, train_lbls, tokenizer, tag2id)
val_dataset = POSDataset(val_sents, val_lbls, tokenizer, tag2id)

# 5. TrainingArguments
training_args = TrainingArguments(
    output_dir="./results",
    num_train_epochs=10,  # tăng lên nếu dùng early stopping
    per_device_train_batch_size=4,
    per_device_eval_batch_size=8,
    eval_strategy="epoch",
    save_strategy="epoch",
    logging_dir="./logs",
    logging_steps=10,
    learning_rate=5e-5,
    weight_decay=0.01,  # thêm weight decay
    load_best_model_at_end=True,
    metric_for_best_model="f1",  # chọn metric tốt nhất
    greater_is_better=True,
    report_to="none",
)

# 6. Metrics
def compute_metrics(p):
    predictions, labels = p
    predictions = predictions.argmax(axis=-1)

    true_labels, true_preds = [], []
    for pred, label in zip(predictions, labels):
        for p_i, l_i in zip(pred, label):
            if l_i != -100:
                true_labels.append(l_i)
                true_preds.append(p_i)

    precision, recall, f1, _ = precision_recall_fscore_support(
        true_labels, true_preds, average="micro", zero_division=0
    )
    accuracy = (torch.tensor(true_labels) == torch.tensor(true_preds)).float().mean().item()

    return {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1,
    }

# 7. Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
    tokenizer=tokenizer,
    compute_metrics=compute_metrics,
    callbacks=[EarlyStoppingCallback(early_stopping_patience=2)]
)

# 8. Train
trainer.train()

# 9. Save model
model.save_pretrained("models/bert-pos")
tokenizer.save_pretrained("models/bert-pos")