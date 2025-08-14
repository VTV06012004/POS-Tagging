from torch.utils.data import Dataset
import torch
from transformers import AutoTokenizer, AutoModelForTokenClassification

class POSDataset(Dataset):
    def __init__(self, sentences, tags, tokenizer, tag2id, max_len=128):
        self.sentences = sentences
        self.tags = tags
        self.tokenizer = tokenizer
        self.tag2id = tag2id
        self.max_len = max_len

    def __len__(self):
        return len(self.sentences)

    def __getitem__(self, idx):
        words = self.sentences[idx]
        tags = self.tags[idx]

        encoding = self.tokenizer(
            words,
            is_split_into_words=True,
            return_offsets_mapping=False,
            padding='max_length',
            truncation=True,
            max_length=self.max_len,
            return_tensors='pt'
        )

        labels = [-100] * len(encoding["input_ids"][0])  # -100 để ignore khi tính loss

        word_ids = encoding.word_ids(batch_index=0)  # dùng để align tag
        previous_word_idx = None
        for i, word_idx in enumerate(word_ids):
            if word_idx is None:
                continue
            elif word_idx != previous_word_idx:
                tag = tags[word_idx]
                labels[i] = self.tag2id[tag]
            previous_word_idx = word_idx

        encoding["labels"] = torch.tensor(labels)

        return {
            "input_ids": encoding["input_ids"].squeeze(),
            "attention_mask": encoding["attention_mask"].squeeze(),
            "labels": encoding["labels"].squeeze()
        }

def load_model_and_tokenizer(model_name, tag2id):
    """
    model_name: tên mô hình từ HuggingFace (vd: 'bert-base-cased')
    tag2id: dict ánh xạ tag → số
    """
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForTokenClassification.from_pretrained(
        model_name,
        num_labels=len(tag2id),
        id2label={str(v): k for k, v in tag2id.items()},
        label2id=tag2id
    )
    return model, tokenizer