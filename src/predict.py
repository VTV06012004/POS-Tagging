import torch
from transformers import AutoTokenizer, AutoModelForTokenClassification
from torch.nn.functional import softmax

# ================== CÀI ĐẶT MODEL ==================
MODEL_DIR = "models/bert-pos"  # đường dẫn model đã train

# Load tokenizer + model
tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR)
model = AutoModelForTokenClassification.from_pretrained(MODEL_DIR)

# Load id2tag mapping từ config
id2tag = model.config.id2label

# ================== HÀM DỰ ĐOÁN ==================
def predict_tags(sentence: str):
    """
    Input: câu raw string
    Output: List[Tuple[word, tag]]
    """
    # Tokenize
    tokens = tokenizer(sentence.split(), is_split_into_words=True,
                       return_tensors="pt", truncation=True)

    with torch.no_grad():
        outputs = model(**tokens)
        logits = outputs.logits
        preds = torch.argmax(softmax(logits, dim=-1), dim=-1)[0].tolist()

    # Map token -> label, chỉ lấy token đầu mỗi từ
    word_ids = tokens.word_ids()
    word_to_label = []
    for idx, word_id in enumerate(word_ids):
        if word_id is None:
            continue
        if idx == 0 or word_id != word_ids[idx - 1]:
            word = sentence.split()[word_id]
            label = id2tag[preds[idx]]
            word_to_label.append((word, label))
    return word_to_label

# ================== CHẠY INTERACTIVE ==================
if __name__ == "__main__":
    print("Nhập câu để POS Tagging (nhập 'q' để thoát)")
    while True:
        sent = input(">>> ")
        if sent.lower() == "q":
            break
        results = predict_tags(sent)
        for w, t in results:
            print(f"{w}\t{t}")
