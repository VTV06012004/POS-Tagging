def load_data_from_txt(txt_path):
    sentences = []
    tags = []
    with open(txt_path, 'r', encoding='utf-8') as f:
        words, pos_tags = [], []
        for line in f:
            line = line.strip()
            if not line:
                if words:
                    sentences.append(words)
                    tags.append(pos_tags)
                    words, pos_tags = [], []
                continue
            parts = line.split('\t')
            if len(parts) != 2:
                continue
            word, tag = parts
            words.append(word)
            pos_tags.append(tag)
    return sentences, tags
