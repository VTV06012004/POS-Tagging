import os

def conllu_to_txt(conllu_path, txt_path):
    with open(conllu_path, 'r', encoding='utf-8') as infile, open(txt_path, 'w', encoding='utf-8') as outfile:
        for line in infile:
            line = line.strip()
            if not line or line.startswith('#'):
                outfile.write('\n' if not line else '')
                continue
            parts = line.split('\t')
            if len(parts) < 4 or '-' in parts[0] or '.' in parts[0]:
                continue
            word = parts[1]
            pos = parts[3]
            outfile.write(f"{word}\t{pos}\n")

if __name__ == "__main__":
    input_dir = "data/raw"
    output_dir = "data/processed"
    os.makedirs(output_dir, exist_ok=True)

    conllu_to_txt(f"{input_dir}/train.conllu", f"{output_dir}/train.txt")
    conllu_to_txt(f"{input_dir}/dev.conllu", f"{output_dir}/dev.txt")
    conllu_to_txt(f"{input_dir}/test.conllu", f"{output_dir}/test.txt")
