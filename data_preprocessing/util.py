import json
from typing import List
from tokenizers import ByteLevelBPETokenizer


def write_txt(data: str, path: str) -> None:
    with open(path, 'w') as f:
        f.write(data)


def read_txt(path: str) -> List[str]:
    with open(path, 'r') as f:
        return json.loads(f.read())


def train_bpe_tokenizer(txt_file_path: str, save_path, vocab_size: int = 52_000, min_frequency: int = 2) -> None:
    tokenizer = ByteLevelBPETokenizer()
    tokenizer.train(files=[txt_file_path], vocab_size=vocab_size, min_frequency=min_frequency,
                    special_tokens=["<pad>", "<s>", "</s>", "<unk>", "<mask>",  "<Kate>", "<Daria>"])
    tokenizer.save_model(save_path)
