import random
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader, TensorDataset
from transformers import GPT2LMHeadModel, GPT2Tokenizer, GPT2Config
from typing import List
from loggers.get_logger import get_logger

from data import TOKENIZER_PATH
logger = get_logger("Dataset")


class ChatDataset(Dataset):
    def __init__(self, data: List[str], tokenizer):
        logger.info("Tokenizing and building input...")
        self.tokenized_text = [tokenizer.convert_tokens_to_ids(tokenizer.tokenize(text)) for text in data]

    def __len__(self):
        return len(self.tokenized_text)

    def __getitem__(self, item):
        return torch.tensor(self.tokenized_text[item])


def get_data_loader(data, tokenizer, batch_size: int, pin_memory: bool = True):
    """ Prepare the dataset for training and evaluation """
    dataset = ChatDataset(data, tokenizer)
    logger.info("Train dataset: {:,} samples".format(len(dataset)))
    logger.info("Build dataloaders")
    data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, pin_memory=pin_memory, num_workers=2)
    return data_loader
