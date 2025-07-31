import random
import json
import os
from transformers import AutoTokenizer, PreTrainedTokenizerFast
from tokenizers import (
    decoders, 
    models, 
    pre_tokenizers, 
    trainers, 
    Tokenizer, 
)
from datasets import load_dataset
from tokenizers.normalizers import NFKC
from typing import Generator

import json

dataset = load_dataset("wikitext", "wikitext-103-v1", split="train")
# 准备训练数据
def batch_iterator(batch_size=1000):
    for i in range(0, len(dataset), batch_size):
        yield dataset[i:i+batch_size]["text"]
        

def read_texts_from_jsonl(file_path):
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            data = json.loads(line)
            yield data['text']

# 读取本地文本数据集
def load_text_from_files(path_list):
    text_data = []
    for file_path in path_list:
        with open(file_path, 'r', encoding='utf-8') as file:
            text_data.extend(file.readlines())
    return text_data
def batch_iterator(text_data, batch_size=1000):
    for i in range(0, len(text_data), batch_size):
        yield text_data[i:i+batch_size]


# Tokenizer配置文件
def create_tokenizer_config(save_dir: str) -> None:  
    """创建完整的tokenizer配置文件"""  
    config = {
        "add_bos_token": False,
        "add_eos_token": False,
        "add_prefix_space": False,
        "bos_token": "<|im_start|>",
        "eos_token": "<|im_end|>",
        "pad_token": "<|im_end|>",
        "unk_token": "<unk>", 
        "model_max_length": 1000000000000000019884624838656,
        "clean_up_tokenization_spaces": False,
        "tokenizer_class": "PreTrainedTokenizerFast",
        "chat_template": (
            "{% for message in messages %}"
            "{% if message['role'] == 'system' %}"
            "<|im_start|>system\n{{ message['content'] }}<|im_end|>\n"
            "{% elif message['role'] == 'user' %}"
            "<|im_start|>user\n{{ message['content'] }}<|im_end|>\n"
            "{% elif message['role'] == 'assistant' %}"
            "<|im_start|>assistant\n{{ message['content'] }}<|im_end|>\n"
            "{% endif %}"  "{% endfor %}"
            "{% if add_generation_prompt %}"
            "{{ '<|im_start|>assistant\n' }}"
            "{% endif %}"
        )
    }
    
    # 保存主配置文件
    with open(os.path.join(save_dir, "tokenizer_config.json"), "w", encoding="utf-8") as f:
        json.dump(config, f, ensure_ascii=False, indent=4)
    # 创建special_tokens_map.json
    special_tokens_map = {
        "bos_token": "<|im_start|>",
        "eos_token": "<|im_end|>",
        "unk_token": "<unk>",
        "pad_token": "<|im_end|>",
        "additional_special_tokens": ["<s>", "</s>"]
    }
    with open(os.path.join(save_dir, "special_tokens_map.json"), "w", encoding="utf-8") as f:
        json.dump(special_tokens_map, f, ensure_ascii=False, indent=4)

# 训练BPE Tokenizer
def train_tokenizer(data_path: str, save_dir: str, vocab_size: int = 8192) -> None:
    """训练并保存自定义tokenizer"""
    os.makedirs(save_dir, exist_ok=True)
    
    # 初始化tokenizer
    tokenizer = Tokenizer(models.BPE(unk_token="<unk>"))
    tokenizer.normalizer = NFKC() # 添加文本规范化
    tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    tokenizer.decoder = decoders.ByteLevel()
    
    # 配置特殊token
    special_tokens = [
        "<unk>",
        "<s>",
        "</s>",
        "<|im_start|>",
        "<|im_end|>"
    ]
    
    # 配置训练器
    trainer = trainers.BpeTrainer(
        vocab_size = vocab_size, 
        special_tokens=special_tokens, 
        min_frequency=2, 
        show_progress=True, 
        initial_alphabet=pre_tokenizers.BtyeLevel.alphabet()
    )
    
    # 训练tokenizer
    print(f"Training tokenizer with data from {data_path}")
    text = read_texts_from_jsonl(data_path)