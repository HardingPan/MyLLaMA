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
# from modelscope.msdatasets import MsDataset

# 方式1：使用Hugging Face数据集
def load_wikitext_dataset():
    """加载WikiText数据集"""
    # dataset = MsDataset.load('wikitext', subset_name='wikitext-2-v1', split='train')
    dataset = load_dataset("wikitext", "wikitext-103-v1", split="train")
    return dataset

def batch_iterator_from_dataset(dataset, batch_size=1000):
    """从数据集创建批次迭代器"""
    for i in range(0, len(dataset), batch_size):
        yield dataset[i:i+batch_size]["text"]

# 方式2：从JSONL文件读取
def read_texts_from_jsonl(file_path):
    """从JSONL文件读取文本"""
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            data = json.loads(line)
            yield data['text']

# 方式3：从普通文本文件读取
def load_text_from_files(path_list):
    """从多个文本文件加载数据"""
    text_data = []
    for file_path in path_list:
        with open(file_path, 'r', encoding='utf-8') as file:
            text_data.extend(file.readlines())
    return text_data

def batch_iterator_from_text(text_data, batch_size=1000):
    """从文本数据创建批次迭代器"""
    for i in range(0, len(text_data), batch_size):
        yield text_data[i:i+batch_size]

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
            "{% endif %}"
            "{% endfor %}"
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

def train_tokenizer_from_dataset(save_dir: str, vocab_size: int = 8192) -> None:
    """使用WikiText数据集训练tokenizer"""
    os.makedirs(save_dir, exist_ok=True)
    
    # 初始化tokenizer
    tokenizer = Tokenizer(models.BPE(unk_token="<unk>"))
    tokenizer.normalizer = NFKC()
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
        vocab_size=vocab_size, 
        special_tokens=special_tokens, 
        min_frequency=2, 
        show_progress=True, 
        initial_alphabet=pre_tokenizers.ByteLevel.alphabet()
    )
    
    # 加载数据集
    print("Loading WikiText dataset...")
    dataset = load_wikitext_dataset()
    
    # 训练tokenizer
    print(f"Training tokenizer with vocab_size={vocab_size}")
    tokenizer.train_from_iterator(
        batch_iterator_from_dataset(dataset), 
        trainer=trainer, 
        length=len(dataset)
    )
    
    # 验证特殊token映射
    print("Verifying special token mappings...")
    print(f"<unk> token ID: {tokenizer.token_to_id('<unk>')}")
    print(f"<s> token ID: {tokenizer.token_to_id('<s>')}")
    print(f"</s> token ID: {tokenizer.token_to_id('</s>')}")
    print(f"<|im_start|> token ID: {tokenizer.token_to_id('<|im_start|>')}")
    print(f"<|im_end|> token ID: {tokenizer.token_to_id('<|im_end|>')}")
    
    # 保存tokenizer文件
    tokenizer.save(os.path.join(save_dir, "tokenizer.json"))
    
    # 创建配置文件
    create_tokenizer_config(save_dir)
    print(f"Tokenizer saved to {save_dir}")

def train_tokenizer_from_jsonl(data_path: str, save_dir: str, vocab_size: int = 8192) -> None:
    """从JSONL文件训练tokenizer"""
    os.makedirs(save_dir, exist_ok=True)
    
    # 初始化tokenizer
    tokenizer = Tokenizer(models.BPE(unk_token="<unk>"))
    tokenizer.normalizer = NFKC()
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
        vocab_size=vocab_size, 
        special_tokens=special_tokens, 
        min_frequency=2, 
        show_progress=True, 
        initial_alphabet=pre_tokenizers.ByteLevel.alphabet()
    )
    
    # 训练tokenizer
    print(f"Training tokenizer with data from {data_path}")
    text_iterator = read_texts_from_jsonl(data_path)
    tokenizer.train_from_iterator(
        text_iterator, 
        trainer=trainer, 
        length=sum(1 for _ in open(data_path, 'r'))  # 计算行数
    )
    
    # 验证特殊token映射
    print("Verifying special token mappings...")
    for token in special_tokens:
        token_id = tokenizer.token_to_id(token)
        print(f"{token} token ID: {token_id}")
    
    # 保存tokenizer文件
    tokenizer.save(os.path.join(save_dir, "tokenizer.json"))
    create_tokenizer_config(save_dir)
    print(f"Tokenizer saved to {save_dir}")

def test_tokenizer(tokenizer_path: str):
    """测试训练好的tokenizer"""
    # 加载tokenizer
    tokenizer = PreTrainedTokenizerFast(tokenizer_file=os.path.join(tokenizer_path, "tokenizer.json"))
    
    # 测试文本
    test_texts = [
        "Hello, world!",
        "<|im_start|>user\nHello!<|im_end|>\n<|im_start|>assistant\nHi there!<|im_end|>",
        "This is a test sentence for tokenization."
    ]
    
    print("Testing tokenizer:")
    for text in test_texts:
        tokens = tokenizer.tokenize(text)
        token_ids = tokenizer.encode(text)
        decoded = tokenizer.decode(token_ids)
        
        print(f"Original: {text}")
        print(f"Tokens: {tokens}")
        print(f"Token IDs: {token_ids}")
        print(f"Decoded: {decoded}")
        print("-" * 50)
        

def eval_tokenizer(tokenizer_path: str) -> None:
    """评估tokenizer功能"""
    try:
        tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
    except Exception as e:
        print(f"Error loading tokenizer: {e}")
        return
    
    # 测试基本属性
    print("\n=== Tokenizer基本信息 ===")
    print(f"Vocab size: {len(tokenizer)}")  
    print(f"Special tokens: {tokenizer.all_special_tokens}")  
    print(f"Special token IDs: {tokenizer.all_special_ids}")  
    
    # 测试聊天模板  
    messages = [  
                {"role": "system", "content": "你是一个AI助手。"},  
                {"role": "user", "content": "How are you?"},  
                {"role": "assistant", "content": "I'm fine, thank you. and you?"},  
                {"role": "user", "content": "I'm good too."},  
                {"role": "assistant", "content": "That's great to hear!"},  
            ]  
    
    print("\n=== 聊天模板测试 ===")  
    prompt = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        # add_generation_prompt=True
    )
    print("Generated prompt:\n", prompt, sep="")  
    # 测试编码解码  print("\n=== 编码解码测试 ===")  
    encoded = tokenizer(prompt, truncation=True, max_length=256)  
    decoded = tokenizer.decode(encoded["input_ids"], skip_special_tokens=False)  
    print("Decoded text matches original:", decoded == prompt)  
    
    # 测试特殊token处理  
    print("\n=== 特殊token处理 ===")  
    test_text = "<|im_start|>user\nHello<|im_end|>"  
    encoded = tokenizer(test_text).input_ids  
    decoded = tokenizer.decode(encoded)  
    print(f"Original: {test_text}")  
    print(f"Decoded: {decoded}")  
    print("Special tokens preserved:", decoded == test_text)

if __name__ == "__main__":
    # 使用示例
    
    # 方式1：使用WikiText数据集训练
    # print("Training tokenizer from WikiText dataset...")
    # train_tokenizer_from_dataset("./my_tokenizer", vocab_size=8192)
    
    # 方式2：从JSONL文件训练（如果你有JSONL数据）
    # train_tokenizer_from_jsonl("your_data.jsonl", "./my_tokenizer", vocab_size=8192)
    
    # 测试tokenizer
    print("\nTesting trained tokenizer...")
    # test_tokenizer("./my_tokenizer")
    eval_tokenizer("./my_tokenizer")