import torch
from torch import nn
from torch.nn import functional as F

from transformers import PretrainedConfig

class ModelConfig(PretrainedConfig):
    model_tyle = "Tiny-K"
    def __init__(
        self, 
        dim: int = 768,
        n_layers: int = 12,
        
        # 在标准的多头注意力（MHA）中，Query (Q)、Key (K)、Value (V) 的头数 相同（通常是 n_heads）
        # GQA (Grouped Query Attention) 的目标是减少计算量和显存
        # 让 Query 有更多的头数 n_q_heads，而 Key 和 Value 用更少的头数 n_kv_heads。
        # 这样在计算注意力时，Key 和 Value 的存储和计算压力大大降低。
        n_heads: int = 16, 
        n_kv_heads: int = 8,
        
        # vocab_size 是词汇表的大小，通常用于语言模型的输出层
        vocab_size: int = 6144,
        hidden_dim: int = None,
        multiple_of: int = 64,
        norm_eps: float = 1e-5,
        max_seq_len: int = 512,
        dropout: float = 0.0,
        flash_attn: bool = True,
        **kwargs,
    ):
        self.dim = dim
        self.n_layers = n_layers
        self.n_heads = n_heads
        self.n_kv_heads = n_kv_heads
        self.vocab_size = vocab_size
        self.hidden_dim = hidden_dim if hidden_dim is not None else dim * 4
        self.multiple_of = multiple_of
        self.norm_eps = norm_eps
        self.max_seq_len = max_seq_len
        self.dropout = dropout
        self.flash_attn = flash_attn
        super().__init__(**kwargs)
        
class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim)) # weight是一个可学习的参数
        
    def _nrom(self, x):
        # torch.rsqrt 是平方根的倒数
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)
    
    def forward(self, x):
        # 训练时，有些模型的输入张量 x 可能是 float16半精度 或 bfloat16
        # 因为这可以加快计算速度、节省显存
        # 但是在 x.pow(2).mean(-1) 这种操作中, 如果使用 float16 进行平方和平均
        # 可能会出现 溢出overflow 或 精度不足, 导致归一化计算不稳定
        # 解决办法：在归一化前把数据转换成 float32
        # 确保 RMS 计算有足够的数值精度。
        output = self._nrom(x.float()).type_as(x)
        return output * self.weight

# 通过 repeat_kv 把 K、V 的头数复制 n_rep 倍，使其与 Q 匹配
def repeat_kv(x: torch.tensor, n_rep: int) -> torch.tensor:
    # bs: batch size, slen: sequence length, n_kv_heads: number of key-value heads, 
    # head_dim: dimension of each head
    bs, slen, n_kv_heads, head_dim = x.shape
    
    if n_rep == 1:
        return x # 如果 n_rep 为 1，直接返回原始张量
    
    return (
        x[:, :, :, None, :] # 在 n_kv_heads 和 head_dim 之间插入一个维度 (bs, slen, n_kv_heads, 1, head_dim)
        .expand(bs, slen, n_kv_heads, n_rep, head_dim) # 在新插入的维度上复制 n_rep 次
        .reshape(bs, slen, n_kv_heads * n_rep, head_dim) # 合并 n_kv_heads 和 n_rep -> n_q_heads
    )




    
if __name__ == "__main__":
    
    # 测试 RMSNorm
    args = ModelConfig()
    print(args)
    norm = RMSNorm(dim=args.dim, eps=args.norm_eps)
    x = torch.randn(1, 50, args.dim)
    output = norm(x)
    print(output.shape)  # 输出形状应为 (1, 50, args.dim)
    print(output.mean(), output.std())  # 输出均值和标准差
    
    