import torch
from torch import nn
from torch.nn import functional as F

from typing import Tuple


# 生成长度为 end 的序列位置编码信息，用正余弦函数映射到 dim 维空间（通常是 head_dim）
# 此处的dim应为 dim//n_head,因为我们是对每个head进行旋转嵌入
def precompute_freqs_cis(dim: int, end: int, theta: float = 10000.0) -> torch.Tensor:
    """
    dim: 向量维度(一般是每个注意力头的维度 head_dim, 即 hidden_dim // n_heads)
    end: 最大序列长度(即位置编码要覆盖的 token 数)
    theta: 基准频率参数(一般是 10000.0, 和 Transformer 的正余弦位置编码类似)
    """
    # torch.arange(0, dim, 2)
    # 生成一个从 0 开始、步长为 2 的序列： [0, 2, 4, ..., dim-2]
    # 这里取步长为 2 是因为后续 RoPE 会把向量的偶数维和奇数维看作 二维坐标 (x, y)
    # [: (dim // 2)] 取前一半长度，即维度的一半，得到 dim // 2 个频率因子
    # .float() / dim 将序列转换为浮点数，并除以 dim，得到一个范围为 [0, 2/dim, 4/dim, ...] 的数列
    # theta ** ( ... ) 计算theta^(i/dim)，得到每个频率因子的缩放, 类似于原始 Transformer 中的正余弦位置编码
    # 1.0 / (...) 最后取倒数，得到最终的频率值（对应波长的倒数）
    freqs = 1.0 / (theta ** (torch.arange(0, dim, 2)[: (dim // 2)].float() / dim))
    
    # 生成时间（位置）序列: 生成一个从 0 到 end 的序列,⻓度为 end
    t = torch.arange(end, device=freqs.device)
    
    # 计算外积,得到一个二维矩阵,每一行是t的元素乘以freqs的元素
    freqs = torch.outer(t, freqs).float() # 得到 每个位置 i 对应的每个频率分量 j 的“相位”
    
    # 计算频率的余弦值,得到实部
    freqs_cos = torch.cos(freqs)
    # 计算频率的正弦值,得到虚部
    freqs_sin = torch.sin(freqs)
    
    return freqs_cos, freqs_sin

# 调整 freqs_cis 的形状，使其能够与 x 在多维张量上正确广播（broadcast），方便进行逐元素的正弦/余弦旋转操作
# 这种操作常用于 RoPE（Rotary Position Embedding），因为位置编码 freqs_cis 需要和 Q/K 张量在特定维度对齐
def reshape_for_broadcast(freqs_cis: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
    """
    freqs_cis: 来自 precompute_freqs_cis 计算的 (seq_len, head_dim) 形状的正弦/余弦矩阵
    x: 一般是 Query 或 Key 张量，形状通常为 x.shape=(batch_size,seq_len,n_heads,head_dim)或者(batch_size, seq_len, head_dim)
    """
    ndim = x.ndim
    assert 0 <= 1 < ndim #  确认 x 至少有 2 个维度
    # 确认 freqs_cis 的第一维与 x 的序列长度 seq_len 相同，最后一维与 head_dim 相同
    assert freqs_cis.shape == (x.shape[1], x.shape[-1])
    
    # 构造一个新的形状,除了第二维和最后一维,其他维度都为1,这样做是为了能够将freqs_cis与x进行广播操作
    # enumerate(x.shape) 会返回 (i, d)，其中 i 是维度索引，d 是对应的维度大小
    # i == 1 or i == ndim - 1：
    # 第 1 个维度（序列长度 seq_len）要保留。
    # 最后一个维度（head_dim）也要保留
    # 其他维度设置为 1, 比如 batch_size 和 n_heads 维度都设为 1，这样 freqs_cis 就能通过广播扩展成和 x 形状相同
    shape = [d if i == 1 or i == ndim - 1 else 1 for i, d in enumerate(x.shape)]
    
    return freqs_cis.view(shape) # 调整freqs_cis的形状,使其可以与x进行广播操作

def apply_rotary_emb(
    xq: torch.Tensor,
    xk: torch.Tensor,
    freqs_cos: torch.Tensor,
    freqs_sin: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    xq: Query 张量，形状通常为 (batch_size, seq_len, n_heads, head_dim)
    xk: Key 张量，形状通常为 (batch_size, seq_len, n_heads, head_dim)
    freqs_cos: 预计算的余弦位置编码，形状为 (seq_len, head_dim)
    freqs_sin: 预计算的正弦位置编码，形状为 (seq_len, head_dim)
    
    返回值是应用 RoPE 后的 Query 和 Key 张量
    """
    xq_r, xq_i = xq.float().reshape(xq.shape[:-1] + (-1, 2)).unbind(dim=-1) # 将 xq 分解为实部和虚部
    xk_r, xk_i = xk.float().reshape(xk.shape[:-1] + (-1, 2)).unbind(dim=-1) # 将 xk 分解为实部和虚部
    
    freqs_cos = reshape_for_broadcast(freqs_cos, xq_r) # 调整 freqs_cos 的形状以便广播
    freqs_sin = reshape_for_broadcast(freqs_sin, xq_r) # 调整 freqs_sin 的形状以便广播
    
    # 应用旋转,分别计算旋转后的实部和虚部
    xq_out_r = xq_r * freqs_cos - xq_i * freqs_sin
    xq_out_i = xq_r * freqs_sin + xq_i * freqs_cos
    xk_out_r = xk_r * freqs_cos - xk_i * freqs_sin
    xk_out_i = xk_r * freqs_sin + xk_i * freqs_cos
    
    # 将最后两个维度合并,并还原为原始张量的形状
    xq_out = torch.stack([xq_out_r, xq_out_i], dim=-1).flatten(3)
    xk_out = torch.stack([xk_out_r, xk_out_i], dim=-1).flatten(3)
    
    return xq_out.type_as(xq), xk_out.type_as(xk)