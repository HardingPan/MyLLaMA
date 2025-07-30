import torch
from torch import nn
from torch.nn import functional as F

from transformers import PretrainedConfig, PreTrainedModel
from transformers.modeling_outputs import CausalLMOutputWithPast
from typing import Tuple, Optional
import math

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

# LLaMA2 Attention
class Attention(nn.Module):
    def __init__(self, args: ModelConfig):
        super().__init__()
        # 根据是否指定n_kv_heads,确定用于键(key)和值(value)的头的数量
        self.n_kv_heads = args.n_heads if args.n_kv_heads is None else args.n_kv_heads
        assert args.n_heads % self.n_kv_heads == 0, "n_heads must be divisible by n_kv_heads"
        
        model_parallel_size = 1 # 假设没有模型并行
        self.n_local_heads = args.n_heads // model_parallel_size # 每个模型并行的头数
        self.n_local_qv_heads = self.n_kv_heads // model_parallel_size # 每个模型并行的 Q 和 V 头数
        self.n_rep = self.n_local_heads // self.n_local_qv_heads # 重复 K 和 V 的次数
        self.head_dim = args.dim // args.n_heads # 每个头的维度等于模型维度除以头数
        
        # 定义权重矩阵
        self.wq = nn.Linear(args.dim, args.n_heads * self.head_dim, bias=False)
        self.wk = nn.Linear(args.dim, self.n_kv_heads * self.head_dim, bias=False)
        self.wv = nn.Linear(args.dim, self.n_kv_heads * self.head_dim, bias=False)
        # 输出权重矩阵
        self.wo = nn.Linear(args.n_heads * self.head_dim, args.dim, bias=False)
        
        # 定义dropout层
        self.attn_dropout = nn.Dropout(args.dropout)
        self.resid_dropout = nn.Dropout(args.dropout)
        self.dropout = args.dropout
        
        # 是否使用 Flash Attention
        self.flash = hasattr(torch.nn.functional, "scaled_dot_product_attention")
        if not self.flash:
            # 如果 Flash Attention 不可用，使用标准的注意力机制，并设置 mask
            print("Warning: Flash Attention is not available. Using standard attention.")
            mask = torch.full((1, 1, args.max_seq_len, args.max_seq_len), float("-inf"))
            mask.triu(mask, diagonal=1)
            # 注册为模型的缓冲区
            self.register_buffer("mask", mask)
            
    def forward(self, x: torch.Tensor, freqs_cos: torch.Tensor, freqs_sin: torch.Tensor) -> torch.Tensor:
        bsz, seqlen, _ = x.shape # x为 (batch_size, sequence_length, embedding_dim)
        
        # 计算 Q、K、V
        xq, xk, xv = self.wq(x), self.wk(x), self.wv(x)
        # 调整 Q、K、V 的形状以适应多头注意力
        xq = xq.view(bsz, seqlen, self.n_local_heads, self.head_dim) # [bsz, seqlen, n_heads, head_dim]
        xk = xk.view(bsz, seqlen, self.n_local_qv_heads, self.head_dim) # [bsz, seqlen, n_kv_heads, head_dim]
        xv = xv.view(bsz, seqlen, self.n_local_qv_heads, self.head_dim) # [bsz, seqlen, n_kv_heads, head_dim]
        
        # 应用旋转嵌入 RoPE
        xq, xk = apply_rotary_emb(xq, xk, freqs_cos, freqs_sin) # 应用旋转嵌入
        
        # 重复 K 和 V 以匹配 Q 的头数
        xk = repeat_kv(xk, self.n_rep) # 重复 K
        xv = repeat_kv(xv, self.n_rep) # 重复 V
        
        # 将头作为批次维度处理
        xq = xq.transpose(1, 2) # [bsz, n_heads, seqlen, head_dim]
        xk = xk.transpose(1, 2) # [bsz, n_heads, seqlen, head_dim]
        xv = xv.transpose(1, 2) # [bsz, n_heads, seqlen, head_dim]
        
        if self.flash:
            output = torch.nn.functional.scaled_dot_product_attention(
                xq, xk, xv, 
                attn_mask=None,  # 如果需要遮蔽，可以在这里传入 self.mask
                dropout_p=self.dropout if self.training else 0.0, is_causal=True # causal mask
            )
        else:
            scores = torch.matmul(xq, xk.transpose(-2, -1)) / math.sqrt(self.head_dim) # 计算注意力分数
            assert hasattr(self, 'mask'), "Mask must be defined for standard attention."
            scores = scores + self.mask[:, :, :seqlen, :seqlen] # 添加遮蔽
            scores = F.softmax(scores.float(), dim=-1).type_as(xq) # 计算 softmax，type_as 保持数据类型一致为 xq 的类型
            scores = self.attn_dropout(scores) # 应用注意力 dropout
            output = torch.matmul(scores, xv) # 计算注意力输出
            
        # 恢复时间序列维度并合并头
        output = output.transpose(1, 2).contiguous()
        output = output.view(bsz, seqlen, -1) # [bsz, seqlen, n_heads * head_dim]
        
        output = self.wo(output) # 线性变换输出
        output = self.resid_dropout(output) # 应用残差 dropout
        return output
    
class MLP(nn.Module):
    def __init__(self, dim: int, hidden_dim: int, multiple_of: int, dropout: float):
        super().__init__()
        # 如果没有指定 hidden_dim，则默认为 dim 的 4 倍
        # 然后将其减少到2/3，最后确保它是 multiple_of 的倍数
        # 作用是为了确保 MLP 的隐藏层维度符合模型的设计要求
        # 这样可以在不同的模型配置中保持一致性
        if hidden_dim is None:
            hidden_dim = dim * 4
            hidden_dim = int(2 * hidden_dim / 3) # 为什么是 2/3？因为 LLaMA2 的 MLP 设计是这样的
            hidden_dim = multiple_of * ((hidden_dim + multiple_of - 1) // multiple_of) # 确保是 multiple_of 的倍数
            
        # 定义第一层线性变换，从 dim 到 hidden_dim
        self.w1 = nn.Linear(dim, hidden_dim, bias=False)
        # 定义第二层线性变换，从 hidden_dim 到 dim
        self.w2 = nn.Linear(hidden_dim, dim, bias=False)
        # 定义第三层线性变换，从 dim 到 hidden_dim
        self.w3 = nn.Linear(dim, hidden_dim, bias=False)
        # 定义 dropout 层
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x 的形状通常为 (batch_size, seq_len, dim)
        # 先通过第一层线性变换，然后应用激活函数
        # 结果乘以输入x通过第三层线性变换的结果
        # 最后通过第二层线性变换和 dropout
        
        # w3 让 MLP 具备了“门控”能力，使得每个神经元可以根据输入自适应地决定信息是否通过，从而提升模型表达力和训练稳定性
        # 这是 LLaMA2 MLP 的核心创新之一
        # 主分支和门控分支做逐元素相乘，实现门控机制（类似 GLU/Gated Linear Unit），让网络可以自适应地控制信息流通
        # [batch_size, seq_len, dim] -> [batch_size, seq_len, hidden_dim]
        # -> [batch_size, seq_len, dim]
        
        return self.dropout(self.w2(F.silu(self.w1(x)) * self.w3(x)))
    
class DecoderLayer(nn.Module):
    def __init__(self, layer_id: int, args: ModelConfig):
        super().__init__()
        self.n_heads = args.n_heads # 注意力头数
        self.dim = args.dim # 模型维度
        self.head_dim = args.dim // args.n_heads # 每个头的维度
        # 定义LLaMA2的注意力模块，用于多头注意力训练
        self.attention = Attention(args)
        # 定义LLaMA MLP模块，用于前馈神经网络
        self.feed_forward = MLP(
            dim=args.dim, 
            hidden_dim=args.hidden_dim, 
            multiple_of=args.multiple_of, 
            dropout=args.dropout
        )
        # 定义层的id
        self.layer_id = layer_id
        # 定义注意力和前馈网络的归一化层
        self.attention_norm = RMSNorm(dim=args.dim, eps=args.norm_eps)
        self.ffn_norm = RMSNorm(dim=args.dim, eps=args.norm_eps)
        
    def forward(self, x: torch.Tensor, freqs_cos: torch.Tensor, freqs_sin: torch.Tensor) -> torch.Tensor:
        # x 的形状通常为 (batch_size, seq_len, dim)
        # 首先，输入x经过注意力归一化层，然后进行多头注意力计算，结果与输入x相加得到h
        # 接着，结果经过前馈网络归一化层，然后通过MLP模块，结果与h相加得到输出
        h = x + self.attention(
            self.attention_norm(x), freqs_cos, freqs_sin
        )
        output = h + self.feed_forward(
            self.ffn_norm(h)
        )
        return output
    
class Transformer(PreTrainedModel):
	config_class = ModelConfig
	last_loss: Optional[torch.Tensor]
 
	def __init__(self, args: ModelConfig = None):
		super().__init__(args)
		# 初始化参数模型
		self.args = args
		# 如果没有传入 args，则使用默认的 ModelConfig
		self.config = args
		# 词汇表大小
		self.vocab_size = args.vocab_size
		# 层数
		self.n_layers = args.n_layers

		# 词嵌入层
		self.tok_embeddings = nn.Embedding(args.vocab_size, args.dim)
		# Dropout 层
		self.dropout = nn.Dropout(args.dropout)
		# Decoder 层列表
		self.layers = nn.ModuleList()
		for layer_id in range(args.n_layers):
			# 每一层都是一个 DecoderLayer
			self.layers.append(DecoderLayer(layer_id, args))
		# 归一化层
		self.norm = RMSNorm(dim=args.dim, eps=args.norm_eps)
		# 输出层
		self.output = nn.Linear(args.dim, args.vocab_size, bias=False)

		# 将词嵌入层的权重与输出层的权重共享，为什么可以共享？
		# 本质对偶
		# 词嵌入层：将词ID映射为向量（查表，查的是 embedding matrix 的某一行）。
		# 输出层：将隐藏向量映射为词表上每个词的得分（本质是与 embedding matrix 的每一行做点积）。
		# 这种写法在 PyTorch 中，实际上是让 tok_embeddings.weight 和 output.weight 这两个参数指向同一块内存
		# 即它们是绑定的（共享权重），不是简单的“赋值一次”
		# 只要你修改其中一个（比如反向传播更新参数），另一个也会同步变化
		self.tok_embeddings.weight = self.output.weight

		# 预计算相对位置嵌入的频率
		freqs_cos, freqs_sin = precompute_freqs_cis(self.args.dim // self.args.n_heads, self.args.max_seq_len)
		self.register_buffer("freqs_cos", freqs_cos, persistent=False) # 不需要保存到模型文件中
		self.register_buffer("freqs_sin", freqs_sin, persistent=False) # 不需要保存到模型文件中

		# 初始化模型参数
		self.apply(self._init_weights)
		# 对残差投影进行特殊的缩放初始化
		for pn, p in self.named_parameters():
			if pn.endswith('w3.weight') or pn.endswith('wo.weight'):
				# 这两组权重（w3.weight 和 wo.weight）在 LLaMA2 架构中对模型的训练稳定性和收敛速度影响较大
				# 采用更小的标准差（随层数增加而减小），可以防止深层网络中残差路径上的信号放大或消失，提升训练稳定性
				# 这种初始化方式是 LLaMA2 官方实现的推荐做法，属于“残差缩放初始化”思想
				torch.nn.init.normal_(p, mean=0.0, std=0.02/math.sqrt(2 * args.n_layers))

		# 初始化最后一次前向传播的损失属性
		self.last_loss = None
		self.OUT = CausalLMOutputWithPast() # 用于存储模型输出和过去的键值对
		self._no_split_modules = [name for name, _ in self.named_modules()] # 不进行分割的模块列表，这是为了在分布式训练时避免不必要的分割

	def _init_weights(self, module):
		# 初始化权重
		if isinstance(module, nn.Linear):
			torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
			if module.bias is not None:
				torch.nn.init.zeros_(module.bias)
		elif isinstance(module, nn.Embedding):
			torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

	def forward(
		self, 
		tokens: torch.Tensor,
		targets: Optional[torch.Tensor] = None,
		**keyargs
	) -> torch.Tensor:
		"""
		参数:
			tokens: 输入的 token ID 张量，形状为 (batch_size, seq_len)
			targets: 目标 token ID 张量，用于计算损失，形状为 (batch_size, seq_len)
			**kwargs: 其他参数（如 past_key_values 等）
		
		返回:
			模型输出张量，CausualLMOutputWithPast 类型
			如果 targets 不为 None，则包含 logits 和损失值
			如果 targets 为 None，则只返回 logits
			形状为 (batch_size, seq_len, vocab_size)
		"""
		if 'input_dix' in keyargs:
			# 如果传入了 input_dix，则使用它作为输入
			tokens = keyargs['input_dix']
		if 'attention_mask' in keyargs:
			# 如果传入了 attention_mask，则使用它作为注意力遮蔽
			attention_mask = keyargs['attention_mask']

		# 前向传播函数
		_bsz, seqlen = tokens.shape # 获取批次大小和序列长度
		# 通过词嵌入和 dropout 层
		h = self.tok_embeddings(tokens) # 词嵌入
		h = self.dropout(h) # dropout
		# 获取相对位置的嵌入
		freqs_cos = self.freqs_cos[:seqlen, :] # 获取当前序列长度的余弦位置编码
		freqs_sin = self.freqs_sin[:seqlen, :] # 获取当前序列长度的正弦位置编码

		# 通过decoder层
		for layer in self.layers:
			# 每一层都进行前向传播
			h = layer(h, freqs_cos, freqs_sin)
		# 最后通过归一化层
		h = self.norm(h) # 归一化处理

		if targets is not None:
			# 如果有目标，则计算损失
			logits = self.output(h) # 通过输出层得到 logits
			# 把 logits 张量（形状通常为 [batch_size, seq_len, vocab_size]）展平成二维 [batch_size * seq_len, vocab_size]，方便和 targets 对齐
			# targets.view(-1) 把目标 token id 展平成一维 [batch_size * seq_len]
			loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1), ignore_index=0, reduction='none')
		else:
			# 如果没有目标，则只返回 logits
			# 推理时的小优化，只对最后一个位置的输出进行前向传播
			logits = self.output(h[:, [-1], :]) # 只取最后一个位置的输出
			self.last_loss = None # 没有损失
		
  		# 设置输出
		self.OUT.__setitem__('logits', logits)
		self.OUT.__setitem__('last_loss', self.last_loss)
		return self.OUT

	@torch.inference_mode()
	def generate(self, idx, stop_id=None, max_new_tokens=256, temperature=1.0, top_k=None):
		"""
		给定输入序列 idx(形状为 (bz,seq_len) 的⻓整型张量),通过多次生成新 token 来完成序列
  		在 model.eval() 模式下运行。效率较低的采样版本,没有使用键k/v cache。
		idx: 输入序列，形状为 (batch_size, seq_len) 的token ID张量
		stop_id: 停止生成的特殊token ID（如EOS token）
		max_new_tokens: 最大生成token数量，默认256
		temperature: 控制生成随机性，越小越确定性，越大越随机
		top_k: Top-K采样，只从概率最高的k个token中选择
  		"""
		index = idx.shape(1) # 记录原始序列长度
		for _ in range(max_new_tokens):
			# 如果序列上下文过长，截断它到最大长度
			idx_cond = idx if idx.size(1) <= self.args.max_seq_len else idx[:, -self.args.max_seq_len:]

			# 前向传播获取序列中最后一个位置的 logits
			logits = self(idx_cond).logits
			logits = logits[:, -1, :] # 只保留最后一个时间步的输出

			if temperature == 0.0:
				# 选择最有可能的索引，返回张量中最大的k个值及其对应的索引
				_, idx_next = torch.topk(logits, k=1, dim=-1) # 直接选择概率最高的token
			else:
				# 缩放 logits 并应用 softmax
				logits = logits / temperature
				if top_k is not None: # Top-K过滤：将概率较低的token设为-inf
					v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
					logits[logits < v[:, [-1]]] = -float('Inf')
				probs = F.softmax(logits, dim=-1) # 转换为概率分布
				# 根据概率分布进行多项式采样（随机抽样）
				idx_next = torch.multinomial(probs, num_samples=1) # 按概率采样

			if idx_next == stop_id: # 遇到停止token就结束
				break

			# 将采样的索引添加到序列中并继续
			idx = torch.cat((idx, idx_next), dim=1) # 将新token添加到序列末尾

		return idx[:, index:] # 只返回生成的token

if __name__ == "__main__":
    # 测试RMSNorm
    args = ModelConfig()
    print(args)
    norm = RMSNorm(dim=args.dim, eps=args.norm_eps)
    x = torch.randn(1, 50, args.dim)
    output = norm(x)
    print(output.shape)  # 输出形状应为 (1, 50, args.dim)
    print(output.mean(), output.std())  # 输出均值和标准差
    
    # 测试旋转嵌入
    xq = torch.randn(1, 50, 6, 48) # bs, seq_len, dim//n_head, n_head_dim
    xk = torch.randn(1, 50, 6, 48) # bs, seq_len, dim//n_head, n_head_dim
    # 使用 precompute_freqs_cis 函数获取 sin和cos
    cos, sin = precompute_freqs_cis(288//6, 50)
    print(cos.shape, sin.shape)
    xq_out, xk_out = apply_rotary_emb(xq, xk, cos, sin)
    print(xq_out.shape, xk_out.shape)
    
    # 测试Attention
    attention_model = Attention(args)
    batch_size = 2
    seq_len = 50
    dim = args.dim
    x = torch.randn(batch_size, seq_len, dim) # 随机生成输入张量

    freqs_cos, freqs_sin = precompute_freqs_cis(dim // args.n_heads, seq_len)
    output = attention_model(x, freqs_cos, freqs_sin) # 通过注意力模型进行前向传播
    print(output.shape)  # 输出形状应为 (batch_size, seq_len, dim)
    print(output.mean(), output.std())  # 输出均值和标准差
    
    # 测试MLP
    mlp = MLP(dim=args.dim, hidden_dim=args.hidden_dim, multiple_of=args.multiple_of, dropout=args.dropout)
    x = torch.randn(4, 50, args.dim) # 随机生成输入张量
    output = mlp(x) # 通过 MLP 模型进行前向传播
    print(output.shape)  # 输出形状应为 (4, 50, args.dim
    
    # 测试DecoderLayer
    # 创建LLaMADecoderLayer实例
    decoderlayer = DecoderLayer(0, args)
    # 模拟输入数据
    x = torch.randn(2, 50, args.dim) # 假设 batch_size=2, seq_len=50, dim=args.dim
    freqs_cos, freqs_sin = precompute_freqs_cis(dim//args.n_heads, seq_len)
    out = decoderlayer(x, freqs_cos, freqs_sin)
    print(out.shape)  # 输出形状应为 (2, 50, args.dim)
    
    # 测试LLaMA2模型
    # LLaMA2Model.forward 接受两个参数,tokens和targets,其中tokens是输入的张量, 应为int类型
    x = torch.randint(0, 6144, (1, 50)) # [bs, seq_len]
    # 实例化LLaMA2Model
    model = Transformer(args=args)
    # 计算model的全部参数
    num_params = sum(p.numel() for p in model.parameters())
    print('Number of parameters:', num_params)
    out = model(x)
    print(out.logits.shape) # [batch_size, 1, vocab_size]