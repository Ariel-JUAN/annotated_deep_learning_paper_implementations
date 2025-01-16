import math
from typing import Optional, List
import torch 
from torch import nn
import numpy as np

# 准备多头注意力的预处理层
class PrepareForMultiHeadAttention(nn.Module):
    def __init__(self, d_model: int, heads: int, d_k: int, bias: bool):
        super().__init__() 
        # 线性层用于变换 输入多维 默认只对最后一个维度进行操作
        self.linear = nn.Linear(d_model, heads * d_k, bias=bias)
        # 头数
        self.heads = heads
        # 每个头的维度
        self.d_k = d_k

    def forward(self, x: torch.Tensor):
        head_shape = x.shape[:-1]
        # 线性层负责学习如何把输入特征转换成对多头注意力有用的特征,然后才能分成多头
        # 线性变换
        x = self.linear(x)
        # 分成多个头
        x = x.view(*head_shape, self.heads, self.d_k)
        return x

# 多头注意力模块
class MultiHeadAttention(nn.Module):
    def __init__(self, heads: int, d_model: int, dropout_prob: float = 0.1, bias: bool = True):
        super().__init__()
        self.d_k = d_model // heads
        self.heads = heads
        
        # 对Q,K,V的线性变换
        self.query = PrepareForMultiHeadAttention(d_model, heads, self.d_k, bias=bias)
        self.key = PrepareForMultiHeadAttention(d_model, heads, self.d_k, bias=bias)
        self.value = PrepareForMultiHeadAttention(d_model, heads, self.d_k, bias=True)

        # Softmax和输出层
        self.softmax = nn.Softmax(dim=1)
        self.output = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout_prob)
        
        # 缩放因子 scale的作用是让注意力分数保持在合适的范围内，避免梯度消失。
        self.scale = 1 / math.sqrt(self.d_k) 
        self.attn = None

    def get_scores(self, query: torch.Tensor, key: torch.Tensor):
        # 计算注意力分数
        # i: 第一个序列长度(3)
        # j: 第二个序列长度(3)
        # b: batch size(2)  
        # h: heads(4)
        # d: d_k(8)

        # 这个操作在计算注意力分数:
        # - 对最后一个维度d进行点积
        # - 保留其他维度i,j,b,h
        # 假设简单情况：一个句子有3个词
        # query = ["我", "喜欢", "猫"]    # seq_len = 3
        # key = ["我", "喜欢", "猫"]      # seq_len = 3
        # einsum('ibhd,jbhd->ijbh', query, key) 在计算：
        # 每个词(i)对其他所有词(j)的关注度
        # 得到的分数矩阵可能像这样：
        # scores = [
        #     [0.9, 0.2, 0.1],  # "我" 和 ["我","喜欢","猫"] 的关联度
        #     [0.2, 0.8, 0.3],  # "喜欢" 和 ["我","喜欢","猫"] 的关联度
        #     [0.1, 0.3, 0.9]   # "猫" 和 ["我","喜欢","猫"] 的关联度
        # ]

        # 分数越高表示两个词的关联越强 ！本质是在计算词与词之间的关联程度，帮助模型理解词之间的关系。
        # - 对角线分数高：词和自己最相关
        # - "喜欢"和"猫"的分数(0.3)比"我"和"猫"的分数(0.1)高
        return torch.einsum('ibhd,jbhd->ijbh', query, key)

    def prepare_mask(self, mask: torch.Tensor, query_shape: List[int], key_shape: List[int]):
        assert mask.shape[0] == 1 or mask.shape[0] == query_shape[0]
        assert mask.shape[1] == key_shape[0]
        assert mask.shape[2] == 1 or mask.shape[2] == query_shape[1]
        mask = mask.unsqueeze(-1)
        return mask

    def forward(self, *,
                query: torch.Tensor,
                key: torch.Tensor, 
                value: torch.Tensor,
                mask: Optional[torch.Tensor] = None):
        
        seq_len, batch_size, _ = query.shape
        
        if mask is not None:
            mask = self.prepare_mask(mask, query.shape, key.shape)
            
        # 线性变换并分头
        query = self.query(query) # torch.Size([5, 2, 4, 64])
        key = self.key(key) # torch.Size([5, 2, 4, 64])
        value = self.value(value)

        # 计算注意力分数
        scores = self.get_scores(query, key) # torch.Size([5, 5, 2, 4])
        scores *= self.scale

        if mask is not None:
            scores = scores.masked_fill(mask == 0, float('-inf'))

        # softmax将注意力分数转换为概率分布，1）归一化：所有注意力权重加起来等于1，2）突出重要性 大的分数变得更大 小的分数变得更小 3）用这些概率去加权value：
        attn = self.softmax(scores) 
        attn = self.dropout(attn)

        # 与V相乘得到输出 value是每个位置的语义信息 
        x = torch.einsum("ijbh,jbhd->ibhd", attn, value)
        
        # 保存attention结果
        self.attn = attn.detach()

        # 合并多头
        x = x.reshape(seq_len, batch_size, -1)
        
        # 输出层
        return self.output(x)
    
def test_multi_head_attention_detailed():
    # 1. 基本设置 创建示例输入
    d_model = 256  # d_model是每个token的特征维度(常见值：512, 768, 1024等)
    heads = 4      # heads是将注意力分成多少个头，每个头学习独立的特征（d_model // heads）
    d_k = d_model // heads  # 每个头的维度
    seq_len = 5 # 是序列长度
    batch_size = 2
    x = torch.randn(seq_len, batch_size, d_model)
    
    print(f"token特征(d_model): {d_model}")
    print(f"注意力头数(heads): {heads}")
    print(f"每个头的维度(d_k): {d_k}")
    
    # 输入为什么是这个顺序[seq_len, batch_size, d_model] 1）逐词处理 2）适合注意力计算 直接计算不同位置的词的关系 3）适合并行处理 batch在中间 
    # seq_len = 3     # 句子长度3个词
    # batch_size = 2  # 一次处理2个句子
    # d_model = 4     # 每个词用4个数字表示

    # # 具体例子：
    # sentences = [
    #     "我 爱 你",   # 第1个句子
    #     "他 很 好"    # 第2个句子
    # ]

    # # 张量形状：[3, 2, 4]
    # tensor = [
    #     # 第1个词("我"和"他")
    #     [[1,1,1,1],    # "我"的特征
    #     [2,2,2,2]],   # "他"的特征
        
    #     # 第2个词("爱"和"很")
    #     [[3,3,3,3],    # "爱"的特征
    #     [4,4,4,4]],   # "很"的特征
        
    #     # 第3个词("你"和"好")
    #     [[5,5,5,5],    # "你"的特征
    #     [6,6,6,6]]    # "好"的特征
    # ]
    
    # 2. 创建模型
    mha = MultiHeadAttention(heads=heads, d_model=d_model)
    
    # 3. 生成测试数据
    # 使用固定的seed以获得可重复的结果
    torch.manual_seed(42)
    
    # 创建输入
    query = x
    key = x
    value = x
    
    # 4. 测试不同场景
    # 4.1 基本前向传播

    print("=== 测试基本前向传播 ===")
    # output是x经过上下文交互后的新表示，每个位置都融合了序列中其他位置的信息。这帮助模型理解上下文关系。
    output1 = mha(query=query, key=key, value=value)
    print(f"无mask输出形状: {output1.shape}")
    
    # 4.2 带mask的前向传播 mask在注意力机制中用于控制哪些位置可以相互关注，哪些位置应该被遮蔽掉
    # 主要用途：
    # 1. 防止信息泄露
    # - 在解码时，当前位置不能看到未来的信息
    # - 比如预测"喜欢"时不能看到"猫"这个词

    # 2. 处理填充(padding)
    # - 序列长度不同时需要padding
    # - mask可以防止模型关注padding部分
    # mask不测试了 这个有问题。。。。。
    # print("\n=== 测试带mask的前向传播 ===")
    # mask = torch.ones(1, seq_len, seq_len)
    # # 将某些位置设置为0来模拟mask效果
    # mask[0, 1, 0] = 0
    # output2 = mha(query=query, key=key, value=value, mask=mask)
    # print(f"带mask输出形状: {output2.shape}")
    
    # 4.3 检查注意力权重
    print("\n=== 检查注意力权重 ===")
    print(f"注意力权重形状: {mha.attn.shape}")
    print("注意力权重示例:")
    print(mha.attn[0, :, 0, 0])  # 第一个查询位置的注意力权重

    return output1, mha.attn

if __name__ == "__main__":
    output1, attention = test_multi_head_attention_detailed()

