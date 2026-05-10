import torch.nn as nn
import torch
import numpy as np

# Scaled Dot Product Attention
class ScaledDotProductAttention(nn.Module):
    def __init__(self, d_k):
        super(ScaledDotProductAttention, self).__init__()
        self.d_k = d_k

    def forward(self, Q, K, V, attn_mask):
        # 输入进来的维度分别是
        # Q: [batch_size * n_heads * len_q * d_k]
        # K: [batch_size * n_heads * len_k * d_k]
        # V: [batch_size * n_heads * len_k * d_v]

        scores = torch.matmul(Q, K.transpose(-1, -2)) / np.sqrt(self.d_k)   # scores: [batch_size * n_heads * len_q * len_k]

        # attn_mask所有为True的位置填充为-∞
        scores.masked_fill_(attn_mask, -1e9)   # 把被mask的地方置为负无穷，softmax之后基本就是0，对Q的单词不起作用
        attn = nn.Softmax(dim=-1)(scores)      # 每一横行做softmax，这样每一行（每个Q）对所有K的注意力权重和为1
        context = torch.matmul(attn, V)        # [batch_size * n_heads * len_q * d_v]。每个Q位置都获得了一个融合了全局信息的向量
        return context, attn

# Multi Head Attention
class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, d_k, d_v, n_heads):
        super(MultiHeadAttention, self).__init__()
        self.d_model = d_model
        self.d_k = d_k
        self.d_v = d_v
        self.n_heads = n_heads

        # 参数矩阵Wq, Wk, Wv，将输入映射到多头空间
        self.W_Q = nn.Linear(d_model, d_k * n_heads)
        self.W_K = nn.Linear(d_model, d_k * n_heads)   # 要保证最后得到的QK矩阵维度相同
        self.W_V = nn.Linear(d_model, d_v * n_heads)
        self.linear = nn.Linear(n_heads * d_v, d_model)   # 把多头输出融合回原始维度
        self.layer_norm = nn.LayerNorm(d_model)    # 残差连接后做层归一化
        self.attention = ScaledDotProductAttention(d_k)

    def forward(self, Q, K, V, attn_mask):  # 注意力层的输入是最原始的QKV，而不是经过 W 权重矩阵映射过的
        # 多头分为几个步骤：首先映射分头，然后计算attn_scores，然后计算attn_value

        # 输入进来的数据形状：
        # Q: [batch_size * len_q * d_model]
        # K: [batch_size * len_k * d_model]
        # V: [batch_size * len_k * d_model]
        residual, batch_size = Q, Q.size(0)   # residual: 保存原始输入，用于残差连接

        # 下面就是先映射，后分头
        # q和k分头之后维度是一致的，都是d_k
        # 用view函数分成8个头，每个头都是d_k维度
        # 分头后得到的QK矩阵维度一定相同，否则无法相乘
        q_s = self.W_Q(Q).view(batch_size, -1, self.n_heads, self.d_k).transpose(1, 2)  # q_s: [batch_size, n_heads, len_q, d_k]
        k_s = self.W_K(K).view(batch_size, -1, self.n_heads, self.d_k).transpose(1, 2)  # k_s: [batch_size, n_heads, len_k, d_k]
        v_s = self.W_V(V).view(batch_size, -1, self.n_heads, self.d_v).transpose(1, 2)  # v_s: [batch_size, n_heads, len_k, d_v]

        # 输入的attn_mask形状是 batch_size * len_q * len_k
        # 经过下面代码得到新的attn_mask: [batch_size, n_heads, len_q, len_k]
        # 就是把pad信息（哪些字符是PAD符号）重复在了n个头上，让所有头共享相同的掩码信息（如 padding 位置）
        attn_mask = attn_mask.unsqueeze(1).repeat(1, self.n_heads, 1, 1)

        # 然后我们计算 ScaledDotProductAttention 这个函数。得到的结果有两个：
        # context: [batch_size * n_heads * len_q * d_v]
        # attn: [batch_size * n_heads * len_q * len_k]
        context, attn = self.attention(q_s, k_s, v_s, attn_mask)

        # 合并多头输出
        # transpose(1, 2)是把1维和2维交换：[batch_size, len_q, n_heads, d_v]
        # contiguous() 让内存连续，为view做准备
        context = context.transpose(1, 2).contiguous().view(batch_size, -1, self.n_heads * self.d_v)
        output = self.linear(context)  # output: [batch_size * len_q * d_model]
        return self.layer_norm(output + residual), attn