import numpy as np
import torch

'''
get_attn_pad_mask: mask掉句子中的填充符 [PAD]，让注意力机制忽略这些无意义的填充位置
get_attn_subsequent_mask: 在Decoder中防止当前位置看到未来的信息，保证自回归生成（只能看到当前及之前的词）
'''

# get_attn_pad_mask  在后续注意力层中，PAD需要置为-∞，让它不能对其他字符产生影响

# 比如现在的句子长度是5，在后面注意力机制的部分，在计算出QK转置除以根号之后、softmax之前
# 得到的形状 len_input * len_input 代表每个单词对其余（包括自己本身）单词的影响力
# 所以这里需要有一个同等大小形状的矩阵，告诉哪个位置是PAD部分，在计算softmax之前会把这里置为无穷小

# 输出的矩阵形状是 batch_size * len_q * len_k
# 是对k中的pad符号进行标识，并没有对q中的做标识，因为没必要
# seq_q 和 seq_k 不一定一致。在交互注意力中，q来自解码端，k来自编码端
# 所以告诉模型编码这边pad符号信息就可以，解码端的pad信息在交互注意力层是没有用到的

def get_attn_pad_mask(seq_q, seq_k):
    # 得到 batch_size、输入长度、输出长度
    batch_size, len_q = seq_q.size()
    batch_size, len_k = seq_k.size()
    # eq(zero) is PAD token
    # 看输入进来的 seq_k 里哪些位置是PAD符号，返回True/False，重复 len_q 次
    pad_attn_mask = seq_k.data.eq(0).unsqueeze(1)  # 扩展一个维度，[batch_size, 1, len_k]，因为word embedding是三维的
    return pad_attn_mask.expand(batch_size, len_q, len_k)


def get_attn_subsequent_mask(seq):   # seq: [batch_size, tgt_len], tgt_len为目标序列的长度（比如要生成的目标句子长度）
    attn_shape = [seq.size(0), seq.size(1), seq.size(1)]    # attn_shape: [batch_size, tgt_len, tgt_len]
    # 生成一个上三角矩阵
    subsequence_mask = np.triu(np.ones(attn_shape), k=1)  # 1表示需要mask的位置，0表示可以关注的位置
    subsequence_mask = torch.from_numpy(subsequence_mask).byte()  # byte类型非0即1
    return subsequence_mask   # [batch_size, tgt_len, tgt_len]