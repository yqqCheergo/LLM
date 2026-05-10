import torch.nn as nn
import torch
import math

# Positional Encoding
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super(PositionalEncoding, self).__init__()

        # 偶数和奇数在公式上有一个共同部分，使用log函数把次方拿下来，方便计算
        # pos代表的是单词在句子中的索引，如max_len是128，那么索引就是从0,1,2...127
        # 假设d_model是512，2i符号中i从0取到255，那么2i对应取值就是0,2,4...510
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)   # 升维
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))  # 共有的部分
        pe[:, 0::2] = torch.sin(position * div_term)   # 从0开始，步长为2，其实代表的就是偶数位置
        pe[:, 1::2] = torch.cos(position * div_term)   # 从1开始，步长为2，其实代表的就是奇数位置
        # 上面代码之后得到的pe: [max_len * d_model]

        # 下面这个代码之后得到的pe: [max_len * 1 * d_model]
        pe = pe.unsqueeze(0).transpose(0, 1)   # batch维度占位（方便广播）

        self.register_buffer('pe', pe)  # 定一个缓冲区，简单理解为这个位置编码pe是一个常规参数，不参与更新

    def forward(self, x):
        """
        x: [seq_len, batch_size, d_model]  经过词向量的输入
        """
        # 经过词向量的输入与位置编码相加
        x = x + self.pe[:x.size(0), :]   # 取前seq_len个位置（只取实际需要的长度）
        return self.dropout(x)