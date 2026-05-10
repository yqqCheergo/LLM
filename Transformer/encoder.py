import torch.nn as nn
from positional_encoding import PositionalEncoding
from mha import MultiHeadAttention
from ffn import PoswiseFeedForwardNet
from padding_mask import get_attn_pad_mask

# Encoder包含三个部分：词向量embedding，位置编码，注意力层及后续的前馈神经网络
class Encoder(nn.Module):
    def __init__(self, src_vocab_size, d_model, n_layers, d_k, d_v, n_heads, d_ff):   # src_vocab_size 为词表大小
        super(Encoder, self).__init__()
        self.src_emb = nn.Embedding(src_vocab_size, d_model)  # 定义一个矩阵，大小是 src_vocab_size * d_model
        self.pos_emb = PositionalEncoding(d_model)    # 位置编码，固定的正余弦函数。也可以使用类似词向量的 nn.Embedding 获得一个可以更新学习的位置编码
        self.layers = nn.ModuleList([EncoderLayer(d_model, d_k, d_v, n_heads, d_ff) for _ in range(n_layers)])    # 使用 ModuleList 对 6 个 Encoder 进行堆叠。因为后续 Encoder 并没有使用词向量和位置编码，所以抽离出来

    def forward(self, enc_inputs):    # [batch_size * src_len]  src_len 即编码端输入句子的长度

        # 通过 src_emb 进行索引定位，把对应数字的词向量提取出来，形成矩阵
        # enc_outputs 输出形状是 [batch_size, src_len, d_model]
        enc_outputs = self.src_emb(enc_inputs)   # 字符—>数字索引—>Embedding向量

        # 位置编码，接收 enc_output (词向量那一层后的输出) 为位置编码的输入，并将位置编码和词向量相加（把两者相加放入到这个函数里面）
        # pos_emb 输出形状是 [src_len, batch_size, d_model]
        enc_outputs = self.pos_emb(enc_outputs.transpose(0, 1)).transpose(0, 1)

        # get_attn_pad_mask 是为了得到句子中 pad 的位置信息给到模型。后面在计算自注意力和交互注意力的时候去掉 pad 符号的影响
        enc_self_attn_mask = get_attn_pad_mask(enc_inputs, enc_inputs)   # enc_self_attn_mask: [batch_size, src_len, src_len]

        # 自注意力层 + 前馈神经网络
        enc_self_attns = []   # 保存返回的attention值。不参与计算，画热力图用
        for layer in self.layers:  # 堆叠，每一层的输出作为下一层的输入
            # 去看 EncoderLayer 类
            enc_outputs, enc_self_attn = layer(enc_outputs, enc_self_attn_mask)  # 循环6次。接收上一层编码器的输出和传给每一层的attention_mask（记录哪些是PAD符号的信息）
            enc_self_attns.append(enc_self_attn)
        return enc_outputs, enc_self_attns


# EncoderLayer：包含两个部分，多头注意力机制和前馈神经网络
class EncoderLayer(nn.Module):
    def __init__(self, d_model, d_k, d_v, n_heads, d_ff):
        super(EncoderLayer, self).__init__()
        self.enc_self_attn = MultiHeadAttention(d_model, d_k, d_v, n_heads)
        self.pos_ffn = PoswiseFeedForwardNet(d_model, d_ff)   # 特征提取Feed Forward

    def forward(self, enc_inputs, enc_self_attn_mask):
        # 自注意力层，输入是 enc_inputs，形状是 [batch_size * len_q * d_model]
        # 需要注意的是最初始的QKV矩阵是等同于这个输入的
        enc_outputs, attn = self.enc_self_attn(enc_inputs, enc_inputs, enc_inputs, enc_self_attn_mask)  # enc_inputs to same Q,K,V
        enc_outputs = self.pos_ffn(enc_outputs)   # enc_outputs: [batch_size, len_q, d_model]
        return enc_outputs, attn