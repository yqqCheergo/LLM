import torch.nn as nn
from encoder import Encoder
from decoder import Decoder

# 从整体网络结构来看，分为三个部分：编码层，解码层，映射层(最后和真实标签计算损失)
class Transformer(nn.Module):
    def __init__(self, src_vocab_size, tgt_vocab_size, d_model, n_layers, d_k, d_v, n_heads, d_ff):
        super(Transformer, self).__init__()
        self.encoder = Encoder(src_vocab_size, d_model, n_layers, d_k, d_v, n_heads, d_ff)  # 编码层，输出维度512维
        self.decoder = Decoder(tgt_vocab_size, d_model, n_layers, d_k, d_v, n_heads, d_ff)  # 解码层，输出维度512维
        self.projection = nn.Linear(d_model, tgt_vocab_size, bias=False)  # 映射层d_model是解码层每个token输出的维度大小，之后会做一个tgt_vocab_size大小的softmax

    def forward(self, enc_inputs, dec_inputs):
        # 这里有两个数据进行输入，一个是enc_inputs，形状为[batch_size, src_len]，作为编码端的输入
        # 一个是dec_inputs，形状为[batch_size, tgt_len]，作为解码端的输入

        # enc_inputs作为输入，输出由自己函数内部指定，想要什么指定输出什么，可以是全部tokens的输出，可以是特定每一层的输出，也可以是中间某些参数的输出
        # enc_outputs就是主要的输出，维度是[batch_size, src_len, d_model]
        # enc_self_attns是QK转置相乘+softmax之后的矩阵值，代表的是每个单词和其他单词的相关性(for可视化)
        enc_outputs, enc_self_attns = self.encoder(enc_inputs)

        # dec_outputs是decoder的主要输出，用于后续linear映射
        # dec_self_attns类比于enc_self_attns，是查看每个单词对decoder中输入的其余单词的相关性(for可视化)
        # dec_enc_attns是decoder中每个单词对encoder中每个单词的相关性(for可视化)
        dec_outputs, dec_self_attns, dec_enc_attns = self.decoder(dec_inputs, enc_inputs, enc_outputs)  # enc_inputs告诉模型Encoder输入中哪些部分被PAD符号填充

        # dec_outputs维度是[batch_size, tgt_len, d_model]
        # 再做映射到词表大小
        dec_logits = self.projection(dec_outputs)   # [batch_size, tgt_len, tgt_vocab_size]
        return dec_logits.view(-1, dec_logits.size(-1)), enc_self_attns, dec_self_attns, dec_enc_attns