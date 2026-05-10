import torch.nn as nn
import torch
from positional_encoding import PositionalEncoding
from mha import MultiHeadAttention
from ffn import PoswiseFeedForwardNet
from padding_mask import get_attn_pad_mask, get_attn_subsequent_mask

# Decoder 包含词向量层、位置编码层、解码层（堆叠N个）
class Decoder(nn.Module):
    def __init__(self, tgt_vocab_size, d_model, n_layers, d_k, d_v, n_heads, d_ff):
        super(Decoder, self).__init__()
        self.tgt_emb = nn.Embedding(tgt_vocab_size, d_model)
        self.pos_emb = PositionalEncoding(d_model)
        self.layers = nn.ModuleList([DecoderLayer(d_model, d_k, d_v, n_heads, d_ff) for _ in range(n_layers)])   # 解码层堆叠N个

    def forward(self, dec_inputs, enc_inputs, enc_outputs):    # dec_inputs: [batch_size * tgt_len]
        dec_outputs = self.tgt_emb(dec_inputs)    # dec_outputs: [batch_size, tgt_len, d_model]
        dec_outputs = self.pos_emb(dec_outputs.transpose(0, 1)).transpose(0, 1)    # dec_outputs: [batch_size, tgt_len, d_model]

        # get_attn_pad_mask 自注意力层的pad部分
        dec_self_attn_pad_mask = get_attn_pad_mask(dec_inputs, dec_inputs)   # 生成符号矩阵，输入是dec_inputs，输出是其中哪些位置是PAD

        # get_attn_subsequent_mask 自注意层的mask部分。当前单词之后看不到的部分做mask，生成一个上三角为1的矩阵，1表示需要mask
        dec_self_attn_subsequent_mask = get_attn_subsequent_mask(dec_inputs)

        # 两个矩阵相加，大于0的为1，不大于0的为0（torch.gt()的用法），为1的就是被mask的部分，在之后就会被 masked_fill_ 到无限小
        dec_self_attn_mask = torch.gt((dec_self_attn_pad_mask + dec_self_attn_subsequent_mask), 0)

        # 交互注意力机制中的mask矩阵。enc_inputs在交互时告诉解码端哪些是PAD符号
        # enc的输入作为K，看K里哪些位置是pad符号，这些位置没有实际语义信息，所有Q都不应该关注这些无效位置
        # dec的输入作为Q。Q里肯定也有pad符号，但是这里不在意，其对应的输出是无效的，且不会影响其他位置的注意力计算
        dec_enc_attn_mask = get_attn_pad_mask(dec_inputs, enc_inputs)   # 去看enc_inputs里哪些是PAD符号，把它置为1

        dec_self_attns, dec_enc_attns = [], []
        for layer in self.layers:
            dec_outputs, dec_self_attn, dec_enc_attn = layer(dec_outputs, enc_outputs, dec_self_attn_mask, dec_enc_attn_mask)
            dec_self_attns.append(dec_self_attn)
            dec_enc_attns.append(dec_enc_attn)
        return dec_outputs, dec_self_attns, dec_enc_attns


# DecoderLayer 包含多头自注意力层、多头交互注意力层和FFN层
class DecoderLayer(nn.Module):
    def __init__(self, d_model, d_k, d_v, n_heads, d_ff):
        super(DecoderLayer, self).__init__()
        self.dec_self_attn = MultiHeadAttention(d_model, d_k, d_v, n_heads)   # 自注意力层
        self.dec_enc_attn = MultiHeadAttention(d_model, d_k, d_v, n_heads)    # 交互注意力层
        self.pos_ffn = PoswiseFeedForwardNet(d_model, d_ff)      # 前馈神经网络

    def forward(self, dec_inputs, enc_outputs, dec_self_attn_mask, dec_enc_attn_mask):
        dec_outputs, dec_self_attn = self.dec_self_attn(dec_inputs, dec_inputs, dec_inputs, dec_self_attn_mask)
        dec_outputs, dec_enc_attn = self.dec_enc_attn(dec_outputs, enc_outputs, enc_outputs, dec_enc_attn_mask)  # 前三个参数分别是QKV
        dec_outputs = self.pos_ffn(dec_outputs)
        return dec_outputs, dec_self_attn, dec_enc_attn