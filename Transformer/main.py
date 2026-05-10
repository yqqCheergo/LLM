from transformer import Transformer
import torch.nn as nn
from torch import optim
import torch


# 把单词转换为索引
def make_batch(sentences):
    input_batch = [[src_vocab[n] for n in sentences[0].split()]]     # 源语言句子，德语
    output_batch = [[tgt_vocab[n] for n in sentences[1].split()]]    # 解码器输入，英语+起始符
    target_batch = [[tgt_vocab[n] for n in sentences[2].split()]]    # 目标输出，英语+结束符
    return torch.LongTensor(input_batch), torch.LongTensor(output_batch), torch.LongTensor(target_batch)


if __name__ == '__main__':

    # 句子的输入部分
    sentences = ['ich mochte ein bier P', 'S i want a beer', 'i want a beer E']   # 一组句子，batch_size=1

    # Transformer Parameters 配置文件, Padding Should be Zero
    # 构建词表
    src_vocab = {'P': 0, 'ich': 1, 'mochte': 2, 'ein': 3, 'bier': 4}   # 编码端词表，为了方便将字符对应为数字，以便被计算机更好地识别
    src_vocab_size = len(src_vocab)

    tgt_vocab = {'P': 0, 'i': 1, 'want': 2, 'a': 3, 'beer': 4, 'S': 5, 'E': 6}   # 解码端词表 (解码端和编码端可以共用一个词表，但不同语言要分开构建)
    tgt_vocab_size = len(tgt_vocab)

    src_len = 5   # 原句输入长度
    tgt_len = 5   # 目标句输入长度

    # 模型参数
    d_model = 512    # 每一个字符转换为embedding的维度大小
    d_ff = 2048      # 前馈神经网络中linear层映射到多少维度
    d_k = d_v = 64   # dimension of K(=Q), V
    n_layers = 6     # 6个Encoder堆叠在一起
    n_heads = 8      # 多头注意力机制的头的个数

    model = Transformer(src_vocab_size, tgt_vocab_size, d_model, n_layers, d_k, d_v, n_heads, d_ff)

    criterion = nn.CrossEntropyLoss(ignore_index=0)        # 0是PAD的token id，这样会自动忽略target中所有值为0的位置
    optimizer = optim.Adam(model.parameters(), lr=0.001)   # 若效果不好改成SGD试试

    enc_inputs, dec_inputs, target_batch = make_batch(sentences)

    for epoch in range(20):
        # 1. 清零梯度
        optimizer.zero_grad()
        # 2. 前向传播
        outputs, enc_self_attns, dec_self_attns, dec_enc_attns = model(enc_inputs, dec_inputs)
        # 3. 计算损失
        '''
        outputs: [batch_size * tgt_len, tgt_vocab_size]
        target_batch: [batch_size, tgt_len] -> 展平成1维
        '''
        loss = criterion(outputs, target_batch.contiguous().view(-1))    # 把多句话拼成一个长句子，为ground truth
        print('Epoch:', '%04d' % (epoch + 1), 'cost =', '{:.6f}'.format(loss))
        # 4. 反向传播（计算梯度）
        loss.backward()
        # 5. 更新参数
        optimizer.step()

    # infer
    predict, _, _, _ = model(enc_inputs, dec_inputs)    # [batch_size * tgt_len, tgt_vocab_size]
    # 从模型的预测输出中，找出每个位置概率最高的词对应的索引
    predict = predict.data.max(1, keepdim=True)[1]      # max() 返回一个元组 (values, indices)
    number_dict = {i: w for i, w in enumerate(tgt_vocab)}    # 创建一个从数字索引到单词的映射字典（反向词表）
    '''
    predict: [batch_size * tgt_len, 1]
    number_dict 是反向词表：索引 -> 单词
    '''
    print(sentences[0], '->', [number_dict[n.item()] for n in predict.squeeze()])