import torch.nn as nn
import torch

# pos-wise Feed Forward Network 特征提取

# Conv1d 写法
class PoswiseFeedForwardNet(nn.Module):
    def __init__(self, d_model, d_ff):
        super(PoswiseFeedForwardNet, self).__init__()
        # 先升维再降维至原维度
        self.conv1 = nn.Conv1d(in_channels=d_model, out_channels=d_ff, kernel_size=1)   # 512—>2048
        self.conv2 = nn.Conv1d(in_channels=d_ff, out_channels=d_model, kernel_size=1)   # 2048—>512
        self.layer_norm = nn.LayerNorm(d_model)

    def forward(self, inputs):
        residual = inputs     # inputs: [batch_size, len_q, d_model]
        # Conv1d 要求输入是 [batch_size, channels, width]
        # Conv1d 把 d_model 看作通道数，把 seq_len 看作宽度，因此需要转置
        output = nn.ReLU()(self.conv1(inputs.transpose(1, 2)))
        output = self.conv2(output).transpose(1, 2)
        return self.layer_norm(output + residual)


# # Linear 写法
# class PoswiseFeedForwardNet(nn.Module):
#     def __init__(self, d_model, d_ff):
#         super(PoswiseFeedForwardNet, self).__init__()
#         self.linear1 = nn.Linear(d_model, d_ff)
#         self.linear2 = nn.Linear(d_ff, d_model)
#         self.layer_norm = nn.LayerNorm(d_model)
#
#     def forward(self, inputs):
#         residual = inputs     # inputs: [batch_size, len_q, d_model]
#         output = self.linear2(torch.relu(self.linear1(inputs)))
#         return self.layer_norm(output + residual)