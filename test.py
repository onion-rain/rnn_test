import torch
import numpy as np

batch_size = 2
max_length = 3
hidden_size = 2
num_layers = 1
# 这个RNN由两个全连接层组成，对应的两个hidden_state的维度是2，输入向量维度是1
rnn = torch.nn.RNN(1, hidden_size, num_layers, batch_first=True)


x = torch.FloatTensor([[1, 0, 0], [1, 2, 3]]).resize_(2, 3, 1)
# x = Variable(x)  # [batch, seq, feature], [2, 3, 1]
seq_lengths = np.array([1, 3])  # list of integers holding information about the batch size at each sequence step
print(x)


# 对seq_len进行排序
order_idx = np.argsort(seq_lengths)[::-1]
print('order_idx:', str(order_idx))
order_x = x[order_idx.tolist()]
order_seq = seq_lengths[order_idx]
print(order_x)


# 经过以上处理后，长序列的样本调整到短序列样本之前了
# pack it
pack = torch.nn.utils.rnn.pack_padded_sequence(order_x, order_seq, batch_first=True)
print(pack)


# initialize
h0 = torch.randn(num_layers, batch_size, hidden_size)
# forward
out, _ = rnn(pack, h0)
print(out)


# unpack
unpacked = pad_packed_sequence(out)
out, bz = unpacked[0], unpacked[1]
print(out, bz)


# seq_len x batch_size x hidden_size --> batch_size x seq_len x hidden_size
out = out.permute((1, 0, 2))
print("output", out)
print("input", order_x)