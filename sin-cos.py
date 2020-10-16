# import torch
# import torch.nn as nn

# rnn = nn.RNN(10, 20, 2)
# inputs = torch.randn(5, 3, 10)  # (time_step, batch_size, input_size)
# h0 = torch.randn(2, 3, 20)  # (num_layers, batch_size, hidden_size)
# output, hn = rnn(inputs, h0)
# print(output.shape)  # (time_step, batch_size, hidden_size)

# for name, param in rnn.named_parameters():
#     if param.requires_grad:
#         print(name, param.size())



import torch
from torch import nn
import numpy as np
import matplotlib.pyplot as plt

torch.manual_seed(2019)

# 超参设置
TIME_STEP = 20  # RNN时间步长
INPUT_SIZE = 1  # RNN输入尺寸
INIT_LR = 0.02  # 初始学习率
N_EPOCHS = 200  # 训练回数


class RNN(nn.Module):
    def __init__(self):
        super(RNN, self).__init__()
        self.rnn = nn.RNN(
            input_size=INPUT_SIZE,
            hidden_size=32,  # RNN隐藏神经元个数
            num_layers=1,  # RNN隐藏层个数
        )
        self.out = nn.Linear(32, 1)

    def forward(self, x, h):
        # x (time_step, batch_size, input_size)
        # h (n_layers, batch, hidden_size)
        # out (time_step, batch_size, hidden_size)
        out, h = self.rnn(x, h)
        prediction = self.out(out)
        return prediction, h

# class RNN(nn.Module):
#     def __init__(self, input_size=INPUT_SIZE, hidden_size=32, output_size=1):
#         super(RNN, self).__init__()

#         self.num_layers = 1
#         self.hidden_size = hidden_size

#         self.u = nn.Linear(input_size, hidden_size)
#         self.w = nn.Linear(hidden_size, hidden_size)
#         self.v = nn.Linear(hidden_size, output_size)

#         self.tanh = nn.Tanh()
#         self.softmax = nn.LogSoftmax(dim=1)

#     def forward(self, inputs, hidden):

#         if hidden is None:
#             num_directions = 1
#             hidden = torch.zeros(self.num_layers * num_directions,
#                                  1, self.hidden_size,
#                                  dtype=inputs.dtype, device=inputs.device)
#         u_x = self.u(inputs)

#         hidden = self.w(hidden)
#         hidden = self.tanh(hidden + u_x)

#         output = self.v(hidden)

#         return output, hidden

#     def initHidden(self):
#         return torch.zeros(1, self.hidden_size)



rnn = RNN()
print(rnn)

optimizer = torch.optim.Adam(rnn.parameters(), lr=INIT_LR)
criterion = nn.MSELoss()
h_state = None  # 初始化隐藏层

plt.figure()
plt.ion()
for step in range(N_EPOCHS):
    start, end = step * 2*np.pi, (step + 1) * 2*np.pi  # 时间跨度
    # 使用Sin函数预测Cos函数
    steps = np.linspace(start, end, TIME_STEP, dtype=np.float32, endpoint=False)
    x_np = np.sin(steps)
    y_np = np.cos(steps)
    x = torch.from_numpy(x_np[:, np.newaxis, np.newaxis])  # 尺寸大小为(time_step, batch, input_size)
    y = torch.from_numpy(y_np[:, np.newaxis, np.newaxis])

    prediction, h_state = rnn(x, h_state)  # RNN输出（预测结果，隐藏状态）
    h_state = h_state.detach()  # 这一行很重要，将每一次输出的中间状态传递下去(不带梯度)
    loss = criterion(prediction, y)
    print(loss)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    # 绘制中间结果
    plt.cla()
    plt.plot(steps, y_np, 'r-')
    plt.plot(steps, prediction.data.numpy().flatten(), 'b-')
    plt.draw()
    plt.pause(0.1)
plt.ioff()
plt.show()
