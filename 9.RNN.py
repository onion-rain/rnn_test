# %matplotlib inline # jupyter notebook用来直接在python console里面生成图像
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim
import numpy as np
from matplotlib import pyplot as plt
import matplotlib.animation
import math, random
print("pytorch_version: {}".format(torch.__version__))
import platform
print("python_version: {}".format(platform.python_version()))

# 定义超参数
TIME_STEP = 10 # RNN时序步长数
INPUT_SIZE = 1 # RNN输入维度
HIDDEN_SIZE = 64 # of RNN隐藏单元个数
EPOCHS = 300 # 总训练次数
h_state = None # 隐藏层状态

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("device: {}".format(DEVICE))

# 用numpy生成数据
steps = np.linspace(0, np.pi*2, num=256, dtype=np.float32) # linspace线性等分向量，返回一个array
x_np = np.sin(steps) # 计算对应的 sin(steps) array
y_np = np.cos(steps)

# 可视化数据
plt.figure(1) # 建立图形1
plt.suptitle("sin and cos", fontsize="18") # title为sin and cos, 字体大小为18
# b    blue   蓝      .     point              -     solid
# g    green  绿      o     circle             :     dotted
# r    red    红      x     x-mark             -.    dashdot 
# c    cyan   青      +     plus               --    dashed   
# m    magenta 品红   *     star             (none)  no line
# y    yellow 黄      s     square
# k    black  黑      d     diamond
# w    white  白      v     triangle (down)
#                     ^     triangle (up)
#                     <     triangle (left)
#                     >     triangle (right)
#                     p     pentagram 五角星
#                     h     hexagram  六角星
plt.plot(steps, y_np, "r-", label="target (cos)") # plot绘图
plt.plot(steps, x_np, "b-", label="target (sin)")
plt.legend(loc="best") # 图例位置
# plt.show()

class RNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.rnn = nn.RNN(
            input_size=INPUT_SIZE, 
            hidden_size=HIDDEN_SIZE, 
            num_layers=1, 
            batch_first=True
        )
        self.out = nn.Linear(HIDDEN_SIZE, 1)
    def forward(self, x, h_state):
        # x (batch, time_step, input_size)
        # h_state (n_layers, batch, hidden_size)
        # r_out (batch, time_step, hidden_size)
        r_out, h_state = self.rnn(x, h_state)
        outs = [] # 保存所有预测值
        for time_step in range(r_out.size(1)): # 计算每一步长的预测值
            outs.append(self.out(r_out[:, time_step, :]))
        return torch.stack(outs, dim=1), h_state
        # 也可使用以下这样的返回值
        # r_out = r_out.view(-1, 32)
        # outs = self.out(r_out)
        # return outs, h_state

rnn = RNN().to(DEVICE)
optimizer = torch.optim.Adam(rnn.parameters()) # Adam优化，几乎不用调参
criterion = nn.MSELoss() # 因为最终的结果是一个数值，所以损失函数用均方误差

rnn.train() # 训练模式
plt.figure(2)
for step in range(EPOCHS):
    start, end = step * np.pi, (step+1)*np.pi # 一个时间周期
    steps = np.linspace(start, end, TIME_STEP, dtype=np.float32)
    x_np = np.sin(steps) 
    y_np = np.cos(steps)
    x = torch.from_numpy(x_np[np.newaxis, :, np.newaxis])  # shape (batch, time_step, input_size)
    y = torch.from_numpy(y_np[np.newaxis, :, np.newaxis])
    x=x.to(DEVICE)
    prediction, h_state = rnn(x, h_state) # rnn output
    # 这一步非常重要
    h_state = h_state.data # 重置隐藏层的状态, 切断和前一次迭代的链接
    loss = criterion(prediction.cpu(), y)
    # 这三行写在一起就可以
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    if (step+1)%20==0: #每训练20个批次可视化一下效果，并打印一下loss
        print("EPOCHS: {},Loss:{:4f}".format(step,loss))
        plt.plot(steps, y_np.flatten(), 'r-')
        plt.plot(steps, prediction.cpu().data.numpy().flatten(), 'b-')
        plt.draw()
        plt.pause(0.01)

print()