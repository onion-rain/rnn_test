import csv
import torch
from torch import nn
import random
import os
import matplotlib.pyplot as plt
import numpy as np
from misc import accuracy, cal_score

init_data_path = "data/MoopLab/data_b_train.csv"
squeue_data_path = "data\MoopLab\data_m_train.csv"
label_path = "data\MoopLab\y_train.csv"

# set_seed(1)  # 设置随机种子

def load_csv(path):
    with open(path, "r") as f:
        reader = csv.reader(f)
        # print(type(reader))

        result = list(reader)
        # for row in reader:
        #     print(row)
        return result

# class INIT(nn.Module):
#     def __init__(self, input, hidden, output):
#         super(INIT, self).__init__()
#         self.linear1 = nn.Linear(input, hidden)
#         self.linear2 = nn.Linear(hidden, output)

#     def forward(self, inputs):
#         out = self.linear1(inputs)
#         out = self.linear2(out)
#         return out

# class RNN(nn.Module):
#     def __init__(self, input_size, hidden_size, output_size):
#         super(RNN, self).__init__()
#         self.rnn = nn.RNN(
#             input_size=input_size,
#             hidden_size=hidden_size,  # RNN隐藏神经元个数
#             num_layers=1,  # RNN隐藏层个数
#         )
#         self.out = nn.Linear(hidden_size, output_size)
#         self.softmax = nn.LogSoftmax(dim=1)

#     def forward(self, x, h):
#         # x (time_step, batch_size, input_size)
#         # h (n_layers, batch, hidden_size)
#         # out (time_step, batch_size, hidden_size)
#         out, h = self.rnn(x, h)
#         prediction = self.softmax(self.out(out))
#         return prediction, h

class RNN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(RNN, self).__init__()

        self.hidden_size = hidden_size

        self.u = nn.Linear(input_size, hidden_size)
        self.w = nn.Linear(hidden_size, hidden_size)
        self.v = nn.Linear(hidden_size, output_size)

        self.tanh = nn.Tanh()
        self.softmax = nn.LogSoftmax(dim=1)

    def forward(self, inputs, hidden):

        u_x = self.u(inputs)

        hidden = self.w(hidden)
        hidden = self.tanh(hidden + u_x)

        output = self.softmax(self.v(hidden))

        return output, hidden



if __name__ == "__main__":
    init_data = load_csv(init_data_path)[1:]
    squeue_data = load_csv(squeue_data_path)[1:]
    label = load_csv(label_path)[1:]

    # init data
    init_data_tensor = []
    init_data_num = []
    init_data_cat = []
    for id in range(len(init_data)):
        init_data_tensor.append(torch.Tensor(list(map(float, init_data[id][1:]))))
        init_data_num.append(list(map(float, init_data[id][1:6])))
        init_data_cat.append(list(map(int,   init_data[id][6:])))

    # squeue data
    squeue_data_float = []
    for id in range(len(squeue_data)):
        squeue_data_float.append(list(map(float, squeue_data[id])))
    data_num = int(max([squeue_data_float[i][0] for i in range(len(squeue_data_float))])) + 1
    squeue_data = [[] for _ in range(data_num)]
    for line in range(len(squeue_data_float)):
        squeue_data[int(squeue_data_float[line][0])].append(squeue_data_float[line][2:])
    squeue_data_tensor = []
    for l in squeue_data:
        squeue_data_tensor.append(torch.Tensor(l))

    # label
    # label_tensor = label
    for id in range(len(label)):
        label[id] = list(map(int, label[id]))[-1:]
    label_tensor = torch.Tensor(label)

    n_input = 66
    n_hidden = 128
    n_output = 2
    N_EPOCHS = 100
    INIT_LR = 1e-6
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))

    # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # device = torch.device("cpu")

    # init_net = INIT()
    rnn_net = RNN(n_input, n_hidden, n_output)
    # rnn_net.to(device)

    # optimizer = torch.optim.Adam(rnn_net.parameters(), lr=INIT_LR)
    optimizer = torch.optim.SGD(rnn_net.parameters(), lr=INIT_LR)
    # criterion = nn.MSELoss()
    # criterion = nn.NLLLoss()
    criterion = nn.CrossEntropyLoss()

    sample_id = l = list(range(data_num))
    
    all_losses = []
    all_prec1 = []
    mean_loss = []
    mean_prec1 = []

    for step in range(N_EPOCHS):
        random.shuffle(sample_id)
        for man_id in sample_id:
            h_state = torch.cat((init_data_tensor[man_id], torch.zeros(110)), 0).unsqueeze(0)
            input = squeue_data_tensor[id].unsqueeze(1)
            # for time_id in range(squeue_data_tensor[id].size(0)):
            for i in range(input.size()[0]):
                output, h_state = rnn_net(input[i], h_state)
            # output = prediction[-1, :]
            target = label_tensor[man_id][-1:].long()
            loss = criterion(output, target)
            prec1 = accuracy(output.data, target.data, topk=(1,))
            print(loss, end="")
            print(prec1)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            if len(all_losses) < 1000:
                all_losses.append(loss.data)
                all_prec1.append(prec1)
            else:
                mean_loss.append(np.mean(all_losses))
                mean_prec1.append(np.mean(all_prec1))
                all_losses = []
                all_prec1 = []
                fig_loss = plt.figure('train_fig_loss')
                plt.ion()
                plt.plot(mean_loss)
                plt.draw()
                plt.pause(0.1)
                fig_prec = plt.figure('train_fig_prec')
                plt.ion()
                plt.plot(mean_prec1)
                plt.draw()
                plt.pause(0.1)

        torch.save(rnn_net.state_dict(), os.path.join(BASE_DIR, "rnn_state_dict.pkl"))
    
    plt.show()