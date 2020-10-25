import torch
from torch import nn
import random
import os
import matplotlib.pyplot as plt
import numpy as np
from misc import accuracy, csvDataset
from tqdm import tqdm
from misc import cal_score

init_data_path = "data/MoopLab/data_b_train.csv"
squeue_data_path = "data\MoopLab\data_m_train.csv"
label_path = "data\MoopLab\y_train.csv"
n_input = 66
n_hidden = 128
n_output = 2
N_EPOCHS = 10
INIT_LR = 1e-4
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# set_seed(1)  # 设置随机种子

class LSTM(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(LSTM, self).__init__()

        self.rnn = nn.LSTM(         # if use nn.RNN(), it hardly learns
            input_size=input_size,
            hidden_size=hidden_size,         # rnn hidden unit
            num_layers=1,           # number of rnn layer
            batch_first=False,      # input & output will has batch size as 1s dimension. e.g. (batch, time_step, input_size)
        )
        self.out = nn.Linear(hidden_size, output_size)

    def forward(self, x, h):
        # x shape (time_step, batch, input_size)
        # r_out shape (time_step, batch, output_size)
        # h_n shape (n_layers, batch, hidden_size)
        # h_c shape (n_layers, batch, hidden_size)
        r_out, (h_n, h_c) = self.rnn(x, None)   # None represents zero initial hidden state

        # choose r_out at the last time step
        prediction = self.out(r_out[-1, :, :])
        return prediction

class RNN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(RNN, self).__init__()
        self.rnn = nn.RNN(
            input_size=input_size,
            hidden_size=hidden_size,  # RNN隐藏神经元个数
            num_layers=1,  # RNN隐藏层个数
        )
        self.out = nn.Linear(hidden_size, output_size)

    def forward(self, x, h):
        # x (time_step, batch_size, input_size)
        # h (n_layers, batch, hidden_size)
        # out (time_step, batch_size, hidden_size)
        out, h = self.rnn(x, h)
        prediction = self.out(out[-1, :, :])
        return prediction


def dataloader_init(init_data_path, squeue_data_path, label_path, train_percent=0.8):
    print("initializing dataset ... ", end="")
    if train_percent > 0:
        train_dataset = csvDataset(init_data_path, squeue_data_path, label_path, train=True, train_percent=train_percent)
        train_dataloader = torch.utils.data.DataLoader(
            dataset=train_dataset,
            batch_size=1,
            shuffle=True,
            num_workers=0,
            pin_memory=True,
            drop_last=False,
        )
    else:
        train_dataloader = None
    
    if train_percent < 1:
        val_dataset = csvDataset(init_data_path, squeue_data_path, label_path, train=False, train_percent=train_percent)
        val_dataloader = torch.utils.data.DataLoader(
            dataset=val_dataset,
            batch_size=1,
            shuffle=False,
            num_workers=0,
            drop_last=False,
        )
    else:
        val_dataloader = None

    print(" done")
    return train_dataloader, val_dataloader


def train(rnn_net, train_dataloader, criterion, optimizer, epoch, statistic_dic):

    # torch.save(rnn_net.state_dict(), os.path.join(BASE_DIR, "{}_random.pkl".format(epoch)))
    pbar = tqdm(
        train_dataloader,
        desc="Train   {:3}".format(epoch),
        ncols=130
    )
    for batch_index, (h_state, inputs, target) in enumerate(pbar):

        inputs = inputs[0]
        target = target[0]
        output = rnn_net(inputs, h_state)

        loss = criterion(output, target)
        top1 = accuracy(output.data, target.data, topk=(1,))
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        if len(statistic_dic["all_losses"]) < 1000:
            statistic_dic["all_losses"].append(loss.data)
            statistic_dic["all_top1"].append(top1)
        else:
            statistic_dic["mean_loss"].append(np.mean(statistic_dic["all_losses"]))
            statistic_dic["all_losses"] = []
            fig_loss = plt.figure('train_fig_loss')
            plt.ion()
            plt.plot(statistic_dic["mean_loss"])
            plt.draw()
            plt.pause(0.1)

            statistic_dic["mean_top1"].append(np.mean(statistic_dic["all_top1"]))
            statistic_dic["all_top1"] = []
            fig_top1 = plt.figure('train_fig_top1')
            plt.ion()
            plt.plot(statistic_dic["mean_top1"])
            plt.draw()
            plt.pause(0.1)

            pbar.set_postfix_str(
                ", loss: {:.3f}".format(statistic_dic["mean_loss"][-1])+
                ", top1: {:0.1f}%".format(statistic_dic["mean_top1"][-1])+
                ", lr: {:0.1e}".format(optimizer.param_groups[0]['lr'])
            )

    torch.save(rnn_net.state_dict(), os.path.join(BASE_DIR, "{}_rnn_state_dict.pkl".format(epoch)))


def valuate(rnn_net, val_dataloader, epoch):
    all_top1 = []
    mean_top1 = []
    y_true = []
    y_pred = []

    pbar = tqdm(
        val_dataloader,
        desc="Valuate {:3}".format(epoch),
        ncols=130
    )
    for batch_index, (h_state, inputs, target) in enumerate(pbar):
        
        inputs = inputs[0]
        target = target[0]
        output = rnn_net(inputs, h_state)

        top1 = accuracy(output.data, target.data, topk=(1,))
        all_top1.append(top1)
        mean_top1.append(np.mean(all_top1))

        pbar.set_postfix_str(
            ", top1: {:0.1f}%".format(mean_top1[-1])
        )

        y_true.append(target.item())
        y_pred.append(output.topk(1, 1, True, True)[1].item())
    return y_true, y_pred


if __name__ == "__main__":

    # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # device = torch.device("cpu")
    train_dataloader, val_dataloader = dataloader_init(init_data_path, squeue_data_path, label_path)

    print("initializing model ... ", end="")
    # init_net = INIT()
    # rnn_net = RNN(n_input, n_hidden, n_output)
    rnn_net = LSTM(n_input, n_hidden, n_output)
    # rnn_net.to(device)

    # criterion = nn.MSELoss()
    # criterion = nn.NLLLoss()
    criterion = nn.CrossEntropyLoss()

    optimizer = torch.optim.Adam(rnn_net.parameters(), lr=INIT_LR)
    # optimizer = torch.optim.SGD(rnn_net.parameters(), lr=INIT_LR)
    print(" done")

    statistic_dic = {"all_losses": [], "mean_loss": [], "all_top1": [], "mean_top1": []}
    y_true, y_pred = valuate(rnn_net, val_dataloader, -1)
    score = cal_score(y_true, y_pred)
    torch.save(rnn_net.state_dict(), os.path.join(BASE_DIR, "randomx.pkl"))
    for epoch in range(N_EPOCHS):
        train(rnn_net, train_dataloader, criterion, optimizer, epoch, statistic_dic)
        y_true, y_pred = valuate(rnn_net, val_dataloader, epoch)
        score = cal_score(y_true, y_pred)
        print()

    plt.show()
