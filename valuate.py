import csv
import torch
from torch import nn
import random
import os
import matplotlib.pyplot as plt
import numpy as np
from misc import accuracy

from train import dataloader_init, valuate, RNN, LSTM
from misc import cal_score

init_data_path = "data/MoopLab/data_b_train.csv"
squeue_data_path = "data\MoopLab\data_m_train.csv"
label_path = "data\MoopLab\y_train.csv"
n_input = 66
n_hidden = 128
n_output = 2

checkpoint_path = "random0.212.pkl"
# checkpoint_path = "rnn_state_dict.pkl"
# checkpoint_path = "sgd_rnn_state_dict.pkl"
# checkpoint_path = "adam_rnn_state_dict.pkl"

# set_seed(1)  # 设置随机种子

if __name__ == "__main__":

    train_dataloader, val_dataloader = dataloader_init(init_data_path, squeue_data_path, label_path, train_percent=0)

    print("initializing model ... ", end="")
    # init_net = INIT()
    # rnn_net = RNN(n_input, n_hidden, n_output)
    rnn_net = LSTM(n_input, n_hidden, n_output)
    rnn_net.load_state_dict(torch.load(checkpoint_path))
    # rnn_net.to(device)
    print(" done")

    y_true, y_pred = valuate(rnn_net, val_dataloader, 0)
    # y_true = [1,0,1]
    # y_pred = [0,1,1]
    score = cal_score(y_true, y_pred)

    print()