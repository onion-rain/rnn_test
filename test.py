import csv
import torch
from torch import nn
import random
import os
import matplotlib.pyplot as plt
import numpy as np
from misc import accuracy

from train import dataloader_init, valuate, RNN

init_data_path = "data/MoopLab/data_b_train.csv"
squeue_data_path = "data\MoopLab\data_m_train.csv"
label_path = "data\MoopLab\y_train.csv"
n_input = 66
n_hidden = 128
n_output = 2

# checkpoint_path = "sgd_rnn_state_dict.pkl"
checkpoint_path = "adam_rnn_state_dict.pkl"

# set_seed(1)  # 设置随机种子

def load_csv(path):
    with open(path, "r") as f:
        reader = csv.reader(f)
        # print(type(reader))

        result = list(reader)
        # for row in reader:
        #     print(row)
        return result

if __name__ == "__main__":

    train_dataloader, val_dataloader = dataloader_init(init_data_path, squeue_data_path, label_path)

    print("initializing model ... ", end="")
    # init_net = INIT()
    rnn_net = RNN(n_input, n_hidden, n_output)
    rnn_net.load_state_dict(torch.load(checkpoint_path))
    # rnn_net.to(device)
    print(" done")
    

    valuate(rnn_net, val_dataloader, 0)
        
    plt.show()