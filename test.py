import csv
import torch
from torch import nn
import random
import os
from tqdm import tqdm
import matplotlib.pyplot as plt
import numpy as np

from misc import accuracy
from train import dataloader_init, valuate, RNN, LSTM
from misc import cal_score, csvDataset

init_data_path = "data/MoopLab/data_b_test.csv"
squeue_data_path = "data/MoopLab/data_m_test.csv"
n_input = 66
n_hidden = 128
n_output = 2

checkpoint_path = "random0.212.pkl"
# checkpoint_path = "rnn_state_dict.pkl"
# checkpoint_path = "sgd_rnn_state_dict.pkl"
# checkpoint_path = "adam_rnn_state_dict.pkl"

# set_seed(1)  # 设置随机种子

def test(rnn_net, test_dataloader, epoch):
    all_top1 = []
    mean_top1 = []
    y_true = []
    y_pred = []

    pbar = tqdm(
        test_dataloader,
        desc="Test {:3}".format(epoch),
        ncols=130
    )
    for batch_index, (h_state, inputs, target) in enumerate(pbar):
        
        inputs = inputs[0]
        output = rnn_net(inputs, h_state)

        y_pred.append(output.topk(1, 1, True, True)[1].item())
    return y_pred


if __name__ == "__main__":

    test_dataset = csvDataset(init_data_path, squeue_data_path, train=False,train_percent=0)
    test_dataloader = torch.utils.data.DataLoader(
        dataset=test_dataset,
        batch_size=1,
        shuffle=False,
        num_workers=0,
        drop_last=False,
    )

    print("initializing model ... ", end="")
    # rnn_net = RNN(n_input, n_hidden, n_output)
    rnn_net = LSTM(n_input, n_hidden, n_output)
    rnn_net.load_state_dict(torch.load(checkpoint_path))
    print(" done")

    y_pred = test(rnn_net, test_dataloader, 0)
    
    with open('output.csv', 'w', encoding='utf-8', newline="") as f:
        csv_writer = csv.writer(f)
        csv_writer.writerow(["id", "target"])
        for i in range(len(y_pred)):
            csv_writer.writerow([i, y_pred[i]])

    print()