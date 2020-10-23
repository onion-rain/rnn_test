# from sklearn.metrics import precision_score, recall_score
from torch.utils.data import Dataset
import csv
import torch

# y_true = [1,0,1]
# y_pred=[0,1,1]

# def cal_score(y_true, y_pred):
#     precision = precision_score(y_true, y_pred)
#     recall = recall_score(y_true, y_pred)
#     score = (precision * recall) / (0.4 * precision + 0.6 * recall)

#     print(score)
#     return score


def accuracy(output, target, topk=(1,)):
    """
    Computes the precision@k for the specified values of k
    """
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True) # 取指定维度(第二维度)上的最大值(或最大几个) pred.shape[batch_size, maxk]
    pred = pred.t() # 转置 pred.shape[maxk, batch_size]
    correct = pred.eq(target.view(1, -1).expand_as(pred)) # correct.shape[maxk, batch_size], correct.dtype=bool

    res = []
    for k in topk:
        correct_k = correct[:k].view(-1).float().sum(0)
        res.append(correct_k.mul_(100.0 / batch_size))
    return res


class csvDataset(Dataset):
    def __init__(self, init_path, squeue_path, label_path, train=True, train_percent=0.8):

        init_data = self.load_csv(init_path)[1:]

        squeue_data = self.load_csv(squeue_path)[1:]
        squeue_data_float = []
        for id in range(len(squeue_data)):
            squeue_data_float.append(list(map(float, squeue_data[id])))
        data_num = int(max([squeue_data_float[i][0] for i in range(len(squeue_data_float))])) + 1
        squeue_data = [[] for _ in range(data_num)]
        for line in range(len(squeue_data_float)):
            squeue_data[int(squeue_data_float[line][0])].append(squeue_data_float[line][2:])

        label = self.load_csv(label_path)[1:]

        assert len(init_data) == len(squeue_data) == len(label)
        total_length = len(init_data)
        if train:
            self.length = int(total_length*train_percent)
            init_data = init_data[:self.length]
            squeue_data = squeue_data[:self.length]
            label = label[:self.length]
        else:
            self.length = int(total_length*(1-train_percent))
            init_data = init_data[self.length:]
            squeue_data = squeue_data[self.length:]
            label = label[self.length:]
        
        # init data
        init_data_tensor = []
        # init_data_num = []
        # init_data_cat = []
        for id in range(len(init_data)):
            init_data_tensor.append(torch.Tensor(list(map(float, init_data[id][1:]))))
            # init_data_num.append(list(map(float, init_data[id][1:6])))
            # init_data_cat.append(list(map(int,   init_data[id][6:])))
        self.init_data_tensor = init_data_tensor

        # squeue data
        squeue_data_tensor = []
        for l in squeue_data:
            squeue_data_tensor.append(torch.Tensor(l))
        self.squeue_data_tensor = squeue_data_tensor

        # label
        # label_tensor = label
        for id in range(len(label)):
            label[id] = list(map(int, label[id]))[-1:]
        label_tensor = torch.Tensor(label)
        self.label_tensor = label_tensor

    def __getitem__(self, index):
        inital_h_state = torch.cat((self.init_data_tensor[index], torch.zeros(110)), 0).unsqueeze(0)
        inputs = self.squeue_data_tensor[index].unsqueeze(1)
        target = self.label_tensor[index][-1:].long()

        return inital_h_state, inputs, target

    def __len__(self):
        return self.length

    def load_csv(self, path):
        with open(path, "r") as f:
            reader = csv.reader(f)
            # print(type(reader))

            result = list(reader)
            # for row in reader:
            #     print(row)
            return result