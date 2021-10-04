import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class FocalLossMulty(nn.Module):
    def __init__(self, alpha, gamma, size_average=True):
        super(FocalLossMulty, self).__init__()
        self.alpha = torch.tensor(alpha)#.to("cuda")
        self.gamma = gamma
        self.size_average = size_average

    def forward(self, loginput, label):
        inp = torch.exp(loginput)

        alpha_label = self.alpha*label
        # print(alpha_label)
        inp_label = inp*label
        # print(inp_label)
        loginput_label = loginput*label
        # print(loginput)
        focal_loss = -(1-inp_label)**self.gamma * alpha_label * loginput_label
        focal_loss = torch.sum(focal_loss, dim=1)
        # print(focal_loss.shape)
        if self.size_average:
            return torch.mean(focal_loss)
        else:
            return torch.sum(focal_loss)


def calc_scores(pred, labels):
    # batch, classes
    batch, classes = pred.shape
    preds = []
    labs = []
    # sample class numbers
    for i in range(classes):
        pred_i = (torch.argmax(pred, dim=1) == i).type(torch.int8)
        preds.append(pred_i)
        lab_i = (torch.argmax(labels, dim=1) == i).type(torch.int8)
        labs.append(lab_i)
        tp_i = (pred_i == lab_i).type(torch.int8)






if __name__ == "__main__":
    x = np.eye(3)
    label = x[np.random.choice(x.shape[0], size=4)]
    # print(label)
    label = torch.tensor(label)
    loginput = torch.rand(4,3)
    sm= nn.LogSoftmax(dim=1)
    loginput =sm(loginput)
    # print(torch.exp(loginput))
    loss = FocalLossMulty([0.4,0.1,0.4],5)
    print(loss(loginput,label))