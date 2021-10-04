import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class FocalLossMulty(nn.Module):
    def __init__(self, alpha, gamma, size_average=True):
        super(FocalLossMulty, self).__init__()
        self.alpha = torch.tensor(alpha).to("cuda")
        self.gamma = gamma
        self.size_average = size_average

    def forward(self, loginput, label):
        inp = torch.exp(loginput)

        alpha_label = self.alpha * label
        # print(alpha_label)
        inp_label = inp * label
        # print(inp_label)
        loginput_label = loginput * label
        # print(loginput)
        focal_loss = -(1 - inp_label) ** self.gamma * alpha_label * loginput_label
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
    TP, FN, FP, TN = [], [], [], []
    # sample class numbers
    for i in range(classes):
        pred_i = (torch.argmax(pred, dim=1) == i)  # .type(torch.int8)
        # preds.append(pred_i)
        lab_i = (torch.argmax(labels, dim=1) == i)  # .type(torch.int8)
        # labs.append(lab_i)
        # i-dik osztálynak lett sorolva, és valóban i-dik osztályban van
        ##tp_i = (pred_i == lab_i)#.type(torch.int8)
        # vagy
        # a második a jó
        tp_i = torch.sum(pred_i * lab_i)
        fn_i = torch.sum((~ pred_i) * lab_i)
        fp_i = torch.sum(pred_i * (~ lab_i))
        tn_i = torch.sum((~ pred_i) * (~ lab_i))
        TP.append(tp_i)
        FN.append(fn_i)
        FP.append(fp_i)
        TN.append(tn_i)
        # print(tp_i)
    return {'tp': torch.tensor(TP), 'fn':torch.tensor(FN),'fp': torch.tensor(FP),'tn': torch.tensor(TN)}


if __name__ == "__main__":
    x = np.eye(3)
    N = 100
    label = x[np.random.choice(x.shape[0], size=N)]
    # print(label)
    label = torch.tensor(label)
    loginput = torch.rand(N, 3)
    sm = nn.LogSoftmax(dim=1)
    loginput = sm(loginput)
    # print(torch.exp(loginput))
    loss = FocalLossMulty([0.4, 0.1, 0.4], 5)
    # print(loss(loginput,label))
    print(calc_scores(torch.exp(loginput), label))
