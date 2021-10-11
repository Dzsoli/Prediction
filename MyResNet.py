import torch
import torch.nn as  nn
import torch.nn.functional as F


class MyResBlock(nn.Module):
    def __init__(self, in_size: int, hidden_size: int, out_size: int):
        super(MyResBlock, self).__init__()
        self.conv1 = nn.Conv3d(in_size, hidden_size, 6, padding=1)
        self.conv2 = nn.Conv3d(hidden_size, out_size, 6, padding=1)
        self.batchnorm1 = nn.BatchNorm3d(hidden_size)
        self.batchnorm2 = nn.BatchNorm3d(out_size)

    def convblock_1(self, x):
        """
        ReLU before addition
        Original
        :param x:
        :return:
        """
        x = F.relu(self.batchnorm1(self.conv1(x)))
        x = F.relu(self.batchnorm2(self.conv2(x)))
        return x

    def convblock_2(self, x):
        """
        Full pre-activation
        :param x:
        :return:
        """
        x = self.conv1(F.relu(self.batchnorm1(x)))
        x = self.conv2(F.relu(self.batchnorm2(x)))
        print('x: ',x.shape)
        return x

    def forward(self, x):
        y = self.convblock_1(x)
        # y = x+y
        return y

    # TODO: kell az a downsampling, de úgy, hogy blokkonként nem nő a dimenzió


if __name__ == "__main__":
    t = torch.ones((10,3,16,16,16))
    m=MyResBlock(3,3,3)
    print(m(t).shape)
