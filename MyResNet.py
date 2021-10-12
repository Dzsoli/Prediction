import torch
import torch.nn as  nn
import torch.nn.functional as F


class MyResBlock(nn.Module):
    def __init__(self, in_size: int, hidden_size: int, out_size: int, stride=1, downsample=None, mode=2):
        super(MyResBlock, self).__init__()
        self.mode = mode
        if mode not in [1,2]:
            raise ValueError
        self.downsample = downsample
        self.stride = stride
        self.conv1 = nn.Conv3d(in_size, hidden_size, 3, padding=1, stride=stride)
        self.conv2 = nn.Conv3d(hidden_size, out_size, 3, padding=1)
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
        print("x original: ", x.shape)
        if self.mode == 2:
            y = self.convblock_1(x)
        else:
            y = self.convblock_1(x)
        if self.downsample is not None:
            x = self.downsample(x)
            print("x downsample: ", x.shape)

        y = x+y
        return y


class MyResNetEncoder(nn.Module):
    def __init__(self, Block, mode):
        super(MyResNetEncoder, self).__init__()
        self.block = Block
        self.mode = mode
        # todo: hozzáadni a layerekel a lap alapján
        # todo: sajnos ugyan így kell egy transpose ResBlock is
        layers = []
        layers.append(self.make_block(1,3,4))
        layers.append(self.make_block(4,6,8,2))
        layers.append(self.make_block(8,10,12,1))
        layers.append(self.make_block(12,10,8,(2,4,2)))
        layers.append(self.make_block(8,8,8))
        layers.append(self.make_block(8,4,1,2))
        self.layers = nn.Sequential(*layers)

    def make_block(self,in_channel, hidden_channel, out_channel, stride=1):
        ds = None  # downsample

        if stride != 1 or in_channel != out_channel:
            ds = nn.Sequential(
                nn.BatchNorm3d(in_channel),
                nn.Conv3d(in_channel, out_channel, kernel_size=1, stride=stride)
            )
        return self.block(in_channel, hidden_channel, out_channel, stride, downsample=ds, mode=self.mode)

    def forward(self, x):
        return self.layers(x)

if __name__ == "__main__":
    t = torch.ones((10,1,16,128,30))
    stride = (2,4,2)
    downsample = nn.Conv3d(1,12,1,stride)
    m=MyResBlock(1,6,12,stride, downsample=downsample)
    # print(m(t).shape)
    enc = MyResNetEncoder(MyResBlock,mode=2)
    print(enc(t).shape)
