import torch
import torch.nn as  nn
import torch.nn.functional as F


class MyTransResBlock(nn.Module):
    def __init__(self, in_size: int, hidden_size: int, out_size: int, stride=1, downsample=None, mode=2, padd: int=1):
        super(MyTransResBlock, self).__init__()
        self.mode = mode
        if mode not in [1,2]:
            raise ValueError
        self.downsample = downsample
        self.stride = stride
        self.conv1 = nn.ConvTranspose3d(in_size, hidden_size, 3, padding=padd, stride=stride)
        self.conv2 = nn.ConvTranspose3d(hidden_size, out_size, 3, padding=padd)
        self.batchnorm1 = nn.BatchNorm3d(hidden_size if mode == 1 else in_size)
        self.batchnorm2 = nn.BatchNorm3d(out_size if mode == 1 else hidden_size)

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
        # print('x: ',x.shape)
        return x

    def forward(self, x):
        print("x original: ", x.shape)
        if self.mode == 2:
            y = self.convblock_2(x)
        else:
            y = self.convblock_1(x)
        if self.downsample is not None:
            x = self.downsample(x)
            print("x downsample: ", x.shape)

        y = x+y
        return y


class MyResBlock(nn.Module):
    def __init__(self, in_size: int, hidden_size: int, out_size: int, stride=1, downsample=None, mode=2, padd: int=1,
                 type="encoder", out_padd=None):
        super(MyResBlock, self).__init__()
        if mode not in [1,2]:
            raise ValueError
        self.mode = mode
        if type not in ["encoder", "decoder"]:
            raise ValueError
        self.type_ = type
        if type == "encoder":
            self.conv = nn.Conv3d
        else:
            self.conv = nn.ConvTranspose3d
        self.downsample = downsample
        self.stride = stride
        if out_padd is None:
            self.conv1 = self.conv(in_size, hidden_size, 3, padding=padd, stride=stride)
        else:
            self.conv1 = self.conv(in_size, hidden_size, 3, padding=padd, stride=stride, output_padding=out_padd)
        self.conv2 = self.conv(hidden_size, out_size, 3, padding=padd)
        self.batchnorm1 = nn.BatchNorm3d(hidden_size if mode == 1 else in_size)
        self.batchnorm2 = nn.BatchNorm3d(out_size if mode == 1 else hidden_size)

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
        # print('x: ',x.shape)
        return x

    def forward(self, x):
        print("x original: ", x.shape)
        if self.mode == 2:
            y = self.convblock_2(x)
        else:
            y = self.convblock_1(x)
        if self.downsample is not None:
            x = self.downsample(x)
            print("x downsample: ", x.shape)

        y = x+y
        return y


class MyResNet(nn.Module):
    def __init__(self, Block, mode, type):
        super(MyResNet, self).__init__()
        self.block = Block
        self.mode = mode
        # todo: hozzáadni a layerekel a lap alapján
        # todo: sajnos ugyan így kell egy transpose ResBlock is
        self.layers = None
        self.type_ = type
        if type == "encoder":
            self.conv = nn.Conv3d
            self.make_encoder()
        else:
            self.conv = nn.ConvTranspose3d
            self.make_decoder()

    def make_encoder(self):
        layers = []
        layers.append(self.make_block(1, 3, 4))
        layers.append(self.make_block(4, 6, 8, 2))
        layers.append(self.make_block(8, 10, 12, 1))
        layers.append(self.make_block(12, 10, 8, (2, 4, 2)))
        layers.append(self.make_block(8, 8, 8))
        layers.append(self.make_block(8, 4, 1, 2))
        self.layers = nn.Sequential(*layers)

    def make_decoder(self):
        layers = []
        layers.append(self.make_block(1, 3, 4, 1))
        layers.append(self.make_block(4, 6, 8, 2, 1, 1))
        layers.append(self.make_block(8, 10, 12, 1))
        layers.append(self.make_block(12, 10, 8, (2, 4, 2),1,(1,3,0)))
        layers.append(self.make_block(8, 8, 8))
        layers.append(self.make_block(8, 4, 1, 2, 1, 1))
        self.layers = nn.Sequential(*layers)

    def make_block(self,in_channel, hidden_channel, out_channel, stride=1, padd=1, out_padd=None):
        ds = None  # downsample

        if stride != 1 or in_channel != out_channel:
            if padd == 1:
                padd_ds = 0
            else:
                padd_ds = padd
            if out_padd is None:
                conv_ds = self.conv(in_channel, out_channel, kernel_size=1, stride=stride)
            else:
                conv_ds = self.conv(in_channel, out_channel, kernel_size=1, stride=stride, output_padding=out_padd)
            ds = nn.Sequential(
                nn.BatchNorm3d(in_channel),
                conv_ds
            )
        return self.block(in_channel, hidden_channel, out_channel, stride, downsample=ds, mode=self.mode, padd=padd,
                          type=self.type_, out_padd=out_padd)

    def forward(self, x):
        return self.layers(x)


if __name__ == "__main__":
    t = torch.ones((10,1,16,128,30))
    stride = (2,4,2)
    downsample = nn.Conv3d(1,12,1,stride)
    m=MyResBlock(1,6,12,stride, downsample=downsample)
    # print(m(t).shape)
    enc = MyResNet(MyResBlock, mode=2, type="encoder")
    print(enc(t).shape)
    print("####")
    z = torch.ones((10,1,2,8,4))
    # downsample = nn.Conv3d(1,12,1,stride)
    downsample = nn.ConvTranspose3d(1,4,1,2)
    # md = MyTransResBlock(1,3,4,2,downsample)
    dec = MyResNet(MyResBlock, mode=2, type="decoder")
    print(dec(z).shape)
