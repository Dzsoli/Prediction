import torch
import torch.nn as  nn
import torch.nn.functional as F


class QuadResBlock(nn.Module):
    def __init__(self, in_size: int, out_size: int, stride=1, downsample=None, padd: int=1):
        super(QuadResBlock, self).__init__()
        self.conv = nn.Conv3d
        self.downsample = downsample
        self.stride = stride
        self.conv1 = self.conv(in_size, out_size, 3, padding=padd, stride=stride, bias=False) # itt is False lett
        self.conv2 = self.conv(out_size, out_size, 3, padding=padd) # a kvadratikus tagnál legyen
        self.batchnorm1 = nn.BatchNorm3d(in_size)
        self.batchnorm2 = nn.BatchNorm3d(out_size)

    def convblock(self, x):
        Dx = self.conv1(F.relu(self.batchnorm1(x)))
        DDx = self.conv2(F.relu(self.batchnorm2(Dx)))
        # print(Dx.shape)
        if self.downsample is not None:
            x = self.downsample(x)
        xDDx = x * DDx
        # print(xDDx.shape)
        return x + Dx + 0.5 * xDDx

    def forward(self, x):
        return self.convblock(x)
        # if self.downsample is not None:
        #     x = self.downsample(x)
        # return x + res


class QuadResNet(nn.Module):
    def __init__(self):
        super(QuadResNet, self).__init__()
        self.layers = None
        self.make_layers_vol2()

    def make_block(self, in_channel, out_channel, stride=1, padd=1):
        ds = None

        if stride != 1 or in_channel != out_channel:
            conv_ds = nn.Conv3d(in_channel, out_channel, kernel_size=1, stride=stride, bias=False) # Ide tettem False-t
            ds = nn.Sequential(
                nn.BatchNorm3d(in_channel),
                conv_ds
            )
        return QuadResBlock(in_channel, out_channel, stride, downsample=ds, padd=padd)

    def make_layers(self):
        layers = []
        layers.append(self.make_block(1,1,1))
        layers.append(self.make_block(1,4,1))
        layers.append(self.make_block(4,8,2))
        # layers.append(self.make_block(8,8,1))
        layers.append(self.make_block(8, 8, 1))
        layers.append(self.make_block(8,4,(2,4,2)))
        # layers.append(self.make_block(4,4,1))
        layers.append(self.make_block(4,4,(2,4,2)))
        layers.append(self.make_block(4,1,1))
        self.layers = nn.Sequential(*layers)

    def make_layers_vol2(self):
        layers = []
        layers.append(self.make_block(1,4,1))
        layers.append(self.make_block(4,8,1))
        layers.append(self.make_block(8,12,(2,4,2)))
        layers.append(self.make_block(12,8,(2,4,2)))
        layers.append(self.make_block(8,4,2))
        layers.append(self.make_block(4,1,1))
        self.layers = nn.Sequential(*layers)

    def forward(self, x):
        return self.layers(x)

# ds = nn.Conv3d(1,3,1,stride=2)
x = torch.ones((10,1,16,128,60))
t = torch.ones((10,3,8,64,30))
# t = torch.ones((10,1,16,128,60))
# m = QuadResBlock(1,3, downsample=ds, stride=2)
m=QuadResNet()
m.train()
opt = torch.optim.Adam(m.parameters())
loss = nn.MSELoss()
p=m(x)
print("final: ", p.shape)
# l = loss(p,t)
# l.backward()
# opt.step()