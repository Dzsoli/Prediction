import torch
import torch.nn as nn
from torch import Tensor
import torch.nn.functional as F
from matplotlib import pyplot as plt
import numpy as np
import torchvision
import torchvision.datasets as datasets
import torchvision.models as models
from torchvision import transforms
import torch.optim as optim
import time
import tqdm as tqdm
from torch.autograd import Variable
from math import factorial as fct
from typing import Type, Any, Callable, Union, List, Optional
import torchviz
# from resnet3D import conv1x1_3D, conv3x3_3D


def conv3x3_3D(in_planes, out_planes, stride=1, groups=1, dilation=1, kernel_size=3, padding=1):
    """3x3 convolution with padding"""
    return nn.Conv3d(in_planes, out_planes, kernel_size=kernel_size, stride=stride,
                     padding=padding, groups=groups, bias=False, dilation=dilation)


def conv1x1_3D(in_planes, out_planes, stride=1):
    """1x1 convolution"""
    return nn.Conv3d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)


class LeakyHardtanh(nn.Hardtanh):
    def __init__(self, min_val=- 1.0, max_val=1.0, negative_slope=0.01, inplace: bool = False):
        super(LeakyHardtanh, self).__init__(min_val, max_val, inplace)
        self.negative_slope = negative_slope

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        return F.hardtanh(input, self.min_val, self.max_val, self.inplace) + F.leaky_relu(input, self.negative_slope,
                                                                                          self.inplace)


class LeakyHardtanh2(nn.Hardtanh):
    def __init__(self, min_val=- 1.0, max_val=1.0, slope=0.01, inplace: bool = False):
        super(LeakyHardtanh2, self).__init__(min_val, max_val, inplace)
        self.slope = slope

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        return F.hardtanh(input, self.min_val, self.max_val, self.inplace) + self.slope * input


class BasicQuadBlock_3D(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None, groups=1,
                 base_width=64, dilation=1, norm_layer=None, order=3, partial_mix=15):
        super(BasicQuadBlock_3D, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm3d
        if groups != 1 or base_width != 64:
            raise ValueError('BasicBlock only supports groups=1 and base_width=64')
        if dilation > 1:
            raise NotImplementedError("Dilation > 1 not supported in BasicBlock")
        # Both self.conv1 and self.downsample layers downsample the input when stride != 1
        conv_list1 = [conv3x3_3D(inplanes, planes, (stride, stride, 1), kernel_size=(3, 3, 1), padding=(1,1,0))]
        bn_list1 = [norm_layer(planes)]
        conv_list2 = [conv3x3_3D(planes, planes, (1, 1, stride), kernel_size=(1,1,3), padding=(0,0,1))]
        bn_list2 = [norm_layer(planes)]
        # bn_list2 = [nn.InstanceNorm2d(planes)]
        for i in range(order-1):
            conv_list1.append(conv3x3_3D(planes, planes, kernel_size=(3, 3, 1), padding=(1,1,0)))
            bn_list1.append(norm_layer(planes))
            conv_list2.append(conv3x3_3D(planes, planes, kernel_size=(1,1,3), padding=(0,0,1)))
            bn_list2.append(norm_layer(planes))
        self.conv_list1 = nn.ModuleList(conv_list1)
        self.bn_list1 = nn.ModuleList(bn_list1)
        self.conv_list2 = nn.ModuleList(conv_list2)
        self.bn_list2 = nn.ModuleList(bn_list2)
        # self.conv1 = conv3x3(inplanes, planes, stride)
        # self.bn1 = norm_layer(planes)
        self.relu = nn.ReLU(inplace=True)
        # self.mish_relu = nn.Softsign() #inplace=True)
        # self.relu6 = nn.ReLU6(inplace=True)
        # self.leaky_hardtanh = LeakyHardtanh(0., 6., negative_slope=0.1, inplace=True)
        self.leaky_hardtanh = LeakyHardtanh2(-6., 6., slope=0.1, inplace=True)

        # self.conv2 = conv3x3(planes, planes)
        # self.bn2 = norm_layer(planes)
        self.downsample = downsample
        self.stride = stride
        self.pool = nn.AvgPool3d(partial_mix, 1, partial_mix//2)
        self.pool2 = nn.AvgPool3d((1, partial_mix, partial_mix), 1, (0, partial_mix//2, partial_mix//2))

        self.order = order
        self.partial_mix = 3

    def forward(self, x):
        identity = x

        if self.downsample is not None:
            identity = self.downsample(x)

        # t, h, w = identity.shape
        # if identity.shape[2] < 7:
        #     mul = identity
        # else:
        #     mul = self.pool(identity)
        mul = identity
        # print("mul: ", True in torch.isnan(mul))
        out = identity
        internal = x
        first_order = True
        n = 1.0
        for conv1, conv2, bn1, bn2 in zip(self.conv_list1, self.conv_list2, self.bn_list1, self.bn_list2):
            if not first_order:
                internal = self.relu(internal)
            internal = conv1(internal)
            internal = self.relu(bn1(internal))
            internal = conv2(internal)
            internal = bn2(internal)

            if first_order:
                # internal = bn2(internal)
                out = out + internal
                first_order = False
                n = n + 1
                continue
            # ha már túl vagyunk az első renden
            internal = mul * internal
            # internal = self.relu6(internal)
            internal = self.leaky_hardtanh(internal)

            # print("internal: ", True in torch.isnan(internal))
            # internal = (1.0/fct(n)) * internal
            n = n + 1
            out = out + internal

        # out = self.conv1(x)
        # Fx = self.bn1(out)
        # out = self.relu(Fx)
        #
        # out = self.conv2(out)
        # FFx = self.bn2(out)
        #
        #
        #
        # mul = self.pool(identity)
        # xFFx = mul * FFx
        # out = identity + Fx + 0.5 * xFFx

        # out = self.relu6(out)
        out = self.leaky_hardtanh(out)

        return out


class ParallelTaylorBlock(BasicQuadBlock_3D):
    def __init__(self, inplanes, planes, stride=1, downsample=None, groups=1,
                 base_width=64, dilation=1, norm_layer=None, order=3, partial_mix=15):
        super(ParallelTaylorBlock, self).__init__(inplanes, planes, stride, downsample, groups, base_width,
                                                  dilation, norm_layer, order, partial_mix)
        # TODO: conv3x3_3D fix
        conv_list1 = [conv3x3_3D(inplanes, planes, stride)]
        conv_list2 = [conv3x3_3D(planes, planes)]
        # bn_list2 = [nn.InstanceNorm2d(planes)]
        for i in range(order - 1):
            conv_list1.append(conv3x3_3D(inplanes, planes, stride))
            conv_list2.append(conv3x3_3D(planes, planes))
        self.conv_list1 = nn.ModuleList(conv_list1)
        self.conv_list2 = nn.ModuleList(conv_list2)

    def forward(self, x):
        identity = x

        if self.downsample is not None:
            identity = self.downsample(x)
        out = identity
        # internal = x
        first_order = True
        n = 1.0
        for conv1, conv2, bn1, bn2 in zip(self.conv_list1, self.conv_list2, self.bn_list1, self.bn_list2):
            internal = conv1(x)
            internal = self.relu(bn1(internal))
            internal = conv2(internal)
            internal = bn2(internal)

            if first_order:
                # internal = bn2(internal)
                # print(out.shape[1], internal.shape[1])
                out = out + internal
                previous = self.pool(internal)
                first_order = False
                n = n + 1
                continue
            previous = previous * self.pool(internal)
            previous = self.leaky_hardtanh(previous)
            # out = out + (1.0/fct(n)) * previous
            out = out + previous

        return out


class TaylorNet_3D(nn.Module):

    def __init__(self, block, layers, num_classes=1000, zero_init_residual=False,
                 groups=1, width_per_group=64, replace_stride_with_dilation=None,
                 norm_layer=None, order=2, partial_mix=15):
        super(TaylorNet_3D, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm3d
        self._norm_layer = norm_layer

        self.order = order
        self.partial_mix = partial_mix

        self.inplanes = 64
        self.dilation = 1
        if replace_stride_with_dilation is None:
            # each element in the tuple indicates if we should replace
            # the 2x2 stride with a dilated convolution instead
            replace_stride_with_dilation = [False, False, False]
        if len(replace_stride_with_dilation) != 3:
            raise ValueError("replace_stride_with_dilation should be None "
                             "or a 3-element tuple, got {}".format(replace_stride_with_dilation))
        self.groups = groups
        self.base_width = width_per_group
        self.conv1 = nn.Conv3d(3, self.inplanes, kernel_size=7, stride=2, padding=3,
                               bias=False)
        self.bn1 = norm_layer(self.inplanes)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool3d(kernel_size=3, stride=2, padding=1)
        self.maxpool_1 = nn.MaxPool3d(kernel_size=3, stride=1, padding=1)

        self.layer1 = self._make_layer(block, 32, layers[0])
        self.layer2 = self._make_layer(block, 64, layers[1], stride=2,
                                       dilate=replace_stride_with_dilation[0])
        self.layer3 = self._make_layer(block, 128, layers[2], stride=2,
                                       dilate=replace_stride_with_dilation[1])
        self.layer4 = self._make_layer(block, 256, layers[3], stride=2,
                                       dilate=replace_stride_with_dilation[2])
        self.avgpool = nn.AdaptiveAvgPool3d((1, 1, 1))
        self.fc = nn.Linear(256 * block.expansion, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm3d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        # Zero-initialize the last BN in each residual branch,
        # so that the residual branch starts with zeros, and each residual block behaves like an identity.
        # This improves the model by 0.2~0.3% according to https://arxiv.org/abs/1706.02677
        if zero_init_residual:
            for m in self.modules():
                if isinstance(m, models.Bottleneck):
                    nn.init.constant_(m.bn3.weight, 0)
                elif isinstance(m, BasicQuadBlock_3D):
                    nn.init.constant_(m.bn2.weight, 0)

    def _make_layer(self, block, planes, blocks, stride=1, dilate=False):
        norm_layer = self._norm_layer
        downsample = None
        previous_dilation = self.dilation
        if dilate:
            self.dilation *= stride
            stride = 1
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                conv1x1_3D(self.inplanes, planes * block.expansion, stride),
                norm_layer(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample, self.groups,
                            self.base_width, previous_dilation, norm_layer, self.order, self.partial_mix))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes, groups=self.groups,
                                base_width=self.base_width, dilation=self.dilation,
                                norm_layer=norm_layer, order=self.order, partial_mix=self.partial_mix))

        return nn.Sequential(*layers)

    def forward(self, x):
        # See note [TorchScript super()]
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        # x = self.maxpool(x)
        x = self.maxpool_1(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)

        return x


class MyBasicBlock(nn.Module):
    expansion: int = 1

    def __init__(
        self,
        inplanes: int,
        planes: int,
        stride: int = 1,
        downsample: Optional[nn.Module] = None,
        groups: int = 1,
        base_width: int = 64,
        dilation: int = 1,
        norm_layer: Optional[Callable[..., nn.Module]] = None
    ) -> None:
        super(MyBasicBlock, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        if groups != 1 or base_width != 64:
            raise ValueError('BasicBlock only supports groups=1 and base_width=64')
        if dilation > 1:
            raise NotImplementedError("Dilation > 1 not supported in BasicBlock")
        # Both self.conv1 and self.downsample layers downsample the input when stride != 1
        self.conv1 = conv3x3_3D(inplanes, planes, stride)
        self.bn1 = norm_layer(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3_3D(planes, planes)
        self.bn2 = norm_layer(planes)
        self.downsample = downsample
        self.stride = stride
        self.leaky_hardtanh = LeakyHardtanh2(-6., 6., slope=0.1, inplace=True)

    def forward(self, x: Tensor) -> Tensor:
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.leaky_hardtanh(out)

        return out




if __name__ == "__main__":
    # k=3
    # pool = nn.AvgPool2d(k,1,k//2)
    # t = torch.ones((1,5,5))
    # print(t)
    # print(9*pool(t))
    # model = TaylorNet_3D(BasicQuadBlock_3D, [2, 2, 2, 2], num_classes=10)
    # print(model)
    # model = ParallelTaylorBlock(1,1,1,order=4,partial_mix=3)
    model = TaylorNet_3D(ParallelTaylorBlock, [2, 2, 2, 2], num_classes=10, order=5, partial_mix=3)
    x = torch.randn(1,3,50,50)
    y = model(x)
    print(y.shape)
    print(dict(model.named_parameters()).keys())
    torchviz.make_dot((y ** 2).mean(), params=dict(model.named_parameters()), show_attrs=False,
                      show_saved=False).render("Fullmodel_ParallelTaylorBlock_5", format="png")
