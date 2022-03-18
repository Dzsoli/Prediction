import torch
import torch.nn as nn
from BPtools.core.bpmodule import *
from BPtools.utils.models import EncoderBN, VarDecoderConv1d_3
from BPtools.trainer.bptrainer import BPTrainer

# from MyResNet import *
# from QuadNet import *
from data_moduls import *
from focal_loss import *
from grid_3D import *
# from model import Discriminator2D
from data_moduls import DummyPredictionDataModul
# from torchvision.utils import make_grid
# from torchvision.models import resnet18
# from resnet3D import *
from torchvision.transforms import ToTensor
from BPtools.utils.trajectory_plot import trajs_to_img_2, traj_to_img, trajs_to_img
from TaylorNetClone import conv1x1_3D, conv3x3_3D
from trajectory_prediction_new import conv_block1d

import matplotlib.pyplot as plt


class TrajectoryEncoderVar(nn.Module):
    def __init__(self, context_dim, layers, input_channels=2, att=True, label=True, var=True):
        super(TrajectoryEncoderVar, self).__init__()
        self.context_dim = context_dim
        self.input_channels = input_channels
        self.is_att = att
        self.is_label = label
        self.is_var = var
        self.layers = []


    def conv_block(self, in_ch, out_ch, kernel=3, stride=1, padd=None, pool=False):
        return conv_block1d(in_ch, out_ch, kernel, stride, padd, pool)

    def block(self, ):
        self.layers.append(self.conv_block())

