import torch
import torch.nn as nn
from BPtools.core.bpmodule import *
from BPtools.utils.models import EncoderBN, VarDecoderConv1d_3
from BPtools.trainer.bptrainer import BPTrainer

from data_moduls import *
from grid_3D import *
from grid_2D import *
from model import Discriminator2D
from data_moduls import DummyPredictionDataModul
# from torchvision.utils import make_grid
from torchvision.transforms import ToTensor
from BPtools.utils.trajectory_plot import trajs_to_img_2, traj_to_img, trajs_to_img
import random


class Maneuver_prediction(BPModule):
    def __init__(self, encoder, grid_encoder):
        super(Maneuver_prediction, self).__init__()
        self.enc = encoder
        self.grid_encoder = grid_encoder
        self.mse = nn.BCELoss()
        self.losses_keys = ["train", "valid"]

    def sampler(self, mu, logvar):
        std = logvar.mul(0.5).exp_()
        eps = torch.FloatTensor(std.size()).normal_().to(std.device)
        return eps.mul(std).add_(mu)

    def kld_loss(self, mu, logvar):
        KL = mu.pow(2).add_(logvar.exp()).mul_(-1).add_(1).add_(logvar)
        return torch.mean(KL).mul_(-0.5)

    def sparse_loss(self, hidden, cell):
        return torch.mean(torch.abs(hidden)) + torch.mean(torch.abs(cell))

    def forward(self, traj_p, grid):
        # traj: batch, feature, seq
        # grid: batch, 1, 16, 128, seq
        batch_size, feature, seq_length = traj_p.size()
        grid_z, _ = self.grid_encoder(grid.permute((0, 4, 1, 2, 3)).reshape((batch_size * seq_length, 1, 16, 128)))
        # batch * seq, 1, 4, 16
