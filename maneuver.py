import torch
import torch.nn as nn
import torch.nn.functional as F
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
    def __init__(self, encoder_traj, encoder_grid, grid_encoder):
        super(Maneuver_prediction, self).__init__()
        self.enc_traj = encoder_traj
        self.enc_grid = encoder_grid
        self.grid_encoder = grid_encoder
        self.mlp_mu = ResidualMLP()
        self.mlp_logvar = ResidualDense()
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

        grid_z = grid_z.squeeze(1).reshape((batch_size, seq_length, 4, 16)).permute(0, 2, 3, 1).reshape(
            batch_size, 64, seq_length)
        # batch, 64, seq

        # csak az utolsó hidden state kell
        grid_z_z = self.enc_grid(grid_z)
        traj_z = self.enc_traj(traj_p)
        # nem kell a cell state
        # összeg, nem konkatenálás
        mu, logvar = self.mlp(grid_z_z + traj_z)


class ResidualMLP(nn.Module):
    # todo: kell residual blokk ősosztály és abból felépíteni skálázhatóan
    pass


class ResBlockFully(nn.Module):
    def __init__(self, in_features, out_features):
        super(ResBlockFully, self).__init__()
        self.lin1 = nn.Linear(in_features, in_features)
        self.lin2 = nn.Linear(in_features, in_features)
        self.lin3 = nn.Linear(in_features, out_features)
        self.bn = nn.BatchNorm1d(out_features)

    def forward(self, x):
        residual = x
        out = F.relu(self.lin1(x))
        out = F.relu(self.lin2(out))

        out += residual
        out = self.bn(self.lin3(out))
        return
