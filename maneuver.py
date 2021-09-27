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
    def __init__(self, encoder_traj, encoder_grid, grid_encoder, keys):
        super(Maneuver_prediction, self).__init__()
        self.enc_traj = encoder_traj
        self.enc_grid = encoder_grid
        self.grid_encoder = grid_encoder
        self.mlp = ResidualMLP()
        self.mlp_mu = ResidualMLP()
        self.mlp_logvar = ResidualMLP()
        self.bce = nn.BCELoss()
        self.losses_keys = ["train", "valid"] + keys
        self.softmax = nn.Softmax()

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
        internal = self.mlp(grid_z_z + traj_z)
        mu = self.mlp_mu(internal)
        logvar = self.mlp_logvar(internal)
        sampled_z = self.softmax(self.sampler(mu, logvar))
        return mu, logvar, sampled_z

    def training_step(self, optim_config, step):
        self.train()

        epoch_loss = 0
        epoch_kld_loss = 0
        for traj_p, grid, labels in zip(*self.trainer.dataloaders["train"]):
            traj_p.to("cuda")
            grid.to("cuda")
            labels.to("cuda")
            mu, logvar, sampled_z = self(traj_p, grid)
            loss = self.bce(sampled_z, labels)
            epoch_loss += loss.item()
            if "kld_train" in self.losses_keys:
                kld_loss = self.kld_loss(mu, logvar)
                epoch_kld_loss += kld_loss.item()
                loss += kld_loss
            loss.backward()
            optim_config.step()
            traj_p.to("cpu")
            grid.to("cpu")
            labels.to("cpu")

    def validation_step(self, step):
        self.eval()
        epoch_loss = 0

class ResidualMLP(nn.Module):
    def __init__(self, init):
        super(ResidualMLP, self).__init__()
        self.blocks: nn.ModuleList = nn.ModuleList()
        m = None
        for n in init:
            if m is None:
                m = n
                continue
            else:
                self.blocks.append(ResBlockFully(m,n))
                m = n

    def forward(self, x):
        for block in self.blocks:
            x = block(x)
        return x


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
        return out

L = [500, 30, 10]
t = torch.rand((10,500))
m = ResidualMLP(L)
# print(m)
print(m(t).shape)