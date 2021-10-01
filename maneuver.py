import torch
import torch.nn as nn
import torch.nn.functional as F
from BPtools.core.bpmodule import *
from BPtools.utils.models import EncoderBN, VarDecoderConv1d_3
from BPtools.trainer.bptrainer import BPTrainer

from data_moduls import *
from grid_3D import *
from grid_2D import *
from recurrent_prediction import RecurrentCombinedEncoder
from model import Discriminator2D
from data_moduls import DummyPredictionDataModul
# from torchvision.utils import make_grid
from torchvision.transforms import ToTensor
from BPtools.utils.trajectory_plot import trajs_to_img_2, traj_to_img, trajs_to_img
import random


class Maneuver_prediction(BPModule):
    def __init__(self, internal_size, grid_encoder, keys, encoder_traj=None, encoder_grid=None):
        super(Maneuver_prediction, self).__init__()
        # self.enc_traj = RecurrentEncoder(2, 8, internal_size, num_layers=2) if encoder_traj is None else encoder_traj
        # self.enc_grid = RecurrentEncoder(64, 32, internal_size, num_layers=2)
        self.enc_traj = RecurrentCombinedEncoder(2, 8, internal_size)
        self.enc_grid = RecurrentCombinedEncoder(64, 32, internal_size)
        self.grid_encoder = grid_encoder
        self.mlp = ResidualMLP(np.round(np.linspace(internal_size,internal_size//2,5)))
        self.mlp_mu = ResidualMLP(np.round(np.linspace(internal_size//2,3,2)))
        self.mlp_logvar = ResidualMLP(np.round(np.linspace(internal_size//2,3,2)))
        self.bce = nn.BCELoss()
        self.losses_keys = ["train", "valid"] + keys
        self.softmax = nn.Softmax(dim=1)

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
        # print("grid device: ", grid.device)
        grid_z, _ = self.grid_encoder(grid.permute((0, 4, 1, 2, 3)).reshape((batch_size * seq_length, 1, 16, 128)))
        # batch * seq, 1, 4, 16

        grid_z = grid_z.squeeze(1).reshape((batch_size, seq_length, 4, 16)).permute(0, 2, 3, 1).reshape(
            batch_size, 64, seq_length)
        # batch, 64, seq

        # csak az utolsó hidden state kell
        grid_z_z, _ = self.enc_grid(grid_z)
        traj_z, _ = self.enc_traj(traj_p)
        # nem kell a cell state
        # hiddenből csak utolsó réteg kell
        # összeg, és nem konkatenálás
        grid_z_z = grid_z_z[-1].squeeze(0)
        traj_z = traj_z[-1].squeeze(0)
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
            traj_p = traj_p.to("cuda")
            grid = grid.to("cuda")
            labels = labels.to("cuda")
            mu, logvar, sampled_z = self(traj_p, grid)
            loss = self.bce(sampled_z, labels)
            epoch_loss += loss.item()
            if "kld_train" in self.losses_keys:
                kld_loss = self.kld_loss(mu, logvar)
                epoch_kld_loss += kld_loss.item()
                loss += kld_loss
            loss.backward()
            optim_config.step()
            traj_p = traj_p.to("cpu")
            grid = grid.to("cpu")
            labels = labels.to("cpu")

        N = len(self.trainer.dataloaders["train"][0])
        self.trainer.losses["train"].append(epoch_loss / N)
        self.trainer.losses["kld_train"].append(epoch_kld_loss / N)

    def validation_step(self, step):
        self.eval()

        epoch_loss = 0
        epoch_kld_loss = 0
        for traj_p, grid, labels in zip(*self.trainer.dataloaders["valid"]):
            traj_p = traj_p.to("cuda")
            grid = grid.to("cuda")
            labels = labels.to("cuda")
            mu, logvar, sampled_z = self(traj_p, grid)
            loss = self.bce(sampled_z, labels)
            epoch_loss += loss.item()
            if "kld_valid" in self.losses_keys:
                kld_loss = self.kld_loss(mu, logvar)
                epoch_kld_loss += kld_loss.item()
                loss += kld_loss
            traj_p = traj_p.to("cpu")
            grid = grid.to("cpu")
            labels = labels.to("cpu")
        N = len(self.trainer.dataloaders["valid"][0])
        self.trainer.losses["valid"].append(epoch_loss / N)
        self.trainer.losses["kld_valid"].append(epoch_kld_loss / N)

    def configure_optimizers(self):
        return optim.Adam(list(self.enc_grid.parameters()) +
                          list(self.enc_traj.parameters()) +
                          list(self.mlp.parameters()) +
                          list(self.mlp_mu.parameters()) +
                          list(self.mlp_logvar.parameters()),
                          lr=0.001)


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
                self.blocks.append(ResBlockFully(int(m),int(n)))
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
        self.relu = nn.ReLU()

    def forward(self, x):
        residual = x
        out = self.relu(self.lin1(x))
        out = self.relu(self.lin2(out))

        out += residual
        # TODO: meg kell nézni hogy itt mi a baj, és lehet hogy a hidden-ből csak az utolsó kell
        out = self.bn(self.lin3(out))
        return out


class RecurrentEncoder(nn.Module):
    def __init__(self, in_feature, rnn_feature, hidden_size, num_layers):
        super(RecurrentEncoder, self).__init__()
        self.in_feature = in_feature
        self.rnn_feature = rnn_feature
        self.hidden_size = hidden_size
        self.linear = nn.Linear(in_feature, rnn_feature)
        self.rnn = nn.LSTM(rnn_feature, hidden_size, num_layers=num_layers, batch_first=True, bidirectional=False)

    def forward(self, x):
        pass


if __name__ == "__main__":
    # L = [500, 30, 10]
    # L=np.round(np.linspace(500,10,5))
    # t = torch.rand((10,500))
    # m = ResidualMLP(L)
    grid_enc = GridEncoder()
    # grid_enc.to("cuda")
    dm = RecurrentManeuverDataModul("C:/Users/oliver/PycharmProjects/full_data/otthonrol", split_ratio=0.2, batch_size=80)
    # dm = RecurrentManeuverDataModul("D:/dataset", split_ratio=0.2, batch_size=100)
    grid_enc.load_state_dict(torch.load('aae_gauss_grid_encoder_param'))
    model = Maneuver_prediction(32, grid_enc, ["kld_train", "kld_valid"])
    trainer = BPTrainer(epochs=1000, name="proba_recurrent_maneuver_detection")
    trainer.fit(model=model, datamodule=dm)
    # print(m)
    # print(m(t).shape)
