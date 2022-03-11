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
from trajectory_prediction_new import TrajectoryPredData, TrajectoryEncoder, TrajectoryDecoder

import matplotlib.pyplot as plt


def conv_block3d(in_ch, out_ch, kernel: Tuple, stride=1, padd=None, pool=False):
    layers = [nn.Conv3d(in_ch, out_ch, kernel_size=kernel, stride=stride, padding=padd),
              nn.BatchNorm3d(out_ch),
              nn.ReLU(inplace=True), ]
    return nn.Sequential(*layers)


class Traj_gridPred(BPModule):
    def __init__(self, traj_encoder, traj_decoder, grid_encoder, lam):
        super(Traj_gridPred, self).__init__()
        self.traj_encoder = traj_encoder
        self.traj_decoder = traj_decoder
        self.grid_encoder = grid_encoder
        self.mse = nn.MSELoss()
        self.losses_keys = ["train", "valid"]
        self.is_var = grid_encoder.is_var
        self.lam = lam

    def mse_diff(self, traj2, pred):
        d_traj2 = traj2[:, :, 1:] - traj2[:, :, 0:-1]
        d_pred = pred[:, :, 1:] - pred[:, :, 0:-1]
        return self.mse(d_traj2, d_pred)

    def sampler(self, mu, logvar):
        std = logvar.mul(0.5).exp_()
        eps = torch.FloatTensor(std.size()).normal_().to(std.device)
        return eps.mul(std).add_(mu)

    def kld_loss(self, mu, logvar):
        KL = mu.pow(2).add_(logvar.exp()).mul_(-1).add_(1).add_(logvar)
        return torch.mean(KL).mul_(-0.5)

    def forward(self, traj1, grid1, label):
        if self.is_var:
            mu, logvar = self.grid_encoder(grid1)
            grid_z = self.sampler(mu, logvar)
        else:
            grid_z = self.grid_encoder(grid1).squeeze(2).squeeze(2)
        traj_z = self.traj_encoder(traj1, label)
        mul = grid_z * traj_z
        pred = self.traj_decoder(mul)
        if self.is_var:
            return pred, mu, logvar
        return pred

    def training_step(self, optim_configuration, step):
        self.train()

        epoch_loss = 0
        epoch_recon = 0
        epoch_kld = 0

        for traj1, traj2, grid1, label in zip(*self.trainer.dataloaders["train"]):
            traj1 = traj1.to("cuda")
            traj2 = traj2.to("cuda")
            grid1 = grid1.to("cuda")
            label = label.to("cuda")

            if self.is_var:
                pred, mu, logvar = self(traj1, grid1, label)
                kld = self.kld_loss(mu, logvar)
                loss = self.mse(traj2, pred)

                epoch_loss += loss.item()
                epoch_loss += kld.item()
                epoch_recon += loss.item()
                epoch_kld += kld.item()
                loss = loss + self.lam * kld
                loss += 10 * self.mse_diff(traj2, pred)
            else:
                pred = self(traj1, grid1, label)
                loss = self.mse(traj2, pred)
                epoch_loss += loss.item()
                loss += 10 * self.mse_diff(traj2, pred)
            loss.backward()
            optim_configuration.step()
            optim_configuration.zero_grad()

        N = len(self.trainer.dataloaders["train"][0])
        self.trainer.losses["train"].append(epoch_loss / N)
        if self.is_var:
            self.trainer.writer.add_scalar("Reconstruction/train", epoch_recon / N, step)
            self.trainer.writer.add_scalar("KLD/train", epoch_kld / N, step)

        indexes = [1, 2, 3, 4, 5, 6]
        img_batch = np.zeros((len(indexes), 3, 480, 640))
        i = 0
        traj2_mod = traj2 + traj1[:, :, -1][:, :, None]
        pred_mod = pred + traj1[:, :, -1][:, :, None]
        with torch.no_grad():
            # trajektória képek!
            if step % 2 == 0:
                for n in indexes:
                    real_1 = np.transpose(np.array(traj1.to('cpu'))[n], (1, 0))
                    real_2 = np.transpose(np.array(traj2_mod.to('cpu'))[n], (1, 0))
                    out = np.transpose(np.array(pred_mod.to('cpu'))[n], (1, 0))

                    img = trajs_to_img_2("Real and generated. N= " + str(n), traj_1=real_1, traj_2=real_2,
                                         prediction=out)
                    img_real_gen = PIL.Image.open(img)
                    img_real_gen = ToTensor()(img_real_gen)
                    img_batch[i] = img_real_gen[0:3]
                    i = i + 1
                self.trainer.writer.add_images("Train Real & Out", img_batch, step)
                plt.close('all')

    def validation_step(self, step):
        self.eval()
        epoch_loss = 0
        epoch_recon = 0
        epoch_kld = 0

        for traj1, traj2, grid1, label in zip(*self.trainer.dataloaders["valid"]):
            traj1 = traj1.to("cuda")
            traj2 = traj2.to("cuda")
            grid1 = grid1.to("cuda")
            label = label.to("cuda")
            with torch.no_grad():
                if self.is_var:
                    pred, mu, logvar = self(traj1, grid1, label)
                    kld = self.kld_loss(mu, logvar)
                    loss = self.mse(traj2, pred)
                    epoch_loss += loss.item()
                    epoch_loss += kld.item()
                    epoch_recon += loss.item()
                    epoch_kld += kld.item()
                else:
                    pred = self(traj1, grid1, label)
                    loss = self.mse(traj2, pred)
                    epoch_loss += loss.item()

        N = len(self.trainer.dataloaders["valid"][0])
        self.trainer.losses["valid"].append(epoch_loss / N)
        if self.is_var:
            self.trainer.writer.add_scalar("Reconstruction/valid", epoch_recon / N, step)
            self.trainer.writer.add_scalar("KLD/valid", epoch_kld / N, step)

        indexes = [1, 2, 3, 4, 5, 6]
        img_batch = np.zeros((len(indexes), 3, 480, 640))
        i = 0
        traj2_mod = traj2 + traj1[:, :, -1][:, :, None]
        pred_mod = pred + traj1[:, :, -1][:, :, None]
        with torch.no_grad():
            # trajektória képek!
            if step % 2 == 0:
                for n in indexes:
                    real_1 = np.transpose(np.array(traj1.to('cpu'))[n], (1, 0))
                    real_2 = np.transpose(np.array(traj2_mod.to('cpu'))[n], (1, 0))
                    out = np.transpose(np.array(pred_mod.to('cpu'))[n], (1, 0))

                    img = trajs_to_img_2("Real and generated. N= " + str(n), traj_1=real_1, traj_2=real_2,
                                         prediction=out)
                    img_real_gen = PIL.Image.open(img)
                    img_real_gen = ToTensor()(img_real_gen)
                    img_batch[i] = img_real_gen[0:3]
                    i = i + 1
                self.trainer.writer.add_images("Valid Real & Out", img_batch, step)
                plt.close('all')

    def configure_optimizers(self):
        return optim.Adam(list(self.grid_encoder.parameters()) +
                          list(self.traj_encoder.parameters()) +
                          list(self.traj_decoder.parameters()), lr=0.001, amsgrad=True)


class GridEncoder(nn.Module):
    def __init__(self, context_dim, variational=False):
        super(GridEncoder, self).__init__()
        self.expand = 2
        self.layer1 = conv_block3d(1, 4 * self.expand, kernel=(3, 3, 3), padd=(1, 1, 1))
        self.res1 = nn.Sequential(
            conv_block3d(4 * self.expand, 4 * self.expand, (3, 3, 1), padd=(1, 1, 0)),
            conv_block3d(4 * self.expand, 4 * self.expand, (1, 1, 3), padd=(0, 0, 1))
        )
        self.layer2 = conv_block3d(4 * self.expand, 8 * self.expand, kernel=(3, 3, 3), stride=2,
                                   padd=(1, 1, 1))  # 8, 64, 30
        self.res2 = nn.Sequential(
            conv_block3d(8 * self.expand, 8 * self.expand, (3, 3, 1), padd=(1, 1, 0)),
            conv_block3d(8 * self.expand, 8 * self.expand, (1, 1, 3), padd=(0, 0, 1))
        )
        self.layer3 = conv_block3d(8 * self.expand, 16 * self.expand, (3, 3, 3), stride=(2, 2, 2), padd=1)  # 4, 32, 15
        self.res3 = nn.Sequential(
            conv_block3d(16 * self.expand, 16 * self.expand, (3, 3, 1), padd=(1, 1, 0)),
            conv_block3d(16 * self.expand, 16 * self.expand, (1, 1, 3), padd=(0, 0, 1))
        )
        self.layer4 = conv_block3d(16 * self.expand, 32 * self.expand, (3, 3, 3), stride=(1, 2, 1), padd=1)  # 4, 16, 15
        self.res4 = nn.Sequential(
            conv_block3d(32 * self.expand, 32 * self.expand, (3, 3, 1), padd=(1, 1, 0)),
            conv_block3d(32 * self.expand, 32 * self.expand, (1, 1, 3), padd=(0, 0, 1))
        )
        self.layer5 = conv_block3d(32 * self.expand, 32 * self.expand, (3, 3, 3), stride=(1, 2, 1), padd=1)  # 4, 8, 15
        self.context = nn.Conv3d(32 * self.expand, context_dim, (4, 8, 1), padding=0)  # 1, 1, 15

        self.is_var = variational
        if variational:
            self.layer4_2 = conv_block3d(16 * self.expand, 32 * self.expand, (3, 3, 3), stride=(1, 2, 1),
                                         padd=1)  # 4, 16, 15
            self.res4_2 = nn.Sequential(
                conv_block3d(32 * self.expand, 32 * self.expand, (3, 3, 1), padd=(1, 1, 0)),
                conv_block3d(32 * self.expand, 32 * self.expand, (1, 1, 3), padd=(0, 0, 1))
            )
            self.layer5_2 = conv_block3d(32 * self.expand, 32 * self.expand, (3, 3, 3), stride=(1, 2, 1),
                                         padd=1)  # 4, 8, 15
            self.context_2 = nn.Conv3d(32 * self.expand, context_dim, (4, 8, 1), padding=0)  # 1, 1, 15

    def forward(self, g):
        out = self.layer1(g)
        out = self.res1(out)
        out = self.layer2(out)
        out = self.res2(out)
        out = self.layer3(out)
        out = self.res3(out)

        out1 = self.layer4(out)
        out1 = self.res4(out1)
        out1 = self.layer5(out1)
        z1 = self.context(out1)

        if self.is_var:
            out2 = self.layer4(out)
            out2 = self.res4(out2)
            out2 = self.layer5(out2)
            z2 = self.context(out2)
            return z1.squeeze(2).squeeze(2), z2.squeeze(2).squeeze(2)

        return z1


if __name__ == "__main__":
    grid_encoder = GridEncoder(16, variational=True)
    traj_encoder = TrajectoryEncoder(16, label=True)
    traj_decoder = TrajectoryDecoder(16, transpose=False)

    model = Traj_gridPred(traj_encoder, traj_decoder, grid_encoder, lam=0.01)
    path_tanszek = "C:/Users/oliver/PycharmProjects/full_data/otthonrol"
    path_otthoni = "D:/dataset"

    dm = TrajectoryPredData(path_tanszek, split_ratio=0.2, batch_size=128, pred=15, is_grid=True)
    trainer = BPTrainer(epochs=5000, name="trajectory_prediction_grid15_deriv_att-labelhatMAX_double_Sigmoid_VAE001_vol1")
    trainer.fit(model=model, datamodule=dm)
