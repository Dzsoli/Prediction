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
from trayprediction_with_grid import GridEncoder, Traj_gridPred

import matplotlib.pyplot as plt


class ResBlock(nn.Module):
    def __init__(self, res):
        super(ResBlock, self).__init__()
        self.res = res

    def forward(self, x):
        return self.res(x) + x


class TrajectoryEncoderVar(nn.Module):
    def __init__(self, context_dim, expands, vae_expands=None, input_channels=2):
        super(TrajectoryEncoderVar, self).__init__()
        self.expand = 2
        self.internal = input_channels

        self.context_dim = context_dim
        self.input_channels = input_channels

        # self.sigmoid = nn.Sigmoid()
        self.is_var = False if vae_expands is None else True
        self.layers = []
        # self.layer1 = self.conv_block(input_channels, self.internal)
        for exp in expands:
            self.block(exp)
        if not self.is_var:
            self.layers.append(nn.Conv1d(self.internal, context_dim, 1))
        self.blocks1 = nn.Sequential(*self.layers)
        self.layers = []
        if self.is_var:
            for exp in vae_expands:
                self.block(exp)
            self.layers.append(nn.Conv1d(self.internal, context_dim, 1))
            self.blocks_mu = nn.Sequential(*self.layers)

            self.internal = self.layers[0][0].in_channels
            self.layers = []

            for exp in vae_expands:
                self.block(exp)
            self.layers.append(nn.Conv1d(self.internal, context_dim, 1))
            self.blocks_logvar = nn.Sequential(*self.layers)
            self.layers = []

    def conv_block(self, exp=True, stride=1, padd=None, pool=False):
        out_ch = self.internal * self.expand if exp else self.internal
        return conv_block1d(self.internal, out_ch, 3, stride, padd, pool)

    def block(self, l):
        if l > 0:
            self.layers.append(self.conv_block(stride=l))
            self.internal = self.internal * self.expand
        elif l == 'p':
            self.layers.append(self.conv_block(pool=True))
            self.internal = self.internal * self.expand
        elif l == 0:
            res = nn.Sequential(self.conv_block(exp=False),
                                self.conv_block(exp=False))
            self.layers.append(ResBlock(res=res))

    def forward(self, x):
        inner = self.blocks1(x)
        if self.is_var:
            mu = self.blocks_mu(inner)
            logvar = self.blocks_logvar(inner)
            return mu, logvar
        else:
            return inner


class TrajectoryDecoderHybrid(nn.Module):
    def __init__(self, context_dim, output_channels=2, transpose=True):
        super(TrajectoryDecoderHybrid, self).__init__()
        self.context_dim = context_dim
        self.output_channels = output_channels
        self.context = nn.Conv1d(context_dim, 16, 1)
        self.sig = nn.Sigmoid()
        self.att = nn.Sequential(
            conv_block1d(context_dim, 8, 5),
            conv_block1d(8, 8),
            conv_block1d(8, 4),
            conv_block1d(4, 3)
        )
        self.layer1 = conv_block1d(16, 16)
        self.layer2 = self.tr_conv_block(16, 16, kernel=4, stride=2, padd=3) if transpose else conv_block1d(16, 16)
        self.layer3 = conv_block1d(16, 8)
        self.res1 = nn.Sequential(conv_block1d(8, 8), conv_block1d(8, 8))
        self.layer4 = self.tr_conv_block(8, 4, kernel=4, stride=3, padd=2) if transpose else conv_block1d(8, 4)
        self.res2 = nn.Sequential(conv_block1d(4, 4), conv_block1d(4, 4))
        self.layer5 = nn.Conv1d(4, output_channels, kernel_size=3, padding=1)
        # self.average = nn.AvgPool1d(kernel_size=5, stride=1, padding=2)

    def tr_conv_block(self, in_ch, out_ch, kernel=3, stride=1, padd=None, dilation=1, pool=False):
        padd = kernel//2 if padd is None else padd
        layers = [nn.ConvTranspose1d(in_ch, out_ch, kernel_size=kernel, stride=stride, padding=padd, output_padding=0,
                                     dilation=dilation),
                  nn.BatchNorm1d(out_ch),
                  nn.ReLU(inplace=True)]
        if pool: layers.append(nn.MaxPool1d(2))
        return nn.Sequential(*layers)

    def forward(self, x, label):
        attention = self.att(x)
        attention = attention * label.unsqueeze(2)
        attention = torch.sum(attention, dim=1).unsqueeze(1)

        x = x * attention

        x = self.context(x)
        out = self.layer1(x)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.res1(out) + out
        out = self.layer4(out)
        out = self.res2(out) + out
        out = self.layer5(out)
        return out


class Traj_gridPred_version2(Traj_gridPred):
    def __init__(self, *args, **kwargs):
        super(Traj_gridPred_version2, self).__init__(*args, **kwargs)

    def forward(self, traj1, grid1, label):
        Gmu, Glogvar = self.grid_encoder(grid1)
        grid_z = self.sampler(Gmu, Glogvar)

        Tmu, Tlogvar = self.traj_encoder(traj1)
        traj_z = self.sampler(Tmu, Tlogvar)

        mul_z = grid_z * traj_z
        pred = self.traj_decoder(mul_z, label)

        return Gmu, Glogvar, Tmu, Tlogvar, pred

    def training_step(self, optim_configuration, step):
        self.train()

        epoch_loss = 0
        epoch_recon = 0
        epoch_Gkld = 0
        epoch_Tkld = 0

        for traj1, traj2, grid1, label in zip(*self.trainer.dataloaders["train"]):
            traj1 = traj1.to("cuda")
            traj2 = traj2.to("cuda")
            grid1 = grid1.to("cuda")
            label = label.to("cuda")

            Gmu, Glogvar, Tmu, Tlogvar, pred = self(traj1, grid1, label)
            Gkld = self.kld_loss(Gmu, Glogvar)
            Tkld = self.kld_loss(Tmu, Tlogvar)

            # concat traj1 & traj2
            loss = self.mse(torch.cat((traj1, traj2), dim=2), pred)

            epoch_loss += loss.item()
            epoch_loss += Gkld.item()
            epoch_loss += Tkld.item()

            epoch_recon += loss.item()
            epoch_Gkld += Gkld.item()
            epoch_Tkld += Tkld.item()

            loss = loss + self.lam * (Gkld + Tkld)
            loss += 10 * self.mse_diff(torch.cat((traj1, traj2), dim=2), pred)

            loss.backward()
            optim_configuration.step()
            optim_configuration.zero_grad()

        N = len(self.trainer.dataloaders["train"][0])
        self.trainer.losses["train"].append(epoch_loss / N)
        if self.is_var:
            self.trainer.writer.add_scalar("Reconstruction/train", epoch_recon / N, step)
            self.trainer.writer.add_scalar("GRID KLD/train", epoch_Gkld / N, step)
            self.trainer.writer.add_scalar("TRAJ KLD/train", epoch_Tkld / N, step)

        indexes = [1, 2, 3, 4, 5, 6]
        img_batch = np.zeros((len(indexes), 3, 480, 640))
        i = 0
        # transform back

        with torch.no_grad():
            # trajektória képek!
            if step % 2 == 0:
                for n in indexes:
                    real_1 = np.transpose(np.array(traj1.to('cpu'))[n], (1, 0))
                    real_2 = np.transpose(np.array(traj2.to('cpu'))[n], (1, 0))
                    out = np.transpose(np.array(pred.to('cpu'))[n], (1, 0))

                    # transform back
                    real_1[:, 1] = real_1[:, 1] * self.Ystd + self.Ymean
                    real_1[:, 0] = real_1[:, 0] * self.Xstd + self.Xmean

                    real_2[:, 1] = real_2[:, 1] * self.Ystd + self.Ymean
                    real_2[:, 0] = real_2[:, 0] * self.Xstd + self.Xmean

                    out[:, 1] = out[:, 1] * self.Ystd + self.Ymean
                    out[:, 0] = out[:, 0] * self.Xstd + self.Xmean


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
        epoch_Gkld = 0
        epoch_Tkld = 0

        for traj1, traj2, grid1, label in zip(*self.trainer.dataloaders["valid"]):
            traj1 = traj1.to("cuda")
            traj2 = traj2.to("cuda")
            grid1 = grid1.to("cuda")
            label = label.to("cuda")
            with torch.no_grad():
                Gmu, Glogvar, Tmu, Tlogvar, pred = self(traj1, grid1, label)
                Gkld = self.kld_loss(Gmu, Glogvar)
                Tkld = self.kld_loss(Tmu, Tlogvar)
                # concat traj1 & traj2
                loss = self.mse(torch.cat((traj1, traj2), dim=2), pred)
                epoch_loss += loss.item()
                epoch_loss += Gkld.item()
                epoch_loss += Tkld.item()

                epoch_recon += loss.item()
                epoch_Gkld += Gkld.item()
                epoch_Tkld += Tkld.item()

        N = len(self.trainer.dataloaders["valid"][0])
        self.trainer.losses["valid"].append(epoch_loss / N)
        if self.is_var:
            self.trainer.writer.add_scalar("Reconstruction/valid", epoch_recon / N, step)
            self.trainer.writer.add_scalar("GRID KLD/valid", epoch_Gkld / N, step)
            self.trainer.writer.add_scalar("TRAJ KLD/valid", epoch_Tkld / N, step)

        indexes = [1, 2, 3, 4, 5, 6]
        img_batch = np.zeros((len(indexes), 3, 480, 640))
        i = 0
        # traj2_mod = traj2 + traj1[:, :, -1][:, :, None]
        # pred_mod = pred + traj1[:, :, -1][:, :, None]
        with torch.no_grad():
            # trajektória képek!
            if step % 2 == 0:
                for n in indexes:
                    real_1 = np.transpose(np.array(traj1.to('cpu'))[n], (1, 0))
                    real_2 = np.transpose(np.array(traj2.to('cpu'))[n], (1, 0))
                    out = np.transpose(np.array(pred.to('cpu'))[n], (1, 0))

                    # transform back
                    real_1[:, 1] = real_1[:, 1] * self.Ystd + self.Ymean
                    real_1[:, 0] = real_1[:, 0] * self.Xstd + self.Xmean

                    real_2[:, 1] = real_2[:, 1] * self.Ystd + self.Ymean
                    real_2[:, 0] = real_2[:, 0] * self.Xstd + self.Xmean

                    out[:, 1] = out[:, 1] * self.Ystd + self.Ymean
                    out[:, 0] = out[:, 0] * self.Xstd + self.Xmean

                    img = trajs_to_img_2("Real and generated. N= " + str(n), traj_1=real_1, traj_2=real_2,
                                         prediction=out)
                    img_real_gen = PIL.Image.open(img)
                    img_real_gen = ToTensor()(img_real_gen)
                    img_batch[i] = img_real_gen[0:3]
                    i = i + 1
                self.trainer.writer.add_images("Valid Real & Out", img_batch, step)
                plt.close('all')

    @property
    def Ystd(self):
        return self.trainer.datamodule.Ystd

    @property
    def Xstd(self):
        return self.trainer.datamodule.Xstd

    @property
    def Ymean(self):
        return self.trainer.datamodule.Ymean

    @property
    def Xmean(self):
        return self.trainer.datamodule.Xmean


if __name__ == "__main__":
    import inspect
    path_tanszek = "C:/Users/oliver/PycharmProjects/full_data/otthonrol"
    path_otthoni = "D:/dataset"
    dm = TrajectoryPredData_version2(path_otthoni, split_ratio=0.2, batch_size=128, pred=15, is_grid=True)
    # dm.prepare_data()
    # dm.setup()

    traj_encoder = TrajectoryEncoderVar(16, [1, 0, 0, 2], [2, 0, 0, 1])
    traj_decoder = TrajectoryDecoderHybrid(16, 2)
    grid_encoder = GridEncoder(16, variational=True)

    model = Traj_gridPred_version2(traj_encoder, traj_decoder, grid_encoder, lam=0.01)
    # print(model)

    trainer = BPTrainer(epochs=5000,
                        name="1MEAN-0STD_tpgd75_sgd-01-0002_seed420_part1")
    trainer.fit(model=model, datamodule=dm)
    # print(traj_encoder)
    #
    # inp = torch.randn((11,2,60))
    # label = torch.randn(11,3)
    # z = traj_encoder(inp)
    # out = traj_decoder(z[1], label)
    # print('inp: ',inp.shape)
    # print('z: ',z[1].shape)
    # print('out: ',out.shape)


    # print(inspect.getsource(type(traj_encoder)))
    # print(type(traj_encoder))
