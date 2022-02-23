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
from TaylorNetClone import TaylorNet_3D, BasicQuadBlock_3D

import matplotlib.pyplot as plt


def conv_block1d(in_ch, out_ch, kernel=3, stride=1, padd=None, pool=False):
    padd = kernel // 2 if padd is None else padd
    layers = [nn.Conv1d(in_ch, out_ch, kernel_size=kernel, stride=stride, padding=padd),
              nn.BatchNorm1d(out_ch),
              nn.ReLU(inplace=True)]
    if pool: layers.append(nn.MaxPool1d(2))
    return nn.Sequential(*layers)


class TrajectoryEncoder(nn.Module):
    def __init__(self, context_dim, input_channels=2, seq_length=60, att=True):
        super(TrajectoryEncoder, self).__init__()
        self.context_dim = context_dim
        self.input_channels = input_channels
        self.seq_length = seq_length
        self.is_att = att
        self.layer1 = self.conv_block(input_channels, 4)
        self.layer2 = self.conv_block(4, 8, stride=2)
        self.res1 = nn.Sequential(self.conv_block(8, 8), self.conv_block(8, 8))
        self.layer3 = self.conv_block(8, 16, pool=True)
        self.res2 = nn.Sequential(self.conv_block(16, 16), self.conv_block(16, 16))
        self.context = nn.Conv1d(16, context_dim, 1)

        # Attention like layer
        # self.att = nn.Conv1d(16,1,1)
        # Egy módosítás: legyen több rétegű, hátha az egy réteg képtelen kinyerni a hasznos jellemzőket
        # Másik módosítás: felhasználom az attentionban a
        #       labelt. 3 csatornás lesz, és a one-hot vektorral fogom szorozni
        self.att = nn.Sequential(conv_block1d(16,8,1), conv_block1d(8,3,1))

        # A softmax a 3. index, vagyis az "idő" pontok mentén végződik el, egy csatorna van, tehát minden időponthoz
        # lesz egy súly, ami egyre normált.

        # Sigmoiddal is kipróbálom
        self.softmax = nn.Sigmoid()

    def conv_block(self, in_ch, out_ch, kernel=3, stride=1, padd=None, pool=False):
        return conv_block1d(in_ch, out_ch, kernel, stride, padd, pool)

    def forward(self, x, label):
        out = self.layer1(x)
        out = self.layer2(out)
        out = self.res1(out) + out
        out = self.layer3(out)
        out = self.res2(out) + out

        if self.is_att:
            att = self.softmax(self.att(out))
            att = att * label.unsqueeze(2)
            att = torch.sum(att, dim=1).unsqueeze(1)
            out = self.context(out)
            out = att * out
        else:
            pass
            out = self.context(out)
            # TODO: ide kell valami ötlet
        return out


class TrajectoryDecoder(nn.Module):
    def __init__(self, context_dim, output_channels=2, seq_length=60, transpose=True):
        super(TrajectoryDecoder, self).__init__()
        self.context_dim = context_dim
        self.output_channels = output_channels
        self.seq_length = seq_length
        self.context = nn.Conv1d(context_dim, 16, 1)
        self.layer1 = conv_block1d(16, 16)
        self.layer2 = self.tr_conv_block(16, 16, kernel=4, stride=2, padd=1) if transpose else conv_block1d(16, 16)
        self.layer3 = conv_block1d(16, 8)
        self.res1 = nn.Sequential(conv_block1d(8, 8), conv_block1d(8, 8))
        self.layer4 = self.tr_conv_block(8, 4, kernel=4, stride=2, padd=1) if transpose else conv_block1d(8, 4)
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

    def forward(self, x):
        x = self.context(x)
        out = self.layer1(x)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.res1(out) + out
        out = self.layer4(out)
        out = self.res2(out) + out
        out = self.layer5(out)
        return out


class TrajPred_New(BPModule):
    def __init__(self, traj_encoder, traj_decoder):
        super(TrajPred_New, self).__init__()
        self.traj_enc = traj_encoder
        self.traj_dec = traj_decoder
        self.mse = nn.MSELoss()
        self.losses_keys = ["train", "valid"]

    def mse_diff(self, traj2, pred):
        d_traj2 = traj2[:,:,1:] - traj2[:,:,0:-1]
        d_pred = pred[:,:,1:] - pred[:,:,0:-1]
        return self.mse(d_traj2, d_pred)

    def forward(self, traj1, label):
        traj_z = self.traj_enc(traj1, label)
        pred = self.traj_dec(traj_z)
        return pred

    def training_step(self, optim_configuration, step):
        self.train()

        epoch_loss = 0
        for traj1, traj2, label in zip(*self.trainer.dataloaders["train"]):
            traj1 = traj1.to("cuda")
            traj2 = traj2.to("cuda")
            label = label.to("cuda")

            pred = self(traj1, label)
            loss = self.mse(traj2, pred)
            epoch_loss += loss.item()

            loss += 10 * self.mse_diff(traj2, pred)
            loss.backward()
            optim_configuration.step()
            optim_configuration.zero_grad()

        N = len(self.trainer.dataloaders["train"][0])
        self.trainer.losses["train"].append(epoch_loss / N)

        indexes = [1, 2, 3, 4, 5, 6]
        img_batch = np.zeros((len(indexes), 3, 480, 640))
        i = 0
        traj2_mod = traj2 + traj1[:, :, -1][:, :, None]
        pred_mod = pred + traj1[:, :, -1][:, :, None]
        with torch.no_grad():
            # trajektória képek!
            if step % 10 == 0:
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

        for traj1, traj2, label in zip(*self.trainer.dataloaders["valid"]):
            traj1 = traj1.to("cuda")
            traj2 = traj2.to("cuda")
            label = label.to("cuda")
            with torch.no_grad():
                pred = self(traj1, label)
                loss = self.mse(traj2, pred)
            epoch_loss += loss.item()

            # loss += 10 * self.mse_diff(traj2, pred)
            # loss += 10 * self.mse_diff(traj2, pred)
            # loss += 0.1 * self.kld_loss(mu, logvar)
            epoch_loss += loss.item()

        N = len(self.trainer.dataloaders["valid"][0])
        self.trainer.losses["valid"].append(epoch_loss / N)

        indexes = [1,2,3,4,5,6]
        img_batch = np.zeros((len(indexes), 3, 480, 640))
        i = 0
        traj2_mod = traj2 + traj1[:, :, -1][:, :, None]
        pred_mod = pred + traj1[:, :, -1][:, :, None]
        with torch.no_grad():
            # trajektória képek!
            if step % 10 == 0:
                for n in indexes:

                    real_1 = np.transpose(np.array(traj1.to('cpu'))[n], (1,0))
                    real_2 = np.transpose(np.array(traj2_mod.to('cpu'))[n], (1,0))
                    out = np.transpose(np.array(pred_mod.to('cpu'))[n], (1,0))

                    img = trajs_to_img_2("Real and generated. N= " + str(n), traj_1=real_1, traj_2=real_2, prediction=out)
                    img_real_gen = PIL.Image.open(img)
                    img_real_gen = ToTensor()(img_real_gen)
                    img_batch[i] = img_real_gen[0:3]
                    i = i + 1
                self.trainer.writer.add_images("Valid Real & Out", img_batch, step)
                plt.close('all')

    def configure_optimizers(self):
        return optim.Adam(list(self.traj_enc.parameters()) +
                          list(self.traj_dec.parameters()), lr=0.001, amsgrad=True)


if __name__ == "__main__":
    # model = TrajectoryEncoder(10)
    # t = torch.ones(64,2,60)
    # print(model(t).shape)
    # ct = nn.ConvTranspose1d(16,2,4,stride=2,padding=1,dilation=1,output_padding=0)
    # print(ct(model(t)).shape)
    # z = torch.ones_like(model(t))
    # ct.weight = nn.Parameter(torch.ones_like(ct.weight)/16.0)
    # print(ct(z)[0])
    # decoder = TrajectoryDecoder(10)
    # o = decoder(model(t))
    # print(o.shape)
    # print(t[0,1,:])
    # print(o[0,1,:])
    # pred = TrajPred_New(model, decoder)
    # print(pred(t))

    # decoder = TrajectoryDecoder(10, transpose=False)
    # z = torch.randn((1,10,15))
    # print(decoder(z).shape)

    model = TrajPred_New(TrajectoryEncoder(16),TrajectoryDecoder(16, transpose=False))
    path_tanszek = "C:/Users/oliver/PycharmProjects/full_data/otthonrol"
    path_otthoni = "D:/dataset"
    dm = TrajectoryPredData(path_otthoni, split_ratio=0.2, batch_size=512, pred=15)
    trainer = BPTrainer(epochs=5000, name="trajectory_prediction_new15_deriv_Noatt-double-labelhatMAX_Sigmoid_vol1")
    trainer.fit(model=model, datamodule=dm)
    # trainer = BPTrainer(epochs=3000, name="trajectory_prediction_new_deriv_att-double-label_NoAtt_vol1")
