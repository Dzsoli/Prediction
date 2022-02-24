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
from trajectory_prediction_new import TrajectoryPredData, TrajectoryEncoder, TrajectoryDecoder

import matplotlib.pyplot as plt


class Traj_gridPred(BPModule):
    def __init__(self, traj_encoder, traj_decoder, grid_encoder):
        super(Traj_gridPred, self).__init__()
        self.traj_encoder = traj_encoder
        self.traj_decoder = traj_decoder
        self.grid_encoder = grid_encoder
        self.mse = nn.MSELoss()
        self.losses_keys = ["train", "valid"]

    def mse_diff(self, traj2, pred):
        d_traj2 = traj2[:,:,1:] - traj2[:,:,0:-1]
        d_pred = pred[:,:,1:] - pred[:,:,0:-1]
        return self.mse(d_traj2, d_pred)

    def forward(self, traj1, grid1, label):
        grid_z = self.grid_encoder(grid1)
        traj_z = self.traj_encoder(traj1, label)
        mul = grid_z * traj_z
        pred = self.traj_decoder(mul)
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


