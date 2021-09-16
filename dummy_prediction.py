import torch
import torch.nn as nn
from BPtools.core.bpmodule import *
from BPtools.utils.models import EncoderBN, VarDecoderConv1d_3
from BPtools.trainer.bptrainer import BPTrainer

from grid_3D import *
from model import Discriminator2D
from data_moduls import DummyPredictionDataModul
# from torchvision.utils import make_grid
from torchvision.transforms import ToTensor
from BPtools.utils.trajectory_plot import trajs_to_img_2, traj_to_img, trajs_to_img


class Prediction_trajectory_grid3d(BPModule):
    def __init__(self, traj_encoder, traj_decoder, grid_encoder, merge_z):
        super(Prediction_trajectory_grid3d, self).__init__()
        self.traj_enc = traj_encoder
        self.traj_dec = traj_decoder
        self.grid_enc = grid_encoder
        self.merge_z = merge_z
        self.mse = nn.MSELoss()
        self.losses_keys = ["train", "valid"]

    def mse_diff(self, traj2, pred):
        d_traj2 = traj2[:,:,1:] - traj2[:,:,0:-1]
        d_pred = pred[:,:,1:] - pred[:,:,0:-1]
        return self.mse(d_traj2, d_pred)

    def sampler(self, mu, logvar):
        std = logvar.mul(0.5).exp_()
        eps = torch.FloatTensor(std.size()).normal_().to(std.device)
        return eps.mul(std).add_(mu)

    def kld_loss(self, mu, logvar):
        KL = mu.pow(2).add_(logvar.exp()).mul_(-1).add_(1).add_(logvar)
        return torch.mean(KL).mul_(-0.5)

    def forward(self, traj1, grid1):
        traj_mu, traj_logvar = self.traj_enc(traj1)
        traj_z = self.sampler(traj_mu, traj_logvar)
        grid_z = self.grid_enc(grid1)
        merged_z = self.merge_z(traj_z, grid_z)
        return self.traj_dec(merged_z), traj_mu, traj_logvar

    def training_step(self, optim_configuration, step):
        self.train()
        # for param in self.grid_enc.parameters():
        #     param.requires_grad = False

        epoch_loss = 0
        for traj1, traj2, grid1 in zip(*self.trainer.dataloaders["train"]):
            pred, mu, logvar = self(traj1, grid1)
            loss = self.mse(traj2, pred)
            loss += 10 * self.mse_diff(traj2, pred)
            loss += 0.1 * self.kld_loss(mu, logvar)
            loss.backward()
            epoch_loss += loss.item()
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

    def validation_step(self, step):
        self.eval()
        epoch_loss = 0

        for traj1, traj2, grid1 in zip(*self.trainer.dataloaders["valid"]):
            pred, mu, logvar = self(traj1, grid1)
            loss = self.mse(traj2, pred)
            loss += 10 * self.mse_diff(traj2, pred)
            loss += 0.1 * self.kld_loss(mu, logvar)
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

    def configure_optimizers(self):
        return optim.Adam(list(self.traj_enc.parameters()) +
                          list(self.traj_dec.parameters()) +
                          list(self.merge_z.parameters()) +
                          list(self.grid_enc.parameters())
                          , lr=0.001)


class MergeNet(nn.Module):
    def __init__(self, traj_z_dim=10, grid_z_dim=64):
        super(MergeNet, self).__init__()
        self.traj = traj_z_dim
        self.grid = grid_z_dim
        self.layers = nn.Sequential(
            nn.Linear(traj_z_dim + grid_z_dim, 20),
            nn.BatchNorm1d(20),
            nn.PReLU(),
            nn.Linear(20, traj_z_dim)
        )

    def forward(self, traj_z, grid_z):
        # print("innen ",grid_z.shape)
        # print(grid_z.view(-1, self.grid).shape)
        # print(traj_z.shape)
        return self.layers(torch.cat((traj_z, grid_z.view(-1, self.grid)), axis=1))


if __name__ == "__main__":
    traj_enc = EncoderBN(2, 60, 10)
    traj_dec = VarDecoderConv1d_3(2, 60, 10)
    enc = Encoder_Grid3D_3()
    dec = Decoder_Grid3D_3()
    disc = Discriminator2D()
    aae3d = ADVAE3D(encoder=enc, decoder=dec, discriminator=disc)
    aae3d.load_state_dict(torch.load("model_state_dict_3D_pred_proba_img_type3_50_1_13"))
    grid_enc = aae3d.encoder
    del(aae3d)
    grid_enc = enc
    merge = MergeNet()
    model = Prediction_trajectory_grid3d(traj_enc,traj_dec,grid_enc, merge)
    dm = DummyPredictionDataModul("../dataset", split_ratio=0.2, batch_size=300)
    # dm.prepare_data()
    # for traj1, traj2 in zip(dm.traj_1, dm.traj_2):
    #     print(traj1.shape)
    #     print(traj2.shape)
    #     trajs_to_img_2(tr1=traj1,tr2=traj2,label="valami")
    # dm.setup()
    # for ttraj, ttraj2,_ in zip(*dm.train):
    #     print(ttraj.shape)
    #     for traj, traj2 in zip(ttraj, ttraj2):
    #         print(traj.shape)
    #         trajs_to_img(np.transpose(np.array(traj.to("cpu")), (1,0)), np.transpose(np.array(traj2.to("cpu")), (1,0)), "valami")

    trainer = BPTrainer(epochs=1000, name="dummy_derivate10_gridtrain_prediction_model_fulldata_tanszek")
    trainer.fit(model=model, datamodule=dm)
