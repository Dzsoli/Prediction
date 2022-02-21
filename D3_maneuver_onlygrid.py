import torch
import torch.nn as nn
from BPtools.core.bpmodule import *
from BPtools.utils.models import EncoderBN, VarDecoderConv1d_3
from BPtools.trainer.bptrainer import BPTrainer

from MyResNet import *
from QuadNet import *
from data_moduls import *
from focal_loss import *
from grid_3D import *
from model import Discriminator2D
from data_moduls import DummyPredictionDataModul
# from torchvision.utils import make_grid
# from torchvision.models import resnet18
from resnet3D import *
from torchvision.transforms import ToTensor
from BPtools.utils.trajectory_plot import trajs_to_img_2, traj_to_img, trajs_to_img
from TaylorNetClone import TaylorNet_3D, BasicQuadBlock_3D


class Prediction_maneuver_grid3d(BPModule):
    def __init__(self, grid_encoder, merge_z=None, loss=FocalLossMulty([0.2,0.2,0.6],5)):
        super(Prediction_maneuver_grid3d, self).__init__()
        # self.traj_enc = traj_encoder
        # self.traj_dec = traj_decoder
        self.grid_enc = grid_encoder
        self.merge_z = merge_z
        self.mse = loss
        self.losses_keys = ["train", "valid"]
        self.logsoftmax = nn.LogSoftmax(dim=1)

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

    def forward(self, grid1):
        # traj_mu, traj_logvar = self.traj_enc(traj1)
        # traj_z = self.sampler(traj_mu, traj_logvar)
        grid_z = self.grid_enc(grid1)
        if self.merge_z is None:
            result = self.logsoftmax(grid_z.view(-1,3))
        else:
            result = self.merge_z(grid_z)
        return result

    def training_step(self, optim_configuration, step):
        self.train()
        # for param in self.grid_enc.parameters():
        #     param.requires_grad = False

        epoch_loss = 0
        epoch_kld_loss = 0
        epoch_scores = {'tp': 0, 'fn': 0, 'fp': 0, 'tn': 0}
        for _, grid, labels in zip(*self.trainer.dataloaders["train"]):
            # print(grid.device)
            grid = grid.to("cuda")
            # print("2", grid.device)
            labels = labels.to("cuda")
            # mu, logvar, sampled_z = self(grid)
            result = self(grid)
            loss = self.mse(result, labels)
            epoch_loss += loss.item()
            # loss += 10 * self.mse_diff(traj2, pred)
            # loss += 0.1 * self.kld_loss(mu, logvar)
            # if "kld_train" in self.losses_keys:
            #     kld_loss = 0.1 * self.kld_loss(mu, logvar)
            #     epoch_kld_loss += kld_loss.item()
            #     loss += kld_loss
            scores = calc_scores(torch.exp(result), labels)
            loss.backward()
            epoch_loss += loss.item()

            optim_configuration.step()
            optim_configuration.zero_grad()
            for key, value in scores.items():
                epoch_scores[key] += value
            # grid = grid.to("cpu")
            # labels = labels.to("cpu")

        FSCORE = []
        for i in range(3):
            FSCORE.append(
                epoch_scores['tp'][i] / (epoch_scores['tp'][i] + 0.5 * (epoch_scores['fp'][i] + epoch_scores['fn'][i])))
        # print(FSCORE)
        # SZ=epoch_scores['tp'][1].float() + epoch_scores['tn'][1]
        # N=epoch_scores['tp'][1] + epoch_scores['tn'][1] + epoch_scores['fp'][1] + epoch_scores['fn'][1]
        # Q=SZ/N
        ACC_keep = (epoch_scores['tp'][1] + epoch_scores['tn'][1]).float() / (epoch_scores['tp'][1] + epoch_scores['tn'][1] +
                                                                      epoch_scores['fp'][1] + epoch_scores['fn'][1])
        ACC_change = 1.0*(epoch_scores['tp'][0] + epoch_scores['tn'][0] + epoch_scores['tp'][2] + epoch_scores['tn'][2]) / (
                epoch_scores['tp'][0] + epoch_scores['tn'][0] + epoch_scores['fp'][0] + epoch_scores['fn'][0] +
                epoch_scores['tp'][2] + epoch_scores['tn'][2] + epoch_scores['fp'][2] + epoch_scores['fn'][2])
        N = len(self.trainer.dataloaders["train"][0])
        self.trainer.losses["train"].append(epoch_loss / N)
        # self.trainer.losses["kld_train"].append(epoch_kld_loss / N)
        self.trainer.writer.add_scalars('train_FScores', {'left': FSCORE[0], 'keep': FSCORE[1], 'right': FSCORE[2]},
                                        step)
        self.trainer.writer.add_scalars('train_ACC', {'keep': ACC_keep, 'change': ACC_change}, step)


    def validation_step(self, step):
        self.train()
        # for param in self.grid_enc.parameters():
        #     param.requires_grad = False

        epoch_loss = 0
        epoch_kld_loss = 0
        epoch_scores = {'tp': 0, 'fn': 0, 'fp': 0, 'tn': 0}
        for _, grid, labels in zip(*self.trainer.dataloaders["valid"]):
            grid = grid.to("cuda")
            labels = labels.to("cuda")
            # mu, logvar, sampled_z = self(grid)
            result = self(grid)
            loss = self.mse(result, labels)
            epoch_loss += loss.item()
            # loss += 10 * self.mse_diff(traj2, pred)
            # loss += 0.1 * self.kld_loss(mu, logvar)
            # if "kld_train" in self.losses_keys:
            #     kld_loss = 0.1 * self.kld_loss(mu, logvar)
            #     epoch_kld_loss += kld_loss.item()
            #     loss += kld_loss
            scores = calc_scores(torch.exp(result), labels)
            epoch_loss += loss.item()

            for key, value in scores.items():
                epoch_scores[key] += value
            # grid = grid.to("cpu")
            # labels = labels.to("cpu")

        FSCORE = []
        for i in range(3):
            FSCORE.append(
                epoch_scores['tp'][i] / (epoch_scores['tp'][i] + 0.5 * (epoch_scores['fp'][i] + epoch_scores['fn'][i])))
        # print(FSCORE)
        ACC_keep = (epoch_scores['tp'][1] + epoch_scores['tn'][1]).float() / (epoch_scores['tp'][1] + epoch_scores['tn'][1] +
                                                                      epoch_scores['fp'][1] + epoch_scores['fn'][1])
        ACC_change = (epoch_scores['tp'][0] + epoch_scores['tn'][0] + epoch_scores['tp'][2] + epoch_scores['tn'][2]).float() / (
                epoch_scores['tp'][0] + epoch_scores['tn'][0] + epoch_scores['fp'][0] + epoch_scores['fn'][0] +
                epoch_scores['tp'][2] + epoch_scores['tn'][2] + epoch_scores['fp'][2] + epoch_scores['fn'][2])
        N = len(self.trainer.dataloaders["valid"][0])
        self.trainer.losses["valid"].append(epoch_loss / N)
        # self.trainer.losses["kld_train"].append(epoch_kld_loss / N)
        self.trainer.writer.add_scalars('valid_FScores', {'left': FSCORE[0], 'keep': FSCORE[1], 'right': FSCORE[2]},
                                        step)
        self.trainer.writer.add_scalars('valid_ACC', {'keep': ACC_keep, 'change': ACC_change}, step)

    def configure_optimizers(self):
        merge_parameters = list(self.merge_z.parameters()) if self.merge_z is not None else []
        return optim.Adam(
            # list(self.traj_enc.parameters()) +
            #               list(self.traj_dec.parameters()) +
                          merge_parameters +
                          list(self.grid_enc.parameters())
                          , lr=0.001, amsgrad=True, weight_decay=0)
        # return optim.SGD(
        #     # list(self.traj_enc.parameters()) +
        #     #               list(self.traj_dec.parameters()) +
        #     list(self.merge_z.parameters()) +
        #     list(self.grid_enc.parameters())
        #     , lr=0.001, weight_decay=0.0005)


class Grid3D_z_classifier(nn.Module):
    def __init__(self):
        super(Grid3D_z_classifier, self).__init__()
        self.lay1 = nn.Linear(64,16)
        self.lay2 = nn.Linear(16,16)
        self.lay3 = nn.Linear(16,16)
        self.lay4 = nn.Linear(16,3)
        self.bn1 = nn.BatchNorm1d(16)
        self.bn2 = nn.BatchNorm1d(16)
        self.bn3 = nn.BatchNorm1d(16)
        self.relu = nn.ReLU()
        self.logsoftmax = nn.LogSoftmax(dim=1)

    def forward(self, x):
        x1 = self.bn1(self.relu(self.lay1(x.view(-1, 64))))
        x2 = self.bn2(self.relu(self.lay2(x1)))
        x3 = self.bn3(self.relu(self.lay3(x1 + x2)))
        x4 = self.lay4(x3)
        return self.logsoftmax(x4)


if __name__ == "__main__":
    # traj_enc = EncoderBN(2, 60, 10)
    # traj_dec = VarDecoderConv1d_3(2, 60, 10)

    # enc = Encoder_Grid3D_3()

    # enc = MyResNet(MyResBlock, mode=2, type="encoder")
    # enc = QuadResNet()
    # print("Quad res net", sum(p.numel() for p in enc.parameters() if p.requires_grad))

    # bassza meg itt végig 64 numclass vooolt: nem mert a Grid3D_z_classifier vár 64 dimenziót
    res18 = resnet18_3D(num_classes=3, pretrained=False)
    print("Res net 18", sum(p.numel() for p in res18.parameters() if p.requires_grad))

    taylor18_3D = TaylorNet_3D(BasicQuadBlock_3D, [2, 2, 2, 2], num_classes=3, order=2, partial_mix=7)
    # print(res18)
    # dec = Decoder_Grid3D_3()
    # disc = Discriminator2D()
    # aae3d = ADVAE3D(encoder=enc, decoder=dec, discriminator=disc)
    # aae3d.load_state_dict(torch.load("model_state_dict_3D_pred_proba_img_type3_50_1_13"))
    # grid_enc = aae3d.encoder

    # del(aae3d)
    grid_enc = taylor18_3D
    merge = Grid3D_z_classifier()
    # model = Prediction_maneuver_grid3d(grid_enc, merge)

    model = Prediction_maneuver_grid3d(grid_enc, merge_z=None, loss=FocalLossMulty([0.178,0.042,0.78],5))
    # dm = RecurrentManeuverDataModul("C:/Users/oliver/PycharmProjects/full_data/otthonrol", split_ratio=0.2,
    #                                 batch_size=64, dsampling=1)
    # model.load_state_dict(torch.load(
    #     "log_3d_Taylor18_NM_split_AMSGrad_3D_gamma5_ACC/model_state_dict_3d_Taylor18_NM_split_AMSGrad_3D_gamma5_ACC"
    # ))
    dm = RecurrentManeuverDataModul("D:/dataset", split_ratio=0.2, batch_size=16, dsampling=1, resnet18=True)

    # dm = RecurrentManeuverDataModul("D:/dataset", split_ratio=0.2, batch_size=50, dsampling=1)

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



    trainer = BPTrainer(epochs=1000, name="3d_Resnet18_NM_split_AMSGrad_lr1_w0_prob")

    trainer.fit(model=model, datamodule=dm)
