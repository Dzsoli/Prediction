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


class Prediction_trajectory_recurrent(BPModule):
    def __init__(self, decoder, encoder, grid_encoder):
        super(Prediction_trajectory_recurrent, self).__init__()
        self.enc = encoder
        self.dec = decoder
        self.grid_encoder = grid_encoder
        self.mse = nn.MSELoss()
        self.losses_keys = ["train", "valid"]
        self.teacher_force = 0.5

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

    def sparse_loss(self, hidden, cell):
        return torch.mean(torch.abs(hidden)) + torch.mean(torch.abs(cell))

    def forward(self, traj_p, traj_f, grid):
        # traj: batch, feature, seq
        # grid: batch, 1, 16, 128, seq
        batch_size, feature, seq_length = traj_p.size()
        grid_z, _ = self.grid_encoder(grid.permute((0, 4, 1, 2, 3)).reshape((batch_size * seq_length, 1, 16, 128)))
        # batch * seq, 1, 4, 16
        grid_z = grid_z.squeeze(1).reshape((batch_size, seq_length, 4, 16)).permute(0, 2, 3, 1).reshape(
            batch_size, 64, seq_length)
        # batch, 64, seq

        combined_hidden, combined_cell = self.enc(torch.cat((grid_z, traj_p), axis=1))

        prediction = torch.zeros_like(traj_f)

        decoder_input = traj_p[:, :, -1]

        ret_hidden, ret_cell = combined_hidden, combined_cell

        for t in range(seq_length):
            pred, combined_hidden, combined_cell = self.dec(decoder_input, combined_hidden, combined_cell)
            prediction[:, :, t] = pred
            teacher_force = (random.random() < self.teacher_force) and self.training

            decoder_input = traj_f[:, :, t] if teacher_force else pred

        # prediction: batch, feature, seq
        # ret_hidden/ cell: 1, batch, hidden_dim
        return prediction, ret_hidden, ret_cell

    def training_step(self, optim_configuration, step):
        self.train()

        for param in self.grid_encoder.parameters():
            param.requires_grad = False
        epoch_loss = 0
        for traj1, traj2, grid1 in zip(*self.trainer.dataloaders["train"]):
            traj1 = traj1.to("cuda")
            traj2 = traj2.to("cuda")
            grid1 = grid1.to("cuda")
            prediction, ret_hidden, ret_cell = self(traj1, traj2, grid1)
            loss = self.mse(traj2, prediction)
            loss += 10 * self.mse_diff(traj2, prediction)
            loss += 1 * self.sparse_loss(ret_hidden, ret_cell)
            loss.backward()
            epoch_loss += loss.item()
            optim_configuration.step()
            optim_configuration.zero_grad()
            traj1 = traj1.to("cpu")
            traj2 = traj2.to("cpu")
            grid1 = grid1.to("cpu")

        N = len(self.trainer.dataloaders["train"][0])
        self.trainer.losses["train"].append(epoch_loss / N)

        indexes = [0, 1, 2, 3, 4, 5, 6, 7]
        img_batch = np.zeros((len(indexes), 3, 480, 640))
        i = 0
        with torch.no_grad():
            # trajektória képek!
            if step % 1 == 0:
                for n in indexes:
                    real_1 = np.transpose(np.array(traj1.to('cpu'))[n], (1, 0))
                    real_2 = np.transpose(np.array(traj2.to('cpu'))[n], (1, 0))
                    out = np.transpose(np.array(prediction.to('cpu'))[n], (1, 0))

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
            traj1 = traj1.to("cuda")
            traj2 = traj2.to("cuda")
            grid1 = grid1.to("cuda")
            prediction, ret_hidden, ret_cell = self(traj1, traj2, grid1)
            loss = self.mse(traj2, prediction)
            loss += 10 * self.mse_diff(traj2, prediction)
            loss += 1 * self.sparse_loss(ret_hidden, ret_cell)
            epoch_loss += loss.item()
            traj1 = traj1.to("cpu")
            traj2 = traj2.to("cpu")
            grid1 = grid1.to("cpu")

        N = len(self.trainer.dataloaders["valid"][0])
        self.trainer.losses["valid"].append(epoch_loss / N)

        indexes = [0, 1, 2, 3, 4, 5, 6, 7]
        img_batch = np.zeros((len(indexes), 3, 480, 640))
        i = 0
        with torch.no_grad():
            # trajektória képek!
            if step % 2 == 0:
                for n in indexes:
                    real_1 = np.transpose(np.array(traj1.to('cpu'))[n], (1, 0))
                    real_2 = np.transpose(np.array(traj2.to('cpu'))[n], (1, 0))
                    out = np.transpose(np.array(prediction.to('cpu'))[n], (1, 0))

                    img = trajs_to_img_2("Real and generated. N= " + str(n), traj_1=real_1, traj_2=real_2,
                                         prediction=out)
                    img_real_gen = PIL.Image.open(img)
                    img_real_gen = ToTensor()(img_real_gen)
                    img_batch[i] = img_real_gen[0:3]
                    i = i + 1
                self.trainer.writer.add_images("Valid Real & Out", img_batch, step)

    def configure_optimizers(self):
        return optim.Adam(list(self.enc.parameters()) +
                          list(self.dec.parameters()),
                          lr=0.001)


class RecurrentCombinedEncoder(nn.Module):
    def __init__(self, feature=66, input_size=66, hidden_size=32):
        super(RecurrentCombinedEncoder, self).__init__()
        self.feature = feature
        self.hidden_size = hidden_size
        self.input_size = input_size
        self.linear = nn.Linear(feature, input_size)
        self.relu = nn.ReLU()
        self.rnn = nn.LSTM(input_size, hidden_size, num_layers=2, batch_first=True, bidirectional=False)

    def forward(self, x):
        # x: batch, 66, seq
        batch, feature, seq = x.shape
        mixed = self.relu(self.linear(x.permute(0,2,1).reshape((batch * seq, feature)))).reshape(
            (batch, seq, self.input_size))
        # batch, seq, feature
        # _, (hidden, cell) = self.rnn(x.permute(0,2,1))
        _, (hidden, cell) = self.rnn(mixed)
        return hidden, cell


class RecurrentDecoder(nn.Module):
    def __init__(self, feature=2, input_size=2, hidden_size=32):
        super(RecurrentDecoder, self).__init__()
        self.feature = feature
        self.hidden_size = hidden_size
        self.input_size = input_size
        self.linear = nn.Linear(hidden_size, input_size)
        self.rnn = nn.LSTM(input_size, hidden_size, num_layers=2, batch_first=True, bidirectional=False)

    def forward(self, dec_input, hidden, cell):
        # dec_input: batch, (1), feature
        output, (hidden, cell) = self.rnn(dec_input.unsqueeze(1), (hidden, cell))
        # output: batch, 1, feature
        output = self.linear(output.squeeze(1))
        return output, hidden, cell


if __name__ == "__main__":
    dm = RecurrentPredictionDataModul("../dataset", split_ratio=0.2, batch_size=210)
    # dm = RecurrentPredictionDataModul("D:/dataset", split_ratio=0.2, batch_size=210)
    enc = RecurrentCombinedEncoder()
    dec = RecurrentDecoder()
    # grid
    grid_enc = GridEncoder()
    grid_enc.load_state_dict(torch.load('aae_gauss_grid_encoder_param'))
    model = Prediction_trajectory_recurrent(decoder=dec, encoder=enc, grid_encoder=grid_enc)
    trainer = BPTrainer(epochs=1000, name="0_recurrent_pred_proba_batch2cuda_mixed_encoder_sparseLam1_diff_L2")
    trainer.fit(model=model, datamodule=dm)
