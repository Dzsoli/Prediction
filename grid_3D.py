import torch
import torch.nn as nn
from BPtools.core.bpmodule import *
from torchvision.utils import make_grid
from BPtools.utils.trajectory_plot import boundary_for_grid
import torch.nn.functional as F


class Encoder_Grid3D_3(nn.Module):
    def __init__(self, kernel=2):
        super(Encoder_Grid3D_3, self).__init__()
        T = 4
        self.layers = nn.Sequential(
            nn.Conv3d(1, 3, kernel_size=(2, 5, 3), stride=1, padding=(2, 2, 1)),
            nn.LeakyReLU(0.2),

            nn.Conv3d(3, 5, kernel_size=(3, 8, 3), stride=1, padding=(1, 1, 1)),
            nn.BatchNorm3d(5),
            nn.LeakyReLU(0.2),

            nn.Conv3d(5, 8, kernel_size=(3, 8, T), stride=1, padding=0),
            nn.BatchNorm3d(8),
            nn.LeakyReLU(0.2),

            nn.Conv3d(8, 5, kernel_size=(4, 4, T), stride=(1, 2, 1), padding=(1, 2, 0)),
            nn.BatchNorm3d(5),
            nn.LeakyReLU(0.2),

            nn.Conv3d(5, 4, kernel_size=(4, 4, T), stride=(2, 2, 1), padding=(1, 2, 0)),
            nn.LeakyReLU(0.2),

            nn.Conv3d(4, 3, kernel_size=(4, 4, T-1), stride=(2, 2, 1), padding=(1, 2, 0)),
            nn.LeakyReLU(0.2),

            nn.Conv3d(3, 1, kernel_size=1, padding=0)
        )

    def forward(self, x):
        return self.layers(x).squeeze(4)


class Decoder_Grid3D_3(nn.Module):
    def __init__(self, latent_dim=155):
        super(Decoder_Grid3D_3, self).__init__()
        feature = 4
        T = 4

        self.convtr = nn.Sequential(
            nn.ConvTranspose3d(1, feature, kernel_size=(3, 3, 1), stride=1, padding=0),
            nn.BatchNorm3d(feature),
            nn.LeakyReLU(0.2),
            # N, 32, 4, 16

            nn.ConvTranspose3d(feature, feature, kernel_size=(3, 3, 1), stride=(1, 1, 1), padding=(2, 2, 0)),
            nn.BatchNorm3d(feature),
            nn.LeakyReLU(0.2),

            nn.ConvTranspose3d(feature, feature * 2, kernel_size=(4, 4, 3), stride=(2, 2, 1), padding=(1, 2, 1)),
            nn.BatchNorm3d(feature * 2),
            nn.LeakyReLU(0.2),

            nn.ConvTranspose3d(feature * 2, feature * 3, kernel_size=(4, 4, T), stride=(1, 2, 1), padding=(1, 2, 0)),
            nn.BatchNorm3d(feature * 3),
            nn.LeakyReLU(0.2),

            nn.ConvTranspose3d(feature * 3, feature * 2, kernel_size=(3, 8, T), stride=(1, 1, 1), padding=(1, 1, 0)),
            nn.BatchNorm3d(feature * 2),
            nn.LeakyReLU(0.2),

            nn.ConvTranspose3d(feature * 2, feature, kernel_size=(3, 8, T), stride=(2, 2, 1), padding=(1, 2, 0),
                               output_padding=(0, 1, 0)),
            nn.BatchNorm3d(feature),
            nn.LeakyReLU(0.2),

            nn.ConvTranspose3d(feature, 1, kernel_size=(2, 4, T-1), stride=1, padding=(1, 2, 0)),
            # nn.AdaptiveAvgPool2d((16, 128)),
            nn.Sigmoid()
            # N, 1, 16, 128

        )

    def forward(self, l):
        return self.convtr(l.unsqueeze(4))


class Encoder_Grid3D_2(nn.Module):
    def __init__(self, kernel=2):
        super(Encoder_Grid3D_2, self).__init__()
        T = 4
        self.layers = nn.Sequential(
            nn.Conv3d(1, 3, kernel_size=(2, 5, 1), stride=1, padding=(2, 2, 0)),
            nn.LeakyReLU(0.2),

            nn.Conv3d(3, 5, kernel_size=(3, 8, 1), stride=1, padding=(1, 1, 0)),
            nn.BatchNorm3d(5),
            nn.LeakyReLU(0.2),

            nn.Conv3d(5, 8, kernel_size=(3, 8, T), stride=1, padding=0),
            nn.BatchNorm3d(8),
            nn.LeakyReLU(0.2),

            nn.Conv3d(8, 5, kernel_size=(4, 4, T), stride=(1, 2, 1), padding=(1, 2, 0)),
            nn.BatchNorm3d(5),
            nn.LeakyReLU(0.2),

            nn.Conv3d(5, 4, kernel_size=(4, 4, T), stride=(2, 2, 1), padding=(1, 2, 0)),
            nn.LeakyReLU(0.2),

            nn.Conv3d(4, 3, kernel_size=(4, 4, T-1), stride=(2, 2, 1), padding=(1, 2, 0)),
            nn.LeakyReLU(0.2),

            nn.Conv3d(3, 1, kernel_size=1, padding=0)
        )

    def forward(self, x):
        return self.layers(x).squeeze(4)


class Decoder_Grid3D_2(nn.Module):
    def __init__(self, latent_dim=155):
        super(Decoder_Grid3D_2, self).__init__()
        feature = 4
        T = 4

        self.convtr = nn.Sequential(
            nn.ConvTranspose3d(1, feature, kernel_size=(3, 3, 1), stride=1, padding=0),
            nn.BatchNorm3d(feature),
            nn.LeakyReLU(0.2),
            # N, 32, 4, 16

            nn.ConvTranspose3d(feature, feature, kernel_size=(3, 3, 1), stride=(1, 1, 1), padding=(2, 2, 0)),
            nn.BatchNorm3d(feature),
            nn.LeakyReLU(0.2),

            nn.ConvTranspose3d(feature, feature * 2, kernel_size=(4, 4, 1), stride=(2, 2, 1), padding=(1, 2, 0)),
            nn.BatchNorm3d(feature * 2),
            nn.LeakyReLU(0.2),

            nn.ConvTranspose3d(feature * 2, feature * 3, kernel_size=(4, 4, T), stride=(1, 2, 1), padding=(1, 2, 0)),
            nn.BatchNorm3d(feature * 3),
            nn.LeakyReLU(0.2),

            nn.ConvTranspose3d(feature * 3, feature * 2, kernel_size=(3, 8, T), stride=(1, 1, 1), padding=(1, 1, 0)),
            nn.BatchNorm3d(feature * 2),
            nn.LeakyReLU(0.2),

            nn.ConvTranspose3d(feature * 2, feature, kernel_size=(3, 8, T), stride=(2, 2, 1), padding=(1, 2, 0),
                               output_padding=(0, 1, 0)),
            nn.BatchNorm3d(feature),
            nn.LeakyReLU(0.2),

            nn.ConvTranspose3d(feature, 1, kernel_size=(2, 4, T-1), stride=1, padding=(1, 2, 0)),
            # nn.AdaptiveAvgPool2d((16, 128)),
            nn.Sigmoid()
            # N, 1, 16, 128

        )

    def forward(self, l):
        return self.convtr(l.unsqueeze(4))


class Encoder_Grid3D(nn.Module):
    def __init__(self, kernel=2):
        super(Encoder_Grid3D, self).__init__()
        T = 4
        self.layers = nn.Sequential(
            nn.Conv3d(1, 3, kernel_size=(2, 5, T), stride=1, padding=(2, 2, 0)),
            nn.LeakyReLU(0.2),

            nn.Conv3d(3, 5, kernel_size=(3, 8, T), stride=1, padding=(1, 1, 0)),
            nn.BatchNorm3d(5),
            nn.LeakyReLU(0.2),

            nn.Conv3d(5, 8, kernel_size=(3, 8, T), stride=1, padding=0),
            nn.BatchNorm3d(8),
            nn.LeakyReLU(0.2),

            nn.Conv3d(8, 5, kernel_size=(4, 4, T-1), stride=(1, 2, 1), padding=(1, 2, 0)),
            nn.BatchNorm3d(5),
            nn.LeakyReLU(0.2),

            nn.Conv3d(5, 4, kernel_size=(4, 4, 1), stride=(2, 2, 1), padding=(1, 2, 0)),
            nn.LeakyReLU(0.2),

            nn.Conv3d(4, 3, kernel_size=(4, 4, 1), stride=(2, 2, 1), padding=(1, 2, 0)),
            nn.LeakyReLU(0.2),

            nn.Conv3d(3, 1, kernel_size=1, padding=0)
        )

    def forward(self, x):
        return self.layers(x).squeeze(4)


class Decoder_Grid3D(nn.Module):
    def __init__(self, latent_dim=155):
        super(Decoder_Grid3D, self).__init__()
        feature = 4
        T = 4

        self.convtr = nn.Sequential(
            nn.ConvTranspose3d(1, feature, kernel_size=(3, 3, T), stride=1, padding=0),
            nn.BatchNorm3d(feature),
            nn.LeakyReLU(0.2),
            # N, 32, 4, 16

            nn.ConvTranspose3d(feature, feature, kernel_size=(3, 3, T), stride=(1, 1, 1), padding=(2, 2, 0)),
            nn.BatchNorm3d(feature),
            nn.LeakyReLU(0.2),

            nn.ConvTranspose3d(feature, feature * 2, kernel_size=(4, 4, T), stride=(2, 2, 1), padding=(1, 2, 0)),
            nn.BatchNorm3d(feature * 2),
            nn.LeakyReLU(0.2),

            nn.ConvTranspose3d(feature * 2, feature * 3, kernel_size=(4, 4, T-1), stride=(1, 2, 1), padding=(1, 2, 0)),
            nn.BatchNorm3d(feature * 3),
            nn.LeakyReLU(0.2),

            nn.ConvTranspose3d(feature * 3, feature * 2, kernel_size=(3, 8, 1), stride=(1, 1, 1), padding=(1, 1, 0)),
            nn.BatchNorm3d(feature * 2),
            nn.LeakyReLU(0.2),

            nn.ConvTranspose3d(feature * 2, feature, kernel_size=(3, 8, 1), stride=(2, 2, 1), padding=(1, 2, 0),
                               output_padding=(0, 1, 0)),
            nn.BatchNorm3d(feature),
            nn.LeakyReLU(0.2),

            nn.ConvTranspose3d(feature, 1, kernel_size=(2, 4, 1), stride=1, padding=(1, 2, 0)),
            # nn.AdaptiveAvgPool2d((16, 128)),
            nn.Sigmoid()
            # N, 1, 16, 128

        )

    def forward(self, l):
        return self.convtr(l.unsqueeze(4))


class ADVAE3D(BPModule):
    def __init__(self, encoder, decoder, discriminator):
        super(ADVAE3D, self).__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.discriminator = discriminator
        self.bce = nn.BCELoss()
        self.losses_keys = ['disc train', 'generator train', 'disc valid', 'generator valid']

    def forward(self, x):
        z = self.encoder(x)
        pred = self.decoder(z)  # return h
        return pred, z

    def training_step(self, optim_configuration, step):
        self.train()

        epoch_recon = 0
        epoch_gen = 0
        epoch_disc = 0
        for batch in self.trainer.dataloaders["train"]:
            z = self.encoder(batch)
            # itt baj lehet, nem  a self(batch) megy

            ### Disc
            z_real = torch.FloatTensor(z.size()).normal_().to(z.device)
            d_real = self.discriminator(z_real)
            d_fake = self.discriminator(z)

            loss_real = self.bce(d_real, torch.ones_like(d_real))
            loss_fake = self.bce(d_fake, torch.zeros_like(d_fake))
            disc_loss = (loss_real + loss_fake) / 2
            disc_loss.backward(retain_graph=True)
            epoch_disc = epoch_disc + disc_loss.item()
            # optim_configuration[0][2].step()
            # optim_configuration[1][2].step()
            # !Annealing
            optim_configuration[2].step()

            self.optimizer_zero_grad(0, 0, optim_configuration, step)

            ### Generator
            # mu, logvar = self.encoder(self.trainer.dataloaders["train"])
            # itt meglátom mi lesz detach nélkül
            # z = self.sampler(mu, logvar)
            d_fake = self.discriminator(z)
            gen_loss = self.bce(d_fake, torch.ones_like(d_fake))
            # if step > 2000:
            gen_loss.backward()
            epoch_gen = epoch_gen + gen_loss.item()
            # optim_configuration[0][0].step()
            # optim_configuration[1][0].step()
            # !Annealing
            optim_configuration[0].step()
            self.optimizer_zero_grad(0, 0, optim_configuration, step)

            ### Reconstruction
            pred, z = self(batch)
            recon_loss_vae = self.bce(pred, batch)
            recon_loss_vae.backward()
            epoch_recon = epoch_recon + recon_loss_vae.item()
            # opt_vae
            # optim_configuration[0][1].step()
            # optim_configuration[1][1].step()
            # !Annealing
            optim_configuration[1].step()
            self.optimizer_zero_grad(0, 0, optim_configuration, step)

        N = len(self.trainer.dataloaders["train"])
        self.trainer.losses["train"].append(epoch_recon / N)
        self.trainer.losses["disc train"].append(epoch_disc / N)
        self.trainer.losses["generator train"].append(epoch_gen / N)

    def validation_step(self, step):
        self.eval()
        self.freeze()
        epoch_recon = 0
        epoch_gen = 0
        epoch_disc = 0
        for batch in self.trainer.dataloaders["valid"]:
            pred, z = self(batch)
            ### Disc
            z_real = torch.FloatTensor(z.size()).normal_().to(z.device)
            d_real = self.discriminator(z_real)
            d_fake = self.discriminator(z)
            loss_real = self.bce(d_real, torch.ones_like(d_real))
            loss_fake = self.bce(d_fake, torch.zeros_like(d_fake))
            disc_loss = (loss_real + loss_fake) / 2
            epoch_disc = epoch_disc + disc_loss.item()

            ### Generator
            gen_loss = self.bce(d_fake, torch.ones_like(d_fake))
            # if step > 2000:
            epoch_gen = epoch_gen + gen_loss.item()

            ### Reconstruction
            recon_loss_vae = self.bce(pred, batch)
            epoch_recon = epoch_recon + recon_loss_vae.item()

        # var = pred.exp_()
        # mean_of_var = torch.mean(var)
        # std_of_var = torch.std(var)

        N = len(self.trainer.dataloaders["valid"])
        self.trainer.losses["valid"].append(epoch_recon / N)
        self.trainer.losses["disc valid"].append(epoch_disc / N)
        self.trainer.losses["generator valid"].append(epoch_gen / N)

        # Images
        # itt meg egy-egy idősort kell kiírni
        # '''
        with torch.no_grad():
            # if step % 10 == 1:
            img_fake_grid = make_grid(boundary_for_grid(pred[50].permute(3,0,1,2)), normalize=True, nrow=1)
            img_real_grid = make_grid(boundary_for_grid(batch[50].permute(3,0,1,2)), normalize=True, nrow=1)

            img_latent_dist_grid = make_grid(boundary_for_grid(z[:16]), normalize=True, nrow=2)
            img_prior_dist_grid = make_grid(boundary_for_grid(z_real[:16]), normalize=True, nrow=2)

            self.trainer.writer.add_image("Occupancy Real Images", img_real_grid)
            self.trainer.writer.add_image("Occupancy Fake Images", img_fake_grid, step)

            self.trainer.writer.add_image("Latent Distribution Images", img_latent_dist_grid, step)
            self.trainer.writer.add_image("Prior Distribution Images", img_prior_dist_grid, step)

        # '''
        self.unfreeze()

    def configure_optimizers(self):
        opt_encoder = optim.Adam(self.encoder.parameters(), lr=0.001)
        opt_vae = optim.Adam(list(self.encoder.parameters()) + list(self.decoder.parameters()), lr=0.001)
        opt_disc = optim.SGD(self.discriminator.parameters(), lr=0.001)

        # sch_enc = torch.optim.lr_scheduler.CosineAnnealingLR(opt_encoder, T_max=1500)
        # sch_vae = torch.optim.lr_scheduler.MultiStepLR(opt_vae, milestones=[8000, 80000, 120000, 170000], gamma=0.8)
        # sch_disc = torch.optim.lr_scheduler.CosineAnnealingLR(opt_disc, T_max=1000)
        return [opt_encoder, opt_vae, opt_disc]  # , [sch_enc, sch_vae, sch_disc]

    def optimizer_zero_grad(
            self, epoch: int, batch_idx: int, optimizer: Union[optim.Optimizer, List], optimizer_idx: int):
        for opt in optimizer:  # [0]:
            opt.zero_grad()


# class ResBlockConv(nn.Module):
#     def __init__(self, dolgok):
#         super(ResBlockConv, self).__init__()
#         self.conv1 = nn.Conv3d
#         self.conv2 = nn.Conv3d
#         self.conv3 = nn.Conv3d
#         self

class Block(nn.Module):
    expansion = 1

    def __init__(self, in_channels, out_channels, i_downsample=None, stride=1):
        super(Block, self).__init__()

        self.conv1 = nn.Conv3d(in_channels, out_channels, kernel_size=3, padding=1, stride=stride, bias=False)
        self.batch_norm1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv3d(out_channels, out_channels, kernel_size=3, padding=1, stride=stride, bias=False)
        self.batch_norm2 = nn.BatchNorm3d(out_channels)

        self.i_downsample = i_downsample
        self.stride = stride
        self.relu = nn.ReLU()

    def forward(self, x):
        identity = x.clone()

        x = self.relu(self.batch_norm2(self.conv1(x)))
        x = self.batch_norm2(self.conv2(x))

        if self.i_downsample is not None:
            identity = self.i_downsample(identity)
        print(x.shape)
        print(identity.shape)
        x += identity
        x = self.relu(x)
        return x


class ResBlock(nn.Module):
    """
    Iniialize a residual block with two convolutions followed by batchnorm layers
    """

    def __init__(self, in_size: int, hidden_size: int, out_size: int):
        super().__init__()
        self.conv1 = nn.Conv3d(in_size, hidden_size, 3, padding=1)
        self.conv2 = nn.Conv3d(hidden_size, out_size, 3, padding=1)
        self.batchnorm1 = nn.BatchNorm3d(hidden_size)
        self.batchnorm2 = nn.BatchNorm3d(out_size)

    def convblock(self, x):
        x = F.relu(self.batchnorm1(self.conv1(x)))
        x = F.relu(self.batchnorm2(self.conv2(x)))
        print('x: ',x.shape)
        return x

    """
    Combine output with the original input
    """

    def forward(self, x): return x + self.convblock(x)  # skip connection




if __name__ == "__main__":
    block = ResBlock(1,3,5)
    t = torch.ones((10,1,10,10,10))
    t2 = torch.ones((10,3,10,10,10))
    # print(t+t2)
    print(block(t).shape)
