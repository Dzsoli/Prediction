import torch.nn as nn


class Discriminator2D_Latent3_v3(nn.Module):
    def __init__(self):
        super(Discriminator2D_Latent3_v3, self).__init__()

        self.linear = nn.Sequential(
            nn.Linear(64, 32),
            nn.BatchNorm1d(32),
            nn.LeakyReLU(0.2),

            nn.Linear(32, 16),
            nn.BatchNorm1d(16),
            nn.LeakyReLU(0.2),

            nn.Linear(16, 16),
            nn.BatchNorm1d(16),
            nn.LeakyReLU(0.2),

            nn.Linear(16, 8),
            nn.BatchNorm1d(8),
            nn.LeakyReLU(0.2),

            nn.Linear(8, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.linear(x.view(-1, 64))


class GridEncoder(nn.Module):
    def __init__(self, kernel=2):
        super(GridEncoder, self).__init__()

        self.conv = nn.Sequential(
            # input shape: N, 1, 32, 256
            nn.Conv2d(1, 3, kernel_size=(2, 5), stride=1, padding=2),
            nn.LeakyReLU(0.2),

            nn.Conv2d(3, 5, kernel_size=(3, 8), stride=1, padding=1),
            nn.BatchNorm2d(5),
            nn.LeakyReLU(0.2),

            nn.Conv2d(5, 8, kernel_size=(3, 8), stride=1, padding=0),
            nn.BatchNorm2d(8),
            nn.LeakyReLU(0.2),

            nn.Conv2d(8, 5, kernel_size=1, padding=0),
            nn.BatchNorm2d(5),
            nn.LeakyReLU(0.2),
        )
        self.mu = nn.Sequential(

            nn.Conv2d(5, 5, kernel_size=(4, 4), stride=(1, 2), padding=(1, 2)),
            nn.BatchNorm2d(5),
            nn.LeakyReLU(0.2),

            nn.Conv2d(5, 4, kernel_size=(4, 4), stride=(2, 2), padding=(1, 2)),
            nn.LeakyReLU(0.2),

            nn.Conv2d(4, 3, kernel_size=(4, 4), stride=(2, 2), padding=(1, 2)),
            nn.LeakyReLU(0.2),

            nn.Conv2d(3, 1, kernel_size=1, padding=0)
        )
        # [N,  1, 4, 16]
        self.logvar = nn.Sequential(
            nn.Conv2d(5, 5, kernel_size=(4, 4), stride=(1, 2), padding=(1, 2)),
            nn.BatchNorm2d(5),
            nn.LeakyReLU(0.2),

            nn.Conv2d(5, 4, kernel_size=(4, 4), stride=(2, 2), padding=(1, 2)),
            nn.LeakyReLU(0.2),

            nn.Conv2d(4, 3, kernel_size=(4, 4), stride=(2, 2), padding=(1, 2)),
            nn.LeakyReLU(0.2),

            nn.Conv2d(3, 1, kernel_size=1, padding=0)
        )
        # [N,  1, 4, 16]

    def forward(self, x):
        h = self.conv(x)
        return self.mu(h), self.logvar(h)  # .squeeze(1).squeeze(1)


class Decoder2D_v3(nn.Module):
    def __init__(self, latent_dim=155):
        super(Decoder2D_v3, self).__init__()
        feature = 4
        # [N,  1, 4, 16]

        self.convtr = nn.Sequential(
            nn.ConvTranspose2d(1, feature, kernel_size=(3, 3), stride=1, padding=0),
            nn.BatchNorm2d(feature),
            nn.LeakyReLU(0.2),
            # N, 32, 4, 16

            nn.ConvTranspose2d(feature, feature, kernel_size=(3, 3), stride=(1, 1), padding=2),
            nn.BatchNorm2d(feature),
            nn.LeakyReLU(0.2),

            nn.ConvTranspose2d(feature, feature * 2, kernel_size=(4, 4), stride=(2, 2), padding=(1, 2)),
            nn.BatchNorm2d(feature * 2),
            nn.LeakyReLU(0.2),

            nn.ConvTranspose2d(feature * 2, feature * 3, kernel_size=(4, 4), stride=(1, 2), padding=(1, 2)),
            nn.BatchNorm2d(feature * 3),
            nn.LeakyReLU(0.2),

            nn.ConvTranspose2d(feature * 3, feature * 2, kernel_size=(3, 8), stride=(1, 1), padding=(1, 1)),
            nn.BatchNorm2d(feature * 2),
            nn.LeakyReLU(0.2),

            nn.ConvTranspose2d(feature * 2, feature, kernel_size=(3, 8), stride=(2, 2), padding=(1, 2),
                               output_padding=(0, 1)),
            nn.BatchNorm2d(feature),
            nn.LeakyReLU(0.2),

            nn.ConvTranspose2d(feature, 1, kernel_size=(2, 4), stride=1, padding=(1, 2)),
            # nn.AdaptiveAvgPool2d((16, 128)),
            nn.Sigmoid()
            # N, 1, 16, 128

        )

    def forward(self, l):
        # print(l.shape)
        return self.convtr(l)