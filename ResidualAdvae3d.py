import torch
import torch.nn as nn
from BPtools.core.bpmodule import *
from BPtools.utils.models import EncoderBN, VarDecoderConv1d_3
from BPtools.trainer.bptrainer import BPTrainer

from MyResNet import *
from grid_3D import *
from model import Discriminator2D
from data_moduls import *
from focal_loss import *
from data_moduls import DummyPredictionDataModul
# from torchvision.utils import make_grid
from torchvision.transforms import ToTensor
from BPtools.utils.trajectory_plot import trajs_to_img_2, traj_to_img, trajs_to_img


encoder = MyResNet(MyResBlock, mode=2, type="encoder")
decoder = MyResNet(MyResBlock, mode=2, type="decoder")
discriminator = Discriminator2D()

model = ADVAE3D(encoder=encoder, decoder=decoder, discriminator= discriminator, loss=nn.BCELoss(), decay=0.2)  # FocalLossBinary(1, alpha=0.75)
g = torch.Generator(device="cuda")
g.manual_seed(420)
dm = Grid3DSelected("C:/Users/oliver/PycharmProjects/full_data/otthonrol", split_ratio=0.2, batch_size=300, generator=g)
# dm.prepare_data()
# dm.setup()
trainer = BPTrainer(epochs=1000, name="residual_proba_decay02")
trainer.fit(model=model, datamodule=dm)
# todo: Focal loss with Sigmoid
#   ez kész, demeglepő módon gyengébb

# todo: csak a fehér pixelekre (1) adni neki boostot -> hülyeség
