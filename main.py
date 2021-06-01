from BPtools.trainer.bptrainer import BPTrainer
from BPtools.utils.models import *
from BPtools.metrics.criterions import KLD_BCE_loss_2Dvae
from model import *
from data_moduls import *
from pred_3D import *


# Adatok az OccupancyGrid tanuláshoz
# dm = GridVAE_DataModul(path='D:/dataset/grids/31349.0_11.npy', split_ratio=0.2)
# dm.prepare_data()
dm = Grid3D_DataModul(path='../BPtools/BPtools/utils//31349.0_', first=1, last=13, split_ratio=0.2)
enc = Encoder_Grid3D_3()
dec = Decoder_Grid3D_3()
disc = Discriminator2D()
aae3d = ADVAE3D(encoder=enc, decoder=dec, discriminator=disc)

trainer = BPTrainer(epochs=20000, name='3D_pred_proba_img_type3_50_1_13')
trainer.fit(model=aae3d, datamodule=dm)
