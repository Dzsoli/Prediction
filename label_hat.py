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
from D3_maneuver_onlygrid import Prediction_maneuver_grid3d


model_path = "C:\\repos\\Prediction\\log_3d_Taylor18_NM_split_AMSGrad_tovabb_lr01_w0\\model_state_dict_3d_Taylor18_NM_split_AMSGrad_tovabb_lr01_w0"
taylor = TaylorNet_3D(BasicQuadBlock_3D, [2, 2, 2, 2], num_classes=3, order=2, partial_mix=7)
model = Prediction_maneuver_grid3d(taylor, merge_z=None, loss=FocalLossMulty([0.178, 0.042, 0.78], 5))
model.load_state_dict(torch.load(model_path))
model = model.cuda()
grids1 = np.load("D:/dataset/grids1.npy", allow_pickle=True)
grids1 = np.expand_dims(grids1, axis=1).transpose((0, 1, 3, 4, 2))
grids1 = grids1.repeat(3,1)
grids1 = torch.tensor(grids1).float()
# labelhat = torch.zeros((grids1.shape[0],3))
grids1 = torch.split(grids1, 8)
# labelhat = list(torch.split(labelhat, 8))
labelhat = []
sm = nn.Softmax(dim=1)
i=0
for g in grids1:
    # l = l.cuda()
    with torch.no_grad():
        a = sm(model.grid_enc(g.cuda()))
        labelhat.append(np.array(a.to("cpu")))
    # labelhat[-1].device
    # print(l)
# labelhat = torch.concat(tuple(labelhat))
labelhat = np.concatenate(labelhat)
print(labelhat)
print(labelhat.shape)
# labelhat = np.array(labelhat.to("cpu"))
np.save("labelhat", labelhat)
