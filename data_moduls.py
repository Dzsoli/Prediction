from BPtools.core.bpdatamodule import *
import numpy as np
import torch

from BPtools.utils.trajectory_plot import trajs_to_img_2


class Grid3D_DataModul(BPDataModule):
    def __init__(self, path, split_ratio, first=11, last=21, shuffle=True):
        super(Grid3D_DataModul).__init__()
        self.path = path
        self.seq_length = None
        self.feature_dim = None
        self.data = None
        self.split_ratio = split_ratio
        self.ngsim_train = None
        self.ngsim_test = None
        self.ngsim_val = None
        self.shuffle = shuffle
        self.range = range(first, last + 1)
        # todo: batch size, and to BPDataModule too

    def prepare_data(self, auto=True):
        data = []

        for i in self.range:
            data.append(np.load(self.path + str(i) + ".npy", allow_pickle=True))

        data = np.concatenate(data)
        # del data
        print(data.shape)
        # data = np.concatenate(data)
        # print(data.shape)
        # print(data.dtype)
        temp = None
        # i=0
        for datum in data:
            # print(len(datum))
            datum = np.array(datum)[::5]
            N = datum.shape[0] // 12 * 12
            datum = np.expand_dims(datum[0:N, 7:23, 63:191], axis=1).reshape((N // 12, 12, 1, 16, 128)).transpose(0, 2,
                                                                                                                  3, 4,
                                                                                                                  1)
            # print(datum.shape)
            # print(i)
            # i+=1
            # x=datum[0, 0, :, :, 0]
            if temp is None:
                temp = datum
            else:
                temp = np.concatenate((temp, datum), axis=0)
        self.data = temp
        # self.data = np.expand_dims(np.concatenate(data), axis=1)
        # print(len(data[1]))
        # self.data = self.data[:, :, 7:23, 63:191]
        print(self.data.shape)
        print(self.data.dtype)
        self.set_has_prepared_data(True)

    def setup(self, stage: Optional[str] = None):
        N = self.data.shape[0]
        T = int(self.split_ratio * N)

        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.ngsim_train = torch.tensor(self.data[T + 1:N]).float()
        self.ngsim_val = torch.tensor(self.data[0:T]).float()
        if self.shuffle:
            self.ngsim_train = self.ngsim_train[torch.randperm(self.ngsim_train.shape[0])].to(device)
            self.ngsim_val = self.ngsim_val[torch.randperm(self.ngsim_val.shape[0])].to(device)
        else:
            self.ngsim_train = self.ngsim_train.to(device)
            self.ngsim_val = self.ngsim_val.to(device)

        self.ngsim_val = torch.split(self.ngsim_val, 500)
        self.ngsim_train = torch.split(self.ngsim_train, 500)
        self.set_has_setup_test(True)
        self.set_has_setup_fit(True)
        self.data = None

    def train_dataloader(self, *args, **kwargs):
        # return DataLoader(self.ngsim_train, batch_size=self.ngsim_train.shape[0])
        return self.ngsim_train

    def val_dataloader(self, *args, **kwargs):
        return self.ngsim_val

    def test_dataloader(self, *args, **kwargs):
        return self.ngsim_test


class DummyPredictionDataModul(BPDataModule):
    def __init__(self, path, split_ratio, batch_size=1560, shuffle=True):
        super(DummyPredictionDataModul, self).__init__()
        self.path = path
        self.q = split_ratio
        self.shuffle = shuffle
        self.train = None
        self.valid = None
        self.test = None
        self.traj_1 = None
        self.traj_2 = None
        self.grids_1 = None
        self.labels = None
        self.batch_size = batch_size

    def prepare_data(self, ):
        self.traj_1 = np.load(self.path + "/trajectories1.npy", allow_pickle=True)
        self.traj_2 = np.load(self.path + "/trajectories2.npy", allow_pickle=True)
        self.grids_1 = np.load(self.path + "/grids1.npy", allow_pickle=True)
        self.labels = np.load(self.path + "/labels.npy", allow_pickle=True)
        print(self.traj_1.shape)
        self.set_has_prepared_data(True)

    def setup(self, stage: Optional[str] = None):
        feature_dim = self.traj_1.shape[2]
        seq_length = self.traj_1.shape[1]
        N = self.traj_1.shape[0]
        q = self.q
        self.traj_1 = np.transpose(self.traj_1, (0, 2, 1))
        self.traj_2 = np.transpose(self.traj_2, (0, 2, 1))

        # for traj1, traj2 in zip(self.traj_1, self.traj_2):
        #     print(traj1.shape)
        #     print(traj2.shape)
        #     trajs_to_img_2(tr1=np.transpose(traj1, (1,0)),tr2=np.transpose(traj2, (1,0)),label="valami")

        self.grids_1 = np.expand_dims(self.grids_1, axis=1).transpose((0, 1, 3, 4, 2))
        print(self.grids_1.shape)
        print(self.traj_2.shape)

        self.traj_1[:, 1, :] = 0.05 * self.traj_1[:, 1, :]
        self.traj_2[:, 1, :] = 0.05 * self.traj_2[:, 1, :]

        self.traj_2 = self.traj_2 - self.traj_1[:, :, -1][:, :, None]
        self.traj_1 = self.traj_1 - self.traj_1[:, :, 0][:, :, None]

        if self.shuffle:
            randomperm = torch.randperm(self.traj_1.shape[0])
            self.traj_1 = self.traj_1[randomperm]
            self.traj_2 = self.traj_2[randomperm]
            self.grids_1 = self.grids_1[randomperm]

        print(self.traj_1.dtype)
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.traj_1 = torch.split(torch.tensor(self.traj_1.astype(np.float)).float().to(device), int((1 - q) * N))
        self.traj_2 = torch.split(torch.tensor(self.traj_2.astype(np.float)).float().to(device), int((1 - q) * N))
        self.grids_1 = torch.split(torch.tensor(self.grids_1).float().to(device), int((1 - q) * N))
        # self.labels =

        keys = ["traj1", "traj2", "grid2"]

        self.train = [torch.split(self.traj_1[0], self.batch_size),
                      torch.split(self.traj_2[0], self.batch_size),
                      torch.split(self.grids_1[0], self.batch_size)]

        self.valid = [torch.split(self.traj_1[1], self.batch_size),
                      torch.split(self.traj_2[1], self.batch_size),
                      torch.split(self.grids_1[1], self.batch_size)]
        # print(self.train.keys(), self.train["grid1"][0].shape)

    def train_dataloader(self, *args, **kwargs):
        return self.train

    def val_dataloader(self, *args, **kwargs):
        return self.valid

    def test_dataloader(self, *args, **kwargs):
        return self.test


class RecurrentPredictionDataModul(BPDataModule):
    def __init__(self, path, split_ratio, batch_size=1560, shuffle=True):
        super(RecurrentPredictionDataModul,self).__init__()
        self.path = path
        self.q = split_ratio
        self.shuffle = shuffle
        self.train = None
        self.valid = None
        self.test = None
        self.traj_1 = None
        self.traj_2 = None
        self.grids_1 = None
        self.labels = None
        self.batch_size = batch_size

    def prepare_data(self, ):
        self.traj_1 = np.load(self.path + "/trajectories1.npy", allow_pickle=True)
        self.traj_2 = np.load(self.path + "/trajectories2.npy", allow_pickle=True)
        self.grids_1 = np.load(self.path + "/grids1.npy", allow_pickle=True)
        self.labels = np.load(self.path + "/labels.npy", allow_pickle=True)
        print(self.traj_1.shape)
        self.set_has_prepared_data(True)

    def setup(self, stage: Optional[str] = None):
        feature_dim = self.traj_1.shape[2]
        seq_length = self.traj_1.shape[1]
        N = self.traj_1.shape[0]
        q = self.q
        self.traj_1 = np.transpose(self.traj_1, (0, 2, 1))
        self.traj_2 = np.transpose(self.traj_2, (0, 2, 1))

        # for traj1, traj2 in zip(self.traj_1, self.traj_2):
        #     print(traj1.shape)
        #     print(traj2.shape)
        #     trajs_to_img_2(tr1=np.transpose(traj1, (1,0)),tr2=np.transpose(traj2, (1,0)),label="valami")

        self.grids_1 = np.expand_dims(self.grids_1, axis=1).transpose((0, 1, 3, 4, 2))
        print(self.grids_1.shape)
        print(self.traj_2.shape)

        self.traj_1[:, 1, :] = 0.05 * self.traj_1[:, 1, :]
        self.traj_2[:, 1, :] = 0.05 * self.traj_2[:, 1, :]

        # self.traj_2 = self.traj_2 - self.traj_1[:, :, -1][:, :, None]
        self.traj_2 = self.traj_2 - self.traj_1[:, :, 0][:, :, None]
        self.traj_1 = self.traj_1 - self.traj_1[:, :, 0][:, :, None]

        if self.shuffle:
            randomperm = torch.randperm(self.traj_1.shape[0])
            self.traj_1 = self.traj_1[randomperm]
            self.traj_2 = self.traj_2[randomperm]
            self.grids_1 = self.grids_1[randomperm]

        print(self.traj_1.dtype)
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.traj_1 = torch.split(torch.tensor(self.traj_1.astype(np.float)).float(), int((1 - q) * N))  # .float().to(device)
        self.traj_2 = torch.split(torch.tensor(self.traj_2.astype(np.float)).float(), int((1 - q) * N))
        self.grids_1 = torch.split(torch.tensor(self.grids_1).float(), int((1 - q) * N))
        # self.labels =

        keys = ["traj1", "traj2", "grid2"]

        self.train = [torch.split(self.traj_1[0], self.batch_size),
                      torch.split(self.traj_2[0], self.batch_size),
                      torch.split(self.grids_1[0], self.batch_size)]

        self.valid = [torch.split(self.traj_1[1], self.batch_size),
                      torch.split(self.traj_2[1], self.batch_size),
                      torch.split(self.grids_1[1], self.batch_size)]
        # print(self.train.keys(), self.train["grid1"][0].shape)

    def train_dataloader(self, *args, **kwargs):
        return self.train

    def val_dataloader(self, *args, **kwargs):
        return self.valid

    def test_dataloader(self, *args, **kwargs):
        return self.test


class RecurrentManeuverDataModul(BPDataModule):
    def __init__(self, path, split_ratio, batch_size=1560, shuffle=True, dsampling=None, resnet18=False):
        super(RecurrentManeuverDataModul, self).__init__()
        self.path = path
        self.q = split_ratio
        self.shuffle = shuffle
        self.train = None
        self.valid = None
        self.test = None
        self.traj_1 = None
        self.grids_1 = None
        self.labels = None
        self.batch_size = batch_size
        self.dsampling = dsampling
        self.res18 = resnet18

    def prepare_data(self, ):

        self.traj_1 = np.load(self.path + "/trajectories1.npy", allow_pickle=True)
        self.grids_1 = np.load(self.path + "/grids1.npy", allow_pickle=True)
        self.labels = np.load(self.path + "/labels.npy", allow_pickle=True)

        # self.traj_1 = np.load(self.path + "/trajectories1_pred15.npy", allow_pickle=True)
        # self.grids_1 = np.load(self.path + "/grids1_pred15.npy", allow_pickle=True)
        # self.labels = np.load(self.path + "/labels.npy", allow_pickle=True)

        print(self.traj_1.shape)
        self.set_has_prepared_data(True)

    def setup(self, stage: Optional[str] = None):
        feature_dim = self.traj_1.shape[2]
        seq_length = self.traj_1.shape[1]
        N = self.traj_1.shape[0]
        q = self.q
        self.traj_1 = np.transpose(self.traj_1, (0, 2, 1))
        # self.traj_2 = np.transpose(self.traj_2, (0, 2, 1))

        # for traj1, traj2 in zip(self.traj_1, self.traj_2):
        #     print(traj1.shape)
        #     print(traj2.shape)
        #     trajs_to_img_2(tr1=np.transpose(traj1, (1,0)),tr2=np.transpose(traj2, (1,0)),label="valami")

        self.grids_1 = np.expand_dims(self.grids_1, axis=1).transpose((0, 1, 3, 4, 2))
        if self.dsampling is not None:
            self.grids_1 = self.grids_1[:,:,:,:,0::self.dsampling]

        # Resnet18-hoz 3 csatornás
        if self.res18:
            self.grids_1 = self.grids_1.repeat(3,1)
        # else:
        #     self.grids_1 = self.grids_1[:, :, :, :, 0::2]
        print(self.grids_1.shape)
        # print(self.traj_2.shape)

        self.traj_1[:, 1, :] = 0.05 * self.traj_1[:, 1, :]

        self.traj_1 = self.traj_1 - self.traj_1[:, :, 0][:, :, None]

        if self.shuffle:
            randomperm = torch.randperm(self.traj_1.shape[0])
            self.traj_1 = self.traj_1[randomperm]
            self.grids_1 = self.grids_1[randomperm]
            self.labels = self.labels[randomperm]

        print(self.traj_1.dtype)
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.traj_1 = torch.split(torch.tensor(self.traj_1.astype(np.float)).float(), int((1 - q) * N))  # .float().to(device)
        self.grids_1 = torch.split(torch.tensor(self.grids_1).float(), int((1 - q) * N))
        self.labels = torch.split(torch.tensor(self.labels).long(), int((1 - q) * N))

        keys = ["traj1", "traj2", "grid2"]

        self.train = [torch.split(self.traj_1[0], self.batch_size),
                      torch.split(self.grids_1[0], self.batch_size),
                      torch.split(self.labels[0], self.batch_size)]

        self.valid = [torch.split(self.traj_1[1], self.batch_size),
                      torch.split(self.grids_1[1], self.batch_size),
                      torch.split(self.labels[1], self.batch_size)]
        # print(self.train.keys(), self.train["grid1"][0].shape)

    def train_dataloader(self, *args, **kwargs):
        return self.train

    def val_dataloader(self, *args, **kwargs):
        return self.valid

    def test_dataloader(self, *args, **kwargs):
        return self.test


class TrajectoryPredData(BPDataModule):
    def __init__(self, path, split_ratio, batch_size=256, shuffle=True, pred=60, is_grid=False):
        super(TrajectoryPredData, self).__init__()
        self.path = path
        self.q = split_ratio
        self.shuffle = shuffle
        self.train = None
        self.valid = None
        self.test = None
        self.traj_1 = None
        self.traj_2 = None
        self.pred = pred
        self.is_grid = is_grid
        if is_grid:
            self.grids_1 = None
        self.labels = None
        self.batch_size = batch_size

    def prepare_data(self, ):
        self.traj_1 = np.load(self.path + "/trajectories1.npy", allow_pickle=True)
        self.traj_2 = np.load(self.path + "/trajectories2.npy", allow_pickle=True)
        self.grids_1 = np.load(self.path + "/grids1.npy", allow_pickle=True)
        # self.labels = np.load(self.path + "/labels.npy", allow_pickle=True)
        self.labels = np.load(self.path + "/labelhat.npy", allow_pickle=True)


        # self.traj_1 = np.load(self.path + "/trajectories1_pred15.npy", allow_pickle=True)
        # self.grids_1 = np.load(self.path + "/grids1_pred15.npy", allow_pickle=True)
        # self.labels = np.load(self.path + "/labels.npy", allow_pickle=True)
        print(self.traj_1.shape)
        self.set_has_prepared_data(True)

    def setup(self, stage: Optional[str] = None):
        feature_dim = self.traj_1.shape[2]
        seq_length = self.traj_1.shape[1]
        N = self.traj_1.shape[0]
        q = self.q
        self.traj_1 = np.transpose(self.traj_1, (0, 2, 1))
        self.traj_2 = np.transpose(self.traj_2, (0, 2, 1))
        self.traj_2 = self.traj_2[:,:,0:self.pred]

        self.traj_1[:, 1, :] = 0.05 * self.traj_1[:, 1, :]
        self.traj_2[:, 1, :] = 0.05 * self.traj_2[:, 1, :]

        self.traj_2 = self.traj_2 - self.traj_1[:, :, -1][:, :, None]
        self.traj_1 = self.traj_1 - self.traj_1[:, :, 0][:, :, None]

        # label_hat maximuma
        arr = np.zeros_like(self.labels)
        arr[self.labels.argmax(axis=1)[:, None] == range(self.labels.shape[1])] = 1
        self.labels = arr

        self.grids_1 = np.expand_dims(self.grids_1, axis=1).transpose((0, 1, 3, 4, 2))

        if self.shuffle:
            randomperm = torch.randperm(self.traj_1.shape[0])
            self.traj_1 = self.traj_1[randomperm]
            self.traj_2 = self.traj_2[randomperm]
            self.grids_1 = self.grids_1[randomperm]
            self.labels = self.labels[randomperm]

        print(self.traj_1.dtype)
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.traj_1 = torch.split(torch.tensor(self.traj_1.astype(np.float)).float(), int((1 - q) * N))  # .float().to(device)
        self.traj_2 = torch.split(torch.tensor(self.traj_2.astype(np.float)).float(), int((1 - q) * N))
        self.grids_1 = torch.split(torch.tensor(self.grids_1).float(), int((1 - q) * N))
        self.labels = torch.split(torch.tensor(self.labels).long(), int((1 - q) * N))

        keys = ["traj1", "traj2", "grid2"]

        self.train = [torch.split(self.traj_1[0], self.batch_size),
                      torch.split(self.traj_2[0], self.batch_size),
                      torch.split(self.grids_1[0], self.batch_size),
                      torch.split(self.labels[0], self.batch_size)]

        self.valid = [torch.split(self.traj_1[1], self.batch_size),
                      torch.split(self.traj_2[1], self.batch_size),
                      torch.split(self.grids_1[1], self.batch_size),
                      torch.split(self.labels[1], self.batch_size)]

    def train_dataloader(self, *args, **kwargs):
        return self.train

    def val_dataloader(self, *args, **kwargs):
        return self.valid

    def test_dataloader(self, *args, **kwargs):
        return self.test


class Grid3DSelected(BPDataModule):
    def __init__(self, path, split_ratio, batch_size=1560, shuffle=True, generator=None, dsampling=False):
        super(Grid3DSelected, self).__init__()
        self.path = path
        self.q = split_ratio
        self.shuffle = shuffle
        self.train = None
        self.valid = None
        self.test = None
        self.traj_1 = None
        self.grids_1 = None
        self.labels = None
        self.generator = generator
        self.batch_size = batch_size
        self.dsampling = dsampling

    def prepare_data(self, ):
        # self.traj_1 = np.load(self.path + "/trajectories1.npy", allow_pickle=True)
        self.grids_1 = np.load(self.path + "/grids1.npy", allow_pickle=True)
        # self.labels = np.load(self.path + "/labels.npy", allow_pickle=True)
        # print(self.traj_1.shape)
        self.set_has_prepared_data(True)

    def setup(self, stage: Optional[str] = None):
        # feature_dim = self.traj_1.shape[2]
        # seq_length = self.traj_1.shape[1]
        N = self.grids_1.shape[0]
        q = self.q
        # self.traj_1 = np.transpose(self.traj_1, (0, 2, 1))
        # self.traj_2 = np.transpose(self.traj_2, (0, 2, 1))

        # for traj1, traj2 in zip(self.traj_1, self.traj_2):
        #     print(traj1.shape)
        #     print(traj2.shape)
        #     trajs_to_img_2(tr1=np.transpose(traj1, (1,0)),tr2=np.transpose(traj2, (1,0)),label="valami")

        self.grids_1 = np.expand_dims(self.grids_1, axis=1).transpose((0, 1, 3, 4, 2))
        if self.dsampling:
            self.grids_1 = self.grids_1[:,:,:,:,0::5]
        else:
            self.grids_1 = self.grids_1[:,:,:,:,0::2]
        print(self.grids_1.shape)
        # print(self.traj_2.shape)

        # self.traj_1[:, 1, :] = 0.05 * self.traj_1[:, 1, :]

        # self.traj_1 = self.traj_1 - self.traj_1[:, :, 0][:, :, None]

        if self.shuffle:
            randomperm = torch.randperm(self.grids_1.shape[0], generator=torch.Generator().manual_seed(420))
            # self.traj_1 = self.traj_1[randomperm]
            self.grids_1 = self.grids_1[randomperm]
            # self.labels = self.labels[randomperm]

        # print(self.traj_1.dtype)
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        # self.traj_1 = torch.split(torch.tensor(self.traj_1.astype(np.float)).float(), int((1 - q) * N))  # .float().to(device)
        self.grids_1 = torch.split(torch.tensor(self.grids_1).float(), int((1 - q) * N))
        # self.labels = torch.split(torch.tensor(self.labels).long(), int((1 - q) * N))

        keys = ["traj1", "traj2", "grid2"]

        self.train = torch.split(self.grids_1[0], self.batch_size)

        self.valid = torch.split(self.grids_1[1], self.batch_size)
        # print(self.train.keys(), self.train["grid1"][0].shape)

    def train_dataloader(self, *args, **kwargs):
        return self.train

    def val_dataloader(self, *args, **kwargs):
        return self.valid

    def test_dataloader(self, *args, **kwargs):
        return self.test
