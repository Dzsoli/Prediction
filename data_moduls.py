from BPtools.core.bpdatamodule import *
import numpy as np
import torch


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
        self.range = range(first, last+1)
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
            N = datum.shape[0]//12*12
            datum = np.expand_dims(datum[0:N, 7:23, 63:191], axis=1).reshape((N//12,12,1,16,128)).transpose(0,2,3,4,1)
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
        self.ngsim_train = torch.tensor(self.data[T+1:N]).float()
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
