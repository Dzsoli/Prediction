from BPtools.utils.vehicle import *
from typing import Any, List, Optional, Tuple, Union
import pandas as pd
# import modin.pandas as pd
# from sodapy import Socrata
from PIL import Image
from matplotlib import cm


class DataProcess:
    def __init__(self,
                 path,
                 delta_T1=60,
                 delta_T2=None,
                 grid_dT=5,
                 break_threshold=0.8,
                 overtaking=False,
                 braking=False
                 ):
        self.delta_T1 = delta_T1
        self.delta_T2 = delta_T2 if delta_T2 is not None else delta_T1
        self.grid_dT = grid_dT
        self.break_threshold = break_threshold
        self.overtaking = overtaking
        self.braking = braking
        self.path = path
        self.__mode = "ob" if overtaking and braking else "o" if overtaking and not braking else "b" if not (
                overtaking or not braking) else None
        # self.time_reg = ["31349.0", "67669.0", "75649.0"] # i-80
        # self.maxnum_reg = [21, 19, 17] # i-80
        # első 3 i-80, második 3 us-101
        self.time_reg = ["31349.0", "67669.0", "75649.0", "54169789.0", "54178420.0", "54187570.0"]
        self.maxnum_reg = [21, 19, 17, 22, 21, 20]

    @property
    def path_grids(self):
        return self.path + '/grids'

    @property
    def path_trajs(self):
        return self.path + '/trajs'

    @property
    def delta_T12(self):
        return self.delta_T1 + self.delta_T2

    def build_dataset(self):
        trajectories1 = []
        trajectories2 = []
        grids_1 = []
        grids_2 = []
        labels = []
        wrongs = 0
        for T, N in zip(self.time_reg, self.maxnum_reg):
            print(T)

            for n in range(1, N):
                if (T == "54178420.0") and ((n == 4) or (n==5)):
                    print("dik")
                    # continue
                print(n)
                try:
                    grid_Tn = np.load(self.path_grids + "/" + T + "_" + str(n) + ".npy", allow_pickle=True)
                except:
                    print("There is no input in grids")
                    continue
                trajs_Tn = np.load(self.path_trajs + "/traj" + T + "_" + str(n) + ".npy", allow_pickle=True)
                # for trajs_Tn_i in trajs_Tn:
                for i in range(len(trajs_Tn)):
                    print(i)
                    # if (T == "54178420.0") and ((n == 4) or (n == 5)):
                    #     print("dik dik")
                    #     print("traj ", trajs_Tn[i].shape)
                    #     print("grid ", len(grid_Tn[i]), grid_Tn[i][0].shape)
                    if trajs_Tn[i].shape[0] != len(grid_Tn[i]):
                        wrongs += 1
                        print(i, "Not equal")
                        continue
                    lefts = np.where(trajs_Tn[i][:, 4] == -1)[0]
                    rights = np.where(trajs_Tn[i][:, 4] == 1)[0]
                    # lefts = np.where(trajs_Tn_i[:, 4] == -1)[0]
                    # rights = np.where(trajs_Tn_i[:, 4] == 1)[0]
                    left_size = lefts.size
                    right_size = rights.size

                    if left_size + right_size == 0:
                        # lane keeping
                        if trajs_Tn[i].shape[0] > self.delta_T12:
                            index = trajs_Tn[i].shape[0] - self.delta_T12
                            trajectories1.append(trajs_Tn[i][index:index+self.delta_T1, 0:2])
                            trajectories2.append(trajs_Tn[i][index + self.delta_T1:index + self.delta_T12, 0:2])
                            gtni_1 = np.array(grid_Tn[i][index:index+self.delta_T1:self.grid_dT])[:, 7:23,63:191]
                            gtni_2 = np.array(grid_Tn[i][index + self.delta_T1:index+self.delta_T12:self.grid_dT])[:,
                                     7:23,63:191]
                            grids_1.append(gtni_1)
                            grids_2.append(gtni_2)
                            labels.append([0,1,0])
                        continue
                    if left_size + right_size == 1:
                        if left_size == 1:
                            # left change
                            # print("left", lefts[0], lefts[0] > 60, lefts[0] + 60 < trajs_Tn[i].shape[0])
                            if (lefts[0] > self.delta_T1) and (lefts[0] + self.delta_T2 < trajs_Tn[i].shape[0]):
                                trajectories1.append(trajs_Tn[i][lefts[0] - self.delta_T1:lefts[0], 0:2])
                                trajectories2.append(trajs_Tn[i][lefts[0]:lefts[0] + self.delta_T2, 0:2])
                                gtni_1 = np.array(grid_Tn[i][lefts[0] - self.delta_T1:lefts[0]:self.grid_dT])[:, 7:23,
                                         63:191]
                                gtni_2 = np.array(grid_Tn[i][lefts[0]:lefts[0] + self.delta_T2:self.grid_dT])[:, 7:23,
                                         63:191]
                                grids_1.append(gtni_1)
                                grids_2.append(gtni_2)
                                labels.append([1,0,0])

                        else:
                            # right change
                            # print("right", rights[0], rights[0] > 60, rights[0] + 60 < trajs_Tn[i].shape[0])
                            if (rights[0] > self.delta_T1) and (rights[0] + self.delta_T2 < trajs_Tn[i].shape[0]):
                                trajectories1.append(trajs_Tn[i][rights[0] - self.delta_T1:rights[0], 0:2])
                                trajectories2.append(trajs_Tn[i][rights[0]:rights[0] + self.delta_T2, 0:2])
                                gtni_1 = np.array(grid_Tn[i][rights[0] - self.delta_T1:rights[0]:self.grid_dT])[:, 7:23,
                                         63:191]
                                gtni_2 = np.array(grid_Tn[i][rights[0]:rights[0] + self.delta_T2:self.grid_dT])[:, 7:23,
                                         63:191]
                                grids_1.append(gtni_1)
                                grids_2.append(gtni_2)
                                labels.append([0,0,1])

                    if left_size + right_size == 2:
                        if left_size == 1:
                            # overtake
                            # print("overtake")
                            # print(lefts < rights)
                            pass
                        elif left_size == 2:
                            # double left change
                            # print("double left")
                            for left_n in lefts:
                                if (left_n > self.delta_T1) and (left_n + self.delta_T2 < trajs_Tn[i].shape[0]):
                                    trajectories1.append(trajs_Tn[i][left_n - self.delta_T1:left_n, 0:2])
                                    trajectories2.append(trajs_Tn[i][left_n:left_n + self.delta_T2, 0:2])
                                    gtni_1 = np.array(grid_Tn[i][left_n - self.delta_T1:left_n:self.grid_dT])[:, 7:23,
                                             63:191]
                                    gtni_2 = np.array(grid_Tn[i][left_n:left_n + self.delta_T2:self.grid_dT])[:, 7:23,
                                             63:191]
                                    grids_1.append(gtni_1)
                                    grids_2.append(gtni_2)
                                    labels.append([1,0,0])

                        else:
                            # double right change
                            # print("double right")
                            for right_n in rights:
                                if (right_n > self.delta_T1) and (right_n + self.delta_T2 < trajs_Tn[i].shape[0]):
                                    trajectories1.append(trajs_Tn[i][right_n - self.delta_T1:right_n, 0:2])
                                    trajectories2.append(trajs_Tn[i][right_n:right_n + self.delta_T2, 0:2])
                                    gtni_1 = np.array(grid_Tn[i][right_n - self.delta_T1:right_n:self.grid_dT])[:, 7:23,
                                             63:191]
                                    gtni_2 = np.array(grid_Tn[i][right_n:right_n + self.delta_T2:self.grid_dT])[:, 7:23,
                                             63:191]
                                    grids_1.append(gtni_1)
                                    grids_2.append(gtni_2)
                                    labels.append([0,0,1])

        np.save("trajectories1", np.array(trajectories1))
        np.save("trajectories2", np.array(trajectories2))
        np.save("grids1", np.array(grids_1))
        np.save("grids2", np.array(grids_2))
        np.save("labels", np.array(labels))
        print("Not equal samples: ", wrongs)


if __name__ == "__main__":
    # data = DataProcess(path='D:/dataset/', grid_dT=1)
    data = DataProcess(path='../../../dataset')
    # print(data.path_grids, data.path_trajs)
    data.build_dataset()
    labels = np.load("labels.npy")
    print("keep", np.sum(np.prod((labels==np.array([0,1,0])), axis=1)))
    print("left", np.sum(np.prod((labels==np.array([1,0,0])), axis=1)))
    print("right", np.sum(np.prod((labels==np.array([0,0,1])), axis=1)))
