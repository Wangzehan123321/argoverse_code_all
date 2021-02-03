# import numpy as np
# a=np.array([[1,2,3],[4,5,6],[3,4,1]])
# print(a[np.argsort(a[:,0]),:])

# import pickle as pkl
# import numpy as np
# with open("../data/forecasting_features_val.pkl","rb")as f:
#     data=pkl.load(f)
# print(data["FEATURES"].values)

# from shapely.geometry import LinearRing, LineString, Point, Polygon
# import numpy as np
# line1=np.array([[0,0],[5,5]])
# line1=LineString(line1)
# point1=line1.interpolate(5)
# print(point1.coords[0])




from torch.utils.data import Dataset, DataLoader
import pickle as pkl
import numpy as np
import torch
import random

# seq_list=[[3],[4],[5]]
# print(np.cumsum(np.asarray(seq_list)).tolist())
#
# with open("../data/data_nerb/forecasting_features_sample.pkl","rb")as f:
#     data=pkl.load(f)
# print((data["NEIGHBOUR"].values)[0]["neighbour_pt"][0].shape)

# ___________________________________________________________________________________________________________________________

FEATURE_FORMAT = {
    "TIMESTAMP": 0,
    "TRACK_ID": 1,
    "OBJECT_TYPE": 2,
    "X": 3,
    "Y": 4,
    "CITY_NAME": 5,
    "MIN_DISTANCE_FRONT": 6,
    "MIN_DISTANCE_BACK": 7,
    "NUM_NEIGHBORS": 8,
    "OFFSET_FROM_CENTERLINE": 9,
    "DISTANCE_ALONG_CENTERLINE": 10,
}


## Dataset class for the NGSIM dataset
class ArgoDataset(Dataset):

    def __init__(self, pkl_file, t_h=20, t_f=30, d_s=1, enc_size=64):
        with open(pkl_file,"rb")as f:
            raw_file=pkl.load(f)
        self.traj = raw_file["FEATURES"].values
        self.nerb = raw_file["NEIGHBOUR"].values
        self.t_h = t_h  # length of track history
        self.t_f = t_f  # length of predicted trajectory
        self.d_s = d_s  # down sampling rate of all sequences
        self.enc_size = enc_size  # size of encoder LSTM

    def __len__(self):
        return len(self.traj)

    def __getitem__(self, idx):
        traj = self.traj[idx]
        nerb = self.nerb[idx]
        #得到预测目标的历史轨迹和未来轨迹
        hist = traj[0:self.t_h:self.d_s,-2:]
        fut = traj[self.t_h::self.d_s,-2:]
        #将沿着车道线的坐标变化为相对坐标
        current_y_refer = hist[-1,1]
        hist[:,1]=hist[:,1]-current_y_refer
        fut[:,1]=fut[:,1]-current_y_refer

        #得到预测目标当前时刻的世界坐标系下的坐标
        current_pt=traj[self.t_h-1,3:5]

        hist_traj_all=[hist]
        current_pt_all=[current_pt]
        #添加周围车辆的历史轨迹和世界坐标，用于网络中的交互模块处理交互关系
        for i in range(len(nerb["neighbour_traj"])):
            nbr_traj = nerb["neighbour_traj"][i]
            # 将周围车辆的轨迹沿着车道线的坐标变化为相对坐标
            current_nbr_pt=nbr_traj[-1,1]
            nbr_traj[:,1]=nbr_traj[:,1]-current_nbr_pt
            hist_traj_all.append(nbr_traj)

            nbr_pt = nerb["neighbour_pt"][i]
            current_pt_all.append(nbr_pt)
        try:
            assert len(hist_traj_all)==len(current_pt_all)
        except:
            raise AssertionError

        return hist_traj_all, current_pt_all, fut

    ## Collate function for dataloader
    def collate_fn(self, samples):

        seq_batch_size = 0
        seq_list = []
        for hist_traj_all, _, _ in samples:
            seq_batch_size+=len(hist_traj_all)
            seq_list.append(len(hist_traj_all))

        #将samples数据整合到batch维度上（sequence,batch,2）
        hist_traj_batch = torch.zeros(self.t_h, seq_batch_size, 2)
        current_pt_batch = torch.zeros(seq_batch_size,2)
        fut_batch = torch.zeros(self.t_f, len(samples), 2)
        cum_start_idx = [0] + np.cumsum(np.asarray(seq_list)).tolist()

        try:
            assert seq_batch_size==cum_start_idx[-1]
        except:
            raise AssertionError

        for sampleId, (hist_traj_all, current_pt_all, fut) in enumerate(samples):

            hist_traj_sample=np.stack(hist_traj_all,1)
            current_pt_sample=np.stack(current_pt_all,0)
            hist_traj_batch[:, cum_start_idx[sampleId]:cum_start_idx[sampleId+1], :] = torch.from_numpy(hist_traj_sample[:,:,:].astype("float"))
            current_pt_batch[cum_start_idx[sampleId]:cum_start_idx[sampleId+1], :] = torch.from_numpy(current_pt_sample[:,:].astype("float"))
            fut_batch[:, sampleId, :] = torch.from_numpy(fut[:, :].astype("float"))

        return hist_traj_batch,current_pt_batch,fut_batch,cum_start_idx


# trdataset=ArgoDataset(pkl_file="../data/data_nerb/forecasting_features_sample.pkl")
# print(len(trdataset))
#hist_traj_all, current_pt_all, fut=list(trdataset)[0]
# trdataloader=DataLoader(trdataset,batch_size=64,shuffle=True,num_workers=8,collate_fn=trdataset.collate_fn)
#
#
#
# hist_traj_batch,current_pt_batch,fut_batch,cum_start_idx=next(iter(trdataloader))
#
# print(hist_traj_batch.shape)
# print(current_pt_batch.shape)
# print(fut_batch.shape)
# print(cum_start_idx)
import os
root="/home/wangzehan/argoverse-api/data/train/data"
print(len(os.listdir(root)))
