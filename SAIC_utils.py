from torch.utils.data import Dataset,DataLoader
import pandas as pd
import pickle as pkl
import numpy as np
import torch
from shapely.geometry import LinearRing, LineString, Point, Polygon
import random

## Dataset class for the NGSIM dataset
class SAICDataset(Dataset):

    def __init__(self, mat_file, t_h=20, t_f=50, d_s=1, enc_size = 64, data_argument=True):
        with open(mat_file,"rb")as f:
            raw_file=pkl.load(f)
        self.D = raw_file['train_data']
        self.T = raw_file['track']
        self.C = raw_file["centerline"]
        self.t_h = t_h  # length of track history
        self.t_f = t_f  # length of predicted trajectory
        self.d_s = d_s  # down sampling rate of all sequences
        self.enc_size = enc_size # size of encoder LSTM
        self.data_len = len(self.D)
        self.data_argument=data_argument

    def __len__(self):
        return self.data_len

    def __getitem__(self, idx):#训练数据集的形式：list_csv_index,vehicle_id,timestamp,x,y,centerline_id
        while True:
            csv_Id = self.D[idx, 0].astype(int)
            veh_Id = self.D[idx, 1].astype(int)
            t = self.D[idx, 2]
            centerline=self.D[idx,5].astype(int)

            # Get track history 'hist' = ndarray, and future track 'fut' = ndarray
            # hist,tang_dist_current = self.getHistory(csv_Id,veh_Id,centerline,t)
            # fut = self.getFuture(csv_Id,veh_Id,centerline,t,tang_dist_current)
            hist,current_point = self.getHistory(csv_Id, veh_Id, centerline, t)
            fut = self.getFuture(csv_Id,veh_Id,centerline,t,current_point)

            if self.data_argument:
                rightleft = random.random()
                if rightleft >= 0.5:
                    hist_out = np.full((hist.shape[0], hist.shape[1]), 0, dtype=float)
                    hist_out[:, 0] = hist[:, 0]
                    hist_out[:, 1] = -hist[:, 1]
                    fut_out = np.full((fut.shape[0], fut.shape[1]), 0, dtype=float)
                    fut_out[:, 0] = fut[:, 0]
                    fut_out[:, 1] = -fut[:, 1]
                else:
                    hist_out=hist
                    fut_out=fut
            else:
                hist_out = hist
                fut_out = fut
            if hist_out[0,0]>0:
                hist_out[:,0]=-hist_out[:,0]
                fut_out[:,0]=-fut_out[:,0]
            return hist_out,fut_out
            # if self.data_argument:#两个数据增强的方式（1、左右翻转。2、向上运动和向下运动）
            #     rightleft=random.random()
            #     updown=random.random()
            #     if rightleft>=0.5 and updown>=0.5:
            #         hist_out=np.full((hist.shape[0],hist.shape[1]),0,dtype=float)
            #         hist_out[:,0]=-hist[:,0]
            #         hist_out[:,1]=-hist[:,1]
            #         fut_out=np.full((fut.shape[0],fut.shape[1]),0,dtype=float)
            #         fut_out[:,0]=-fut[:,0]
            #         fut_out[:,1]=-fut[:,1]
            #     elif updown>=0.5:
            #         hist_out = np.full((hist.shape[0], hist.shape[1]), 0, dtype=float)
            #         hist_out[:, 0] = -hist[:, 0]
            #         hist_out[:, 1] = hist[:, 1]
            #         fut_out = np.full((fut.shape[0], fut.shape[1]), 0, dtype=float)
            #         fut_out[:, 0] = -fut[:, 0]
            #         fut_out[:, 1] = fut[:, 1]
            #     elif rightleft>=0.5:
            #         hist_out = np.full((hist.shape[0], hist.shape[1]), 0, dtype=float)
            #         hist_out[:, 0] = hist[:, 0]
            #         hist_out[:, 1] = -hist[:, 1]
            #         fut_out = np.full((fut.shape[0], fut.shape[1]), 0, dtype=float)
            #         fut_out[:, 0] = fut[:, 0]
            #         fut_out[:, 1] = -fut[:, 1]
            #     else:
            #         hist_out=hist
            #         fut_out=fut
            # else:
            #     hist_out = hist
            #     fut_out = fut
            # return hist_out,fut_out

    def getHistory(self, csv_Id, veh_Id, centerline, t):
        centerline=self.C[csv_Id,centerline]
        track=self.T[csv_Id,veh_Id]
        stpt = np.argwhere(track[:,2] == t).item()-self.t_h
        assert stpt>=0
        enpt = np.argwhere(track[:,2] == t).item()
        hist_track=track[stpt:enpt:self.d_s,0:2]
        current_point=hist_track[-1,:]
        return hist_track-current_point,current_point

    def getFuture(self, csv_Id, veh_Id, centerline, t,current_point):
        centerline = self.C[csv_Id, centerline]
        track = self.T[csv_Id, veh_Id]
        stpt = np.argwhere(track[:, 2] == t).item()+ self.d_s
        enpt =  np.minimum(len(track),np.argwhere(track[:, 2] == t).item() + self.t_f + 1)#这里在数据处理的时候还要设置当前时刻不能为序列最后时刻
        fut_track = track[stpt:enpt:self.d_s, 0:2]
        return fut_track-current_point

    # ## Helper function to get track history
    # def getHistory(self,csv_Id,veh_Id,centerline,t):
    #     centerline=self.C[csv_Id,centerline]
    #     track=self.T[csv_Id,veh_Id]
    #     stpt = np.argwhere(track[:,2] == t).item()-self.t_h
    #     assert stpt>=0
    #     enpt = np.argwhere(track[:,2] == t).item()
    #     hist_track=track[stpt:enpt:self.d_s,0:2]
    #     hist_track_rel=np.full((hist_track.shape[0],2),0,dtype=float)
    #     center_st=centerline[0,:]
    #     center_ed=centerline[-1,:]
    #     #需要填充centerline不够长的情况
    #     centerline_ls = LineString(centerline)
    #     delta=0.01
    #     for seq in range(hist_track.shape[0]):
    #         point=Point(hist_track[seq,:])
    #         tang_dist = centerline_ls.project(point)#得到曲线上与其他点最近点的距离
    #         norm_dist = point.distance(centerline_ls)#得到垂向距离(左侧为正，右侧为负)
    #         point_on_cl = centerline_ls.interpolate(tang_dist)#得到曲线上对应点的坐标
    #         #使用封闭曲线转向的顺时针或逆时针确定垂向坐标的正负
    #         pt1 = point_on_cl.coords[0]
    #         pt2 = centerline_ls.interpolate(tang_dist + delta).coords[0]
    #         pt3 = point.coords[0]
    #         lr_coords = []
    #         lr_coords.extend([pt1, pt2, pt3])
    #         lr = LinearRing(lr_coords)
    #         if lr.is_ccw:#如果封闭曲线是顺时针，就为负；如果是逆时针，就为正
    #             hist_track_rel[seq, 0] = tang_dist
    #             hist_track_rel[seq, 1] = norm_dist
    #         else:
    #             hist_track_rel[seq, 0] = tang_dist
    #             hist_track_rel[seq, 1] = -norm_dist
    #     tang_dist_current=hist_track_rel[-1,0]
    #     hist_track_rel[:,0]-=tang_dist_current
    #     return hist_track_rel,tang_dist_current
    #
    # ## Helper function to get track future
    # def getFuture(self,csv_Id,veh_Id,centerline,t,tang_dist_current):
    #     centerline = self.C[csv_Id, centerline]
    #     track = self.T[csv_Id, veh_Id]
    #     stpt = np.argwhere(track[:, 2] == t).item()+ self.d_s
    #     enpt =  np.minimum(len(track),np.argwhere(track[:, 2] == t).item() + self.t_f + 1)#这里在数据处理的时候还要设置当前时刻不能为序列最后时刻
    #     fut_track = track[stpt:enpt:self.d_s, 0:2]
    #     fut_track_rel = np.full((fut_track.shape[0], 2), 0,dtype=float)
    #     center_st = centerline[0, :]
    #     center_ed = centerline[-1, :]
    #     # 需要填充centerline不够长的情况
    #     centerline_ls = LineString(centerline)
    #     delta = 0.01
    #     for seq in range(fut_track.shape[0]):
    #         point = Point(fut_track[seq, :])
    #         tang_dist = centerline_ls.project(point)  # 得到曲线上与其他点最近点的距离
    #         norm_dist = point.distance(centerline_ls)  # 得到垂向距离
    #         point_on_cl = centerline_ls.interpolate(tang_dist)  # 得到曲线上对应点的坐标
    #         pt1 = point_on_cl.coords[0]
    #         pt2 = centerline_ls.interpolate(tang_dist + delta).coords[0]
    #         pt3 = point.coords[0]
    #         lr_coords = []
    #         lr_coords.extend([pt1, pt2, pt3])
    #         lr = LinearRing(lr_coords)
    #         if lr.is_ccw:  # 如果封闭曲线是顺时针，就为负；如果是逆时针，就为正
    #             fut_track_rel[seq, 0] = tang_dist
    #             fut_track_rel[seq, 1] = norm_dist
    #         else:
    #             fut_track_rel[seq, 0] = tang_dist
    #             fut_track_rel[seq, 1] = -norm_dist
    #     fut_track_rel[:, 0] -= tang_dist_current
    #     return fut_track_rel

    ## Collate function for dataloader
    def collate_fn(self, samples):
        # Initialize history, history lengths, future, output mask, lateral maneuver and longitudinal maneuver batches:
        hist_batch = torch.zeros(self.t_h//self.d_s,len(samples),2)
        fut_batch = torch.zeros(self.t_f//self.d_s,len(samples),2)
        op_mask_batch = torch.zeros(self.t_f//self.d_s,len(samples),2)

        for sampleId,(hist, fut) in enumerate(samples):
            # Set up history, future, lateral maneuver and longitudinal maneuver batches:
            hist_batch[0:len(hist), sampleId, 0] = torch.from_numpy(hist[:, 0])
            hist_batch[0:len(hist), sampleId, 1] = torch.from_numpy(hist[:, 1])
            fut_batch[0:len(fut), sampleId, 0] = torch.from_numpy(fut[:, 0])
            fut_batch[0:len(fut), sampleId, 1] = torch.from_numpy(fut[:, 1])
            op_mask_batch[0:len(fut),sampleId,:] = 1
        return hist_batch, fut_batch, op_mask_batch


def maskedMSE(y_pred, y_gt, mask):
    acc = torch.zeros_like(mask)
    muX = y_pred[:, :, 0]
    muY = y_pred[:, :, 1]
    x = y_gt[:, :, 0]
    y = y_gt[:, :, 1]
    out = torch.pow(x - muX, 2) + torch.pow(y - muY, 2)
    acc[:, :, 0] = out
    acc[:, :, 1] = out
    acc = acc * mask
    lossVal = torch.sum(acc) / torch.sum(mask)
    return lossVal

def maskedMSETest(y_pred, y_gt, mask):
    acc = torch.zeros_like(mask)
    muX = y_pred[:, :, 0]
    muY = y_pred[:, :, 1]
    x = y_gt[:, :, 0]
    y = y_gt[:, :, 1]
    out = torch.pow(x - muX, 2) + torch.pow(y - muY, 2)
    acc[:, :, 0] = out
    acc[:, :, 1] = out
    acc = acc * mask
    lossVal = torch.sum(acc[:, :, 0], dim=1)
    counts = torch.sum(mask[:, :, 0], dim=1)
    return lossVal, counts


# with open("/home/wangzehan/argoverse-forecasting/SAIC/train.pkl","rb")as f:
#     raw_file=pkl.load(f)
#
# csv_Id = raw_file["train_data"][1,0].astype(int)
# veh_Id = raw_file["train_data"][1,1].astype(int)
# t = raw_file["train_data"][1,2]
# centerline=raw_file["train_data"][1,5].astype(int)
# print(csv_Id,veh_Id,t,centerline)
# print(raw_file["train_data"][1,:])
# track=raw_file["track"][csv_Id,veh_Id]
# # track=track[np.argwhere(track[:, 2] == t).item(),:]
# # print(track)
# track=track[np.argwhere(track[:, 2] == t).item()-30:np.argwhere(track[:, 2] == t).item(),0:2]
# print(track)
# centerline=raw_file["centerline"][csv_Id,centerline]
# print(centerline)
# import matplotlib.pylab as plt
# print(np.max(centerline[:,0]))
# #centerline=centerline[[np.argmax(centerline[:,0]),np.argmin(centerline[:,0])],:]
# p1=centerline[np.argmax(centerline[:,0]),:]
# p2=centerline[np.argmax(centerline[:,1]),:]
# k=(p2[1]-p1[1])/(p2[0]-p1[0])
# b=p1[1]-k*p1[0]
# p1_new=np.array([0,0*k+b],dtype=float)
# p2_new=np.array([10000,10000*k+b],dtype=float)
# centerline=np.full((2,2),0,dtype=float)
# centerline[0]=p1_new
# centerline[1]=p2_new
# x_centerline = centerline[:, 0]
# y_centerline = centerline[:, 1]
# plt.plot(x_centerline, y_centerline, color="y", linewidth=1, linestyle="--", zorder=2)
# plt.show()
# hist_track_rel=np.full((track.shape[0],2),0,dtype=float)
# for i in range(track.shape[0]):
#     point=Point(track[i,:])
#     centerline_ls = LineString(centerline)
#     tang_dist = centerline_ls.project(point)#得到曲线上与其他点最近点的距离
#     norm_dist = point.distance(centerline_ls)#得到垂向距离(左侧为正，右侧为负)
#     # print(norm_dist)
#     point_on_cl = centerline_ls.interpolate(tang_dist)#得到曲线上对应点的坐标
#     point_on_cl.coords[0]#TODO:确认下point_on_cl的延伸机制
#     #使用封闭曲线转向的顺时针或逆时针确定垂向坐标的正负
#     delta=0.01
#     pt1 = point_on_cl.coords[0]
#     pt2 = centerline_ls.interpolate(tang_dist + delta).coords[0]
#     pt3 = point.coords[0]
#     lr_coords = []
#     lr_coords.extend([pt1, pt2, pt3])
#     lr = LinearRing(lr_coords)
#     if lr.is_ccw:#如果封闭曲线是顺时针，就为负；如果是逆时针，就为正
#         hist_track_rel[i, 0] = tang_dist
#         hist_track_rel[i, 1] = norm_dist
#     else:
#         hist_track_rel[i, 0] = tang_dist
#         hist_track_rel[i, 1] = -norm_dist
# # tang_dist_current=hist_track_rel[-1,0]
# # hist_track_rel[:,0]-=tang_dist_current
# print(hist_track_rel)


#data=SAICDataset("/home/wangzehan/argoverse-forecasting/SAIC/pkl_file/train_origin.pkl")
#print(data[100])
# dataloader=DataLoader(data,batch_size=10,shuffle=True,num_workers=8,collate_fn=data.collate_fn)
# hist,fut,op_mask=next(iter(dataloader))
# print(hist.shape)
# print(fut.shape)
# print(op_mask.shape)

# num=0
# for hist,fut in data:
#     if hist[0,0]<0:
#         num+=1
# print(num)

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

# FEATURE_FORMAT = {
#     "TIMESTAMP": 0,
#     "TRACK_ID": 1,
#     "OBJECT_TYPE": 2,
#     "X": 3,
#     "Y": 4,
#     "CITY_NAME": 5,
#     "MIN_DISTANCE_FRONT": 6,
#     "MIN_DISTANCE_BACK": 7,
#     "NUM_NEIGHBORS": 8,
#     "OFFSET_FROM_CENTERLINE": 9,
#     "DISTANCE_ALONG_CENTERLINE": 10,
# }
#
#
# ## Dataset class for the NGSIM dataset
# class ArgoDataset(Dataset):
#
#     def __init__(self, pkl_file, t_h=20, t_f=30, d_s=1, enc_size=64):
#         with open(pkl_file,"rb")as f:
#             raw_file=pkl.load(f)
#         self.traj = raw_file["FEATURES"].values
#         self.nerb = raw_file["NEIGHBOUR"].values
#         self.t_h = t_h  # length of track history
#         self.t_f = t_f  # length of predicted trajectory
#         self.d_s = d_s  # down sampling rate of all sequences
#         self.enc_size = enc_size  # size of encoder LSTM
#
#     def __len__(self):
#         return len(self.traj)
#
#     def __getitem__(self, idx):
#         traj = self.traj[idx]
#         nerb = self.nerb[idx]
#         #得到预测目标的历史轨迹和未来轨迹
#         hist = traj[0:self.t_h:self.d_s,-2:]
#         fut = traj[self.t_h::self.d_s,-2:]
#         #将沿着车道线的坐标变化为相对坐标
#         current_y_refer = hist[-1,1]
#         hist[:,1]=hist[:,1]-current_y_refer
#         fut[:,1]=fut[:,1]-current_y_refer
#
#         #得到预测目标当前时刻的世界坐标系下的坐标
#         current_pt=traj[self.t_h-1,3:5]
#
#         hist_traj_all=[hist]
#         current_pt_all=[current_pt]
#         #添加周围车辆的历史轨迹和世界坐标，用于网络中的交互模块处理交互关系
#         for i in range(len(nerb["neighbour_traj"])):
#             nbr_traj = nerb["neighbour_traj"][i]
#             # 将周围车辆的轨迹沿着车道线的坐标变化为相对坐标
#             current_nbr_pt=nbr_traj[-1,1]
#             nbr_traj[:,1]=nbr_traj[:,1]-current_nbr_pt
#             hist_traj_all.append(nbr_traj)
#
#             nbr_pt = nerb["neighbour_pt"][i]
#             current_pt_all.append(nbr_pt)
#         try:
#             assert len(hist_traj_all)==len(current_pt_all)
#         except:
#             raise AssertionError
#
#         return hist_traj_all, current_pt_all, fut
#
#     ## Collate function for dataloader
#     def collate_fn(self, samples):
#
#         seq_batch_size = 0
#         seq_list = []
#         for hist_traj_all, _, _ in samples:
#             seq_batch_size+=len(hist_traj_all)
#             seq_list.append(len(hist_traj_all))
#
#         #将samples数据整合到batch维度上（sequence,batch,2）
#         hist_traj_batch = torch.zeros(self.t_h, seq_batch_size, 2)
#         current_pt_batch = torch.zeros(seq_batch_size,2)
#         fut_batch = torch.zeros(self.t_f, len(samples), 2)
#         cum_start_idx = [0] + np.cumsum(np.asarray(seq_list)).tolist()
#
#         try:
#             assert seq_batch_size==cum_start_idx[-1]
#         except:
#             raise AssertionError
#
#         for sampleId, (hist_traj_all, current_pt_all, fut) in enumerate(samples):
#
#             hist_traj_sample=np.stack(hist_traj_all,1)
#             current_pt_sample=np.stack(current_pt_all,0)
#             hist_traj_batch[:, cum_start_idx[sampleId]:cum_start_idx[sampleId+1], :] = torch.from_numpy(hist_traj_sample[:,:,:].astype("float"))
#             current_pt_batch[cum_start_idx[sampleId]:cum_start_idx[sampleId+1], :] = torch.from_numpy(current_pt_sample[:,:].astype("float"))
#             fut_batch[:, sampleId, :] = torch.from_numpy(fut[:, :].astype("float"))
#
#         return hist_traj_batch,current_pt_batch,fut_batch,cum_start_idx
#
def NLL(y_pred, y_gt):
    #对数据的结果先进行预处理，便于更好地学习
    muX = y_pred[:, :, 0]
    muY = y_pred[:, :, 1]
    sigX_1 = y_pred[:, :, 2]
    sigY_1 = y_pred[:, :, 3]
    rho_1 = y_pred[:, :, 4]
    sigX = torch.exp(sigX_1)#保证其值大于0
    sigY = torch.exp(sigY_1)#保证其值大于0
    rho = torch.tanh(rho_1)#保证其值在-1和1之间

    #根据二元高斯的概率密度函数定义损失函数
    acc = torch.zeros_like(y_gt)
    ohr = torch.pow(1-torch.pow(rho,2),-0.5)
    x = y_gt[:,:, 0]
    y = y_gt[:,:, 1]
    out = torch.pow(ohr, 2)*(torch.pow(sigX, 2)*torch.pow(x-muX, 2) + torch.pow(sigY, 2)*torch.pow(y-muY, 2) - 2*rho*torch.pow(sigX, 1)*torch.pow(sigY, 1)*(x-muX)*(y-muY)) - torch.log(sigX*sigY*ohr)
    acc[:,:,0] = out
    acc[:,:,1] = out
    lossVal = torch.mean(acc[:,:,0])
    return lossVal

def maskedMSE(y_pred,y_gt,mask_seq):
    mask=torch.zeros_like(y_gt)
    mask[0:mask_seq,:,:]=1
    acc = torch.zeros_like(y_gt)
    muX = y_pred[:, :, 0]
    muY = y_pred[:, :, 1]
    x = y_gt[:, :, 0]
    y = y_gt[:, :, 1]
    out = torch.pow(x - muX, 2) + torch.pow(y - muY, 2)
    acc[:, :, 0] = out
    acc[:, :, 1] = out
    acc=acc*mask
    lossVal = torch.mean(acc[:, :, 0])
    return lossVal


def MSE(y_pred, y_gt):
    acc = torch.zeros_like(y_gt)
    muX = y_pred[:, :, 0]
    muY = y_pred[:, :, 1]
    x = y_gt[:, :, 0]
    y = y_gt[:, :, 1]
    out = torch.pow(x - muX, 2) + torch.pow(y - muY, 2)
    acc[:, :, 0] = out
    acc[:, :, 1] = out
    lossVal = torch.mean(acc[:,:,0])
    return lossVal

def ADE(y_pred, y_gt):
    acc = torch.zeros_like(y_gt)
    muX = y_pred[:, :, 0]
    muY = y_pred[:, :, 1]
    x = y_gt[:, :, 0]
    y = y_gt[:, :, 1]
    out = torch.pow((torch.pow(x - muX, 2) + torch.pow(y - muY, 2)),0.5)
    acc[:, :, 0] = out
    acc[:, :, 1] = out
    lossVal = torch.mean(acc[:,:,0])
    return lossVal

def ADE_sequence(y_pred, y_gt):
    acc = torch.zeros_like(y_gt)
    muX = y_pred[:, :, 0]
    muY = y_pred[:, :, 1]
    x = y_gt[:, :, 0]
    y = y_gt[:, :, 1]
    out = torch.pow((torch.pow(x - muX, 2) + torch.pow(y - muY, 2)),0.5)
    acc[:, :, 0] = out
    acc[:, :, 1] = out
    lossVal = torch.mean(acc[:,:,0],0)
    return lossVal

def FDE(y_pred, y_gt):
    muX = y_pred[-1, :, 0]
    muY = y_pred[-1, :, 1]
    x = y_gt[-1, :, 0]
    y = y_gt[-1, :, 1]
    out = torch.pow((torch.pow(x - muX, 2) + torch.pow(y - muY, 2)),0.5)
    lossVal = torch.mean(out)
    return lossVal

def FDE_sequence(y_pred, y_gt):
    muX = y_pred[-1, :, 0]
    muY = y_pred[-1, :, 1]
    x = y_gt[-1, :, 0]
    y = y_gt[-1, :, 1]
    out = torch.pow((torch.pow(x - muX, 2) + torch.pow(y - muY, 2)),0.5)
    lossVal = out
    return lossVal


def MSElastdistance(y_pred, y_gt):
    acc = torch.zeros_like(y_pred[0, :, 0])
    muX = y_pred[-1, :, 0]
    muY = y_pred[-1, :, 1]
    x = y_gt[-1, :, 0]
    y = y_gt[-1, :, 1]
    acc = torch.pow(x - muX, 2) + torch.pow(y - muY, 2)
    return acc

class ArgoverseDataset(Dataset):
    def __init__(self, pkl_file, t_h=20, t_f=30, d_s=1,absolute=False,social_input=False,vehicle_input=False,val_metric=False,save_pkl=False,test=False):
        with open(pkl_file,"rb")as f:
            raw_file=pkl.load(f)
        self.absolute=absolute
        self.social_input = social_input
        self.vehicle_input = vehicle_input
        self.val_metric = val_metric
        self.save_pkl = save_pkl
        self.test=test
        if self.test:
            self.cand_centerlines=raw_file["CANDIDATE_CENTERLINES"].values
            self.cand_centerlines_nt=raw_file["CANDIDATE_NT_DISTANCES"].values
        self.traj = raw_file["FEATURES"].values
        if self.val_metric:
            self.centerline = raw_file["ORACLE_CENTERLINE"].values
        if self.save_pkl:
            self.sequence = raw_file["SEQUENCE"].values
        self.t_h = t_h  # length of track history
        self.t_f = t_f  # length of predicted trajectory
        self.d_s = d_s  # down sampling rate of all sequences

    def __len__(self):
        return len(self.traj)

    def __getitem__(self, idx):
        traj = self.traj[idx]
        if self.test:
            cand_centerlines_nt=self.cand_centerlines_nt[idx]
            cand_centerlines = self.cand_centerlines[idx]
            assert len(cand_centerlines)==len(cand_centerlines_nt)
            hist_list=[]
            current_y_refer_list=[]
            for i in range(len(cand_centerlines_nt)):
                current_hist=cand_centerlines_nt[i][0:self.t_h:self.d_s,:]
                # print(current_hist)
                # print(current_hist.shape)
                current_y_refer=current_hist[-1,1]
                current_hist[:,1]=current_hist[:,1]-current_y_refer
                hist_list.append(current_hist)
                current_y_refer_list.append(current_y_refer)
            # hist = traj[0:self.t_h:self.d_s, [9, 10]]
            # current_y_refer = hist[-1, 1]
            # hist[:, 1] = hist[:, 1] - current_y_refer
            # cand_centerlines=self.cand_centerlines[idx]
            seq = self.sequence[idx]
            return hist_list,current_y_refer_list,cand_centerlines,seq
        #得到预测目标的历史轨迹和未来轨迹
        if self.absolute:
            hist=traj[0:self.t_h:self.d_s, [3,4]]
            hist_ref=hist[-1,:]
            hist=hist-hist_ref
            fut=traj[self.t_h::self.d_s,[3,4]]
            fut=fut-hist_ref
            if self.val_metric:
                # oracle_ct = self.centerline[idx]
                # fut = traj[self.t_h::self.d_s, [3, 4]]
                if self.save_pkl:
                    seq = self.sequence[idx]
                    return seq, hist, fut, hist_ref
            else:
                return hist,fut,hist_ref

        if self.social_input:
            # hist=traj[0:self.t_h:self.d_s,[9,10,6,7,8]]
            hist = traj[0:self.t_h:self.d_s, [9, 10]]
            social = traj[0:self.t_h:self.d_s, [6, 7, 8]]
        else:
            hist=traj[0:self.t_h:self.d_s,-2:]
        fut = traj[self.t_h::self.d_s,-2:]
        #将沿着车道线的坐标变化为相对坐标
        current_y_refer = hist[-1,1]
        hist[:,1]=hist[:,1]-current_y_refer
        fut[:,1]=fut[:,1]-current_y_refer

        if self.val_metric:
            oracle_ct=self.centerline[idx]
            fut = traj[self.t_h::self.d_s,[3,4]]
            if self.save_pkl:
                seq=self.sequence[idx]
                return seq, hist, fut, current_y_refer, oracle_ct
            elif self.vehicle_input:
                hist_v = (traj[self.d_s:self.t_h + self.d_s:self.d_s, -2:] - traj[0:self.t_h:self.d_s, -2:]) / 0.1
                return hist, hist_v, fut, current_y_refer, oracle_ct
            elif self.social_input:
                return hist, social, fut, current_y_refer, oracle_ct

            else:
                return hist, fut, current_y_refer, oracle_ct
        elif self.vehicle_input:
            hist_v=(traj[self.d_s:self.t_h+self.d_s:self.d_s,-2:]-traj[0:self.t_h:self.d_s,-2:])/0.1
            return hist, hist_v, fut
        elif self.social_input:
            return hist, social, fut
        else:
            return hist, fut

    def collate_fn(self, samples):
        #将samples数据整合到batch维度上（sequence,batch,2）
        if self.test:
            hist_traj_batch_list=[]
            current_y_refer_batch_list=[]
            seq_list=[]
            cand_centerlines_list=[]
            for sampleId,(hist,current_y_refer,cand_centerlines,seq) in enumerate(samples):
                hist_traj_batch=torch.zeros(self.t_h,len(hist),2)
                current_y_refer_batch = torch.zeros(len(hist), 1)
                for ii,current_hist_traj in enumerate(hist):
                    hist_traj_batch[:,ii,:]=torch.from_numpy(current_hist_traj[:,:].astype("float"))
                    current_y_refer_batch[ii,:]=current_y_refer[ii]
                hist_traj_batch_list.append(hist_traj_batch)
                current_y_refer_batch_list.append(current_y_refer_batch)
            seq_list.append(seq)
            cand_centerlines_list.append(cand_centerlines)
            return hist_traj_batch_list, current_y_refer_batch_list, seq_list, cand_centerlines_list

        if self.absolute:
            hist_traj_batch = torch.zeros(self.t_h, len(samples), 2)
            fut_batch = torch.zeros(self.t_f, len(samples), 2)
            hist_ref_batch = torch.zeros(len(samples), 2)
            if self.val_metric:
                if self.save_pkl:
                    seq_list = []
                    for sampleId, (seq, hist, fut, hist_ref) in enumerate(samples):
                        hist_traj_batch[:, sampleId, :] = torch.from_numpy(hist[:, :].astype("float"))
                        fut_batch[:, sampleId, :] = torch.from_numpy(fut[:, :].astype("float"))
                        hist_ref_batch[sampleId, :] = torch.from_numpy(hist_ref.astype("float"))
                        seq_list.append(seq)
                    return hist_traj_batch, fut_batch, hist_ref_batch, seq_list
            else:
                for sampleId, (hist, fut,hist_ref) in enumerate(samples):
                    hist_traj_batch[:, sampleId, :] = torch.from_numpy(hist[:, :].astype("float"))
                    fut_batch[:, sampleId, :] = torch.from_numpy(fut[:, :].astype("float"))
                    hist_ref_batch[sampleId, :] = torch.from_numpy(hist_ref.astype("float"))
                return hist_traj_batch, fut_batch, hist_ref_batch

        if self.social_input:
            #hist_traj_batch = torch.zeros(self.t_h, len(samples), 5)
            hist_traj_batch=torch.zeros(self.t_h, len(samples), 2)
            hist_soc_batch=torch.zeros(self.t_h, len(samples), 3)
        else:
            hist_traj_batch = torch.zeros(self.t_h, len(samples), 2)
        fut_batch = torch.zeros(self.t_f, len(samples), 2)
        current_y_refer_batch = torch.zeros(len(samples), 1)
        if self.vehicle_input:
            hist_v_batch = torch.zeros(self.t_h, len(samples), 2)

        if self.val_metric:
            if self.save_pkl:
                seq_list=[]
                centerline_batch=[]
                for sampleId, (seq, hist_traj_all, fut, current_y_refer, oracle_ct) in enumerate(samples):
                    hist_traj_batch[:, sampleId, :] = torch.from_numpy(hist_traj_all[:,:].astype("float"))
                    fut_batch[:, sampleId, :] = torch.from_numpy(fut[:, :].astype("float"))
                    current_y_refer_batch[sampleId,:] = current_y_refer
                    seq_list.append(seq)
                    centerline_batch.append(oracle_ct)
                return hist_traj_batch,fut_batch,current_y_refer_batch,seq_list,centerline_batch

            elif self.vehicle_input:
                centerline_batch=[]
                for sampleId, (hist_traj_all, hist_v_all, fut, current_y_refer, oracle_ct) in enumerate(samples):
                    hist_traj_batch[:, sampleId, :] = torch.from_numpy(hist_traj_all[:,:].astype("float"))
                    hist_v_batch[:, sampleId, :] = torch.from_numpy(hist_v_all[:, :].astype("float"))
                    fut_batch[:, sampleId, :] = torch.from_numpy(fut[:, :].astype("float"))
                    current_y_refer_batch[sampleId,:] = current_y_refer
                    centerline_batch.append(oracle_ct)
                return hist_traj_batch,hist_v_batch,fut_batch,current_y_refer_batch,centerline_batch

            elif self.social_input:
                centerline_batch=[]
                for sampleId, (hist_traj_all, hist_s_all, fut, current_y_refer, oracle_ct) in enumerate(samples):
                    hist_traj_batch[:, sampleId, :] = torch.from_numpy(hist_traj_all[:,:].astype("float"))
                    hist_soc_batch[:, sampleId, :] = torch.from_numpy(hist_s_all[:, :].astype("float"))
                    fut_batch[:, sampleId, :] = torch.from_numpy(fut[:, :].astype("float"))
                    current_y_refer_batch[sampleId,:] = current_y_refer
                    centerline_batch.append(oracle_ct)
                return hist_traj_batch,hist_soc_batch,fut_batch,current_y_refer_batch,centerline_batch

            else:
                centerline_batch=[]
                for sampleId, (hist_traj_all, fut, current_y_refer, oracle_ct) in enumerate(samples):
                    hist_traj_batch[:, sampleId, :] = torch.from_numpy(hist_traj_all[:,:].astype("float"))
                    fut_batch[:, sampleId, :] = torch.from_numpy(fut[:, :].astype("float"))
                    current_y_refer_batch[sampleId,:] = current_y_refer
                    centerline_batch.append(oracle_ct)
                return hist_traj_batch,fut_batch,current_y_refer_batch,centerline_batch

        elif self.vehicle_input:
            for sampleId, (hist_traj_all,hist_v_all, fut) in enumerate(samples):
                hist_traj_batch[:, sampleId, :] = torch.from_numpy(hist_traj_all[:,:].astype("float"))
                hist_v_batch[:, sampleId, :] = torch.from_numpy(hist_v_all[:,:].astype("float"))
                fut_batch[:, sampleId, :] = torch.from_numpy(fut[:, :].astype("float"))
            return hist_traj_batch,hist_v_batch,fut_batch

        elif self.social_input:
            for sampleId, (hist_traj_all,hist_soc_all, fut) in enumerate(samples):
                hist_traj_batch[:, sampleId, :] = torch.from_numpy(hist_traj_all[:,:].astype("float"))
                hist_soc_batch[:, sampleId, :] = torch.from_numpy(hist_soc_all[:,:].astype("float"))
                fut_batch[:, sampleId, :] = torch.from_numpy(fut[:, :].astype("float"))
            return hist_traj_batch,hist_soc_batch,fut_batch

        else:
            for sampleId, (hist_traj_all, fut) in enumerate(samples):
                hist_traj_batch[:, sampleId, :] = torch.from_numpy(hist_traj_all[:,:].astype("float"))
                fut_batch[:, sampleId, :] = torch.from_numpy(fut[:, :].astype("float"))
            return hist_traj_batch,fut_batch

# root_dir="/home/wangzehan/argoverse-forecast/data/data_nerb/forecasting_features_val.pkl"
# with open(root_dir,"rb")as f:
#     file=pkl.load(f)
# print(len((file["NEIGHBOUR"].values[1])["neighbour_traj"]))
# # print(len(file[ "ORACLE_CENTERLINE"].values))
# trdataset=ArgoverseDataset(pkl_file=root_dir)
# trdataloader=DataLoader(trdataset,batch_size=128,shuffle=True,num_workers=8)
# print(len(trdataset))
# hist,fut=trdataset[1]
# print(hist)
# print(fut)
#
# hist,fut = next(iter(trdataloader))
# print(hist.double().dtype)
# print(fut.double().dtype)

# a=torch.tensor([3]).double()
# print(a.dtype)
# print(a.dtype)

# trSet = ArgoverseDataset('../data/forecasting_features_val.pkl',absolute=True,val_metric=False,save_pkl=False)#train datatset
# hist,fut=trSet[0]
# print(hist)
# print(fut)
# trDataloader = DataLoader(trSet,batch_size=128,shuffle=True,num_workers=8,collate_fn=trSet.collate_fn)
# hist,fut=next(iter(trDataloader))
# print(hist.shape)
# print(fut.shape)
#print(op_mask.shape)
# tsSet = ArgoverseDataset('../data/forecasting_features_test.pkl',save_pkl=True,test=True)
# tsDataloader = DataLoader(tsSet,batch_size=1,shuffle=True,num_workers=8,collate_fn=tsSet.collate_fn)
# hist,current_y_refer,cand_centerlines,seq=next(iter(tsDataloader))
# print(hist)
# print(current_y_refer)
# print(cand_centerlines)
# print(seq)

# root_dir="/home/wangzehan/argoverse-forecast/data/forecasting_features_test.pkl"
# with open(root_dir,"rb")as f:
#     file=pkl.load(f)
# print(file.keys())