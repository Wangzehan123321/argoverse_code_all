from torch.utils.data import Dataset,DataLoader
import pandas as pd
import pickle as pkl
import numpy as np
import torch
from shapely.geometry import LinearRing, LineString, Point, Polygon
import random

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
    def __init__(self, pkl_file, t_h=20, t_f=30, d_s=1,road_feature=False,absolute=False,social_input=False,vehicle_input=False,val_metric=False,save_pkl=False):
        with open(pkl_file,"rb")as f:
            raw_file=pkl.load(f)
        self.road_feature=road_feature
        self.absolute=absolute
        self.social_input = social_input
        self.vehicle_input = vehicle_input
        self.val_metric = val_metric
        self.save_pkl = save_pkl
        self.traj = raw_file["FEATURES"].values
        if self.val_metric or self.road_feature:
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
        #得到预测目标的历史轨迹和未来轨迹
        if self.absolute:
            hist=traj[0:self.t_h:self.d_s, [3,4]]
            hist_ref=hist[-1,:]
            hist=hist-hist_ref
            fut=traj[self.t_h::self.d_s,[3,4]]
            fut=fut-hist_ref
            return hist,fut

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

        if self.road_feature:
            v = (traj[19,10] - traj[17,10]) / 2#计算obs的最后两帧的速度
            oracle_ct = self.centerline[idx]
            centerline_ls = LineString(oracle_ct)
            point = Point(traj[19, [3, 4]])
            tang_dist = centerline_ls.project(point)  # 得到曲线上与其他点最近点的距离
            road_feature_list = []
            for i in range(30):
                tang_dist_current = tang_dist + i * v
                point_on_cl = centerline_ls.interpolate(tang_dist_current)
                road_feature_list.append(point_on_cl.coords[0])
            road_feature=np.array(road_feature_list)
            road_feature-=road_feature[0,:]
            if self.val_metric:
                oracle_ct=self.centerline[idx]
                fut = traj[self.t_h::self.d_s,[3,4]]
                if self.save_pkl:
                    seq=self.sequence[idx]
                    return seq, hist,road_feature, fut, current_y_refer, oracle_ct
                elif self.vehicle_input:
                    hist_v = (traj[self.d_s:self.t_h + self.d_s:self.d_s, -2:] - traj[0:self.t_h:self.d_s, -2:]) / 0.1
                    return hist, hist_v, road_feature, fut, current_y_refer, oracle_ct
                elif self.social_input:
                    return hist, social, road_feature, fut, current_y_refer, oracle_ct

                else:
                    return hist, fut, road_feature, current_y_refer, oracle_ct
            elif self.vehicle_input:
                hist_v=(traj[self.d_s:self.t_h+self.d_s:self.d_s,-2:]-traj[0:self.t_h:self.d_s,-2:])/0.1
                return hist, hist_v, road_feature, fut
            elif self.social_input:
                return hist, social, road_feature, fut
            else:
                return hist, road_feature, fut
        else:
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
        if self.road_feature:
            road_feature_batch=torch.zeros(30,len(samples),2)
        #将samples数据整合到batch维度上（sequence,batch,2）
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

        if self.road_feature:
            if self.val_metric:
                if self.save_pkl:
                    seq_list = []
                    centerline_batch = []
                    for sampleId, (seq, hist_traj_all,road_feature, fut, current_y_refer, oracle_ct) in enumerate(samples):
                        hist_traj_batch[:, sampleId, :] = torch.from_numpy(hist_traj_all[:, :].astype("float"))
                        fut_batch[:, sampleId, :] = torch.from_numpy(fut[:, :].astype("float"))
                        road_feature_batch[:,sampleId,:]=torch.from_numpy(road_feature[:, :].astype("float"))
                        current_y_refer_batch[sampleId, :] = current_y_refer
                        seq_list.append(seq)
                        centerline_batch.append(oracle_ct)
                    return hist_traj_batch, road_feature_batch, fut_batch, current_y_refer_batch, seq_list, centerline_batch

                elif self.vehicle_input:
                    centerline_batch = []
                    for sampleId, (hist_traj_all, hist_v_all, road_feature, fut, current_y_refer, oracle_ct) in enumerate(samples):
                        hist_traj_batch[:, sampleId, :] = torch.from_numpy(hist_traj_all[:, :].astype("float"))
                        hist_v_batch[:, sampleId, :] = torch.from_numpy(hist_v_all[:, :].astype("float"))
                        road_feature_batch[:, sampleId, :] = torch.from_numpy(road_feature[:, :].astype("float"))
                        fut_batch[:, sampleId, :] = torch.from_numpy(fut[:, :].astype("float"))
                        current_y_refer_batch[sampleId, :] = current_y_refer
                        centerline_batch.append(oracle_ct)
                    return hist_traj_batch, hist_v_batch, road_feature_batch, fut_batch, current_y_refer_batch, centerline_batch

                elif self.social_input:
                    centerline_batch = []
                    for sampleId, (hist_traj_all, hist_s_all,road_feature, fut, current_y_refer, oracle_ct) in enumerate(samples):
                        hist_traj_batch[:, sampleId, :] = torch.from_numpy(hist_traj_all[:, :].astype("float"))
                        hist_soc_batch[:, sampleId, :] = torch.from_numpy(hist_s_all[:, :].astype("float"))
                        road_feature_batch[:, sampleId, :] = torch.from_numpy(road_feature[:, :].astype("float"))
                        fut_batch[:, sampleId, :] = torch.from_numpy(fut[:, :].astype("float"))
                        current_y_refer_batch[sampleId, :] = current_y_refer
                        centerline_batch.append(oracle_ct)
                    return hist_traj_batch, hist_soc_batch,road_feature_batch,fut_batch, current_y_refer_batch, centerline_batch

                else:
                    centerline_batch = []
                    for sampleId, (hist_traj_all, fut, road_feature, current_y_refer, oracle_ct) in enumerate(samples):
                        hist_traj_batch[:, sampleId, :] = torch.from_numpy(hist_traj_all[:, :].astype("float"))
                        road_feature_batch[:, sampleId, :] = torch.from_numpy(road_feature[:, :].astype("float"))
                        fut_batch[:, sampleId, :] = torch.from_numpy(fut[:, :].astype("float"))
                        current_y_refer_batch[sampleId, :] = current_y_refer
                        centerline_batch.append(oracle_ct)
                    return hist_traj_batch, road_feature_batch, fut_batch, current_y_refer_batch, centerline_batch

            elif self.vehicle_input:
                for sampleId, (hist_traj_all, hist_v_all,road_feature, fut) in enumerate(samples):
                    hist_traj_batch[:, sampleId, :] = torch.from_numpy(hist_traj_all[:, :].astype("float"))
                    hist_v_batch[:, sampleId, :] = torch.from_numpy(hist_v_all[:, :].astype("float"))
                    road_feature_batch[:, sampleId, :] = torch.from_numpy(road_feature[:, :].astype("float"))
                    fut_batch[:, sampleId, :] = torch.from_numpy(fut[:, :].astype("float"))
                return hist_traj_batch, hist_v_batch, road_feature_batch, fut_batch

            elif self.social_input:
                for sampleId, (hist_traj_all, hist_soc_all, road_feature, fut) in enumerate(samples):
                    hist_traj_batch[:, sampleId, :] = torch.from_numpy(hist_traj_all[:, :].astype("float"))
                    hist_soc_batch[:, sampleId, :] = torch.from_numpy(hist_soc_all[:, :].astype("float"))
                    road_feature_batch[:, sampleId, :] = torch.from_numpy(road_feature[:, :].astype("float"))
                    fut_batch[:, sampleId, :] = torch.from_numpy(fut[:, :].astype("float"))
                return hist_traj_batch, hist_soc_batch, road_feature_batch, fut_batch

        else:
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

# valSet = ArgoverseDataset('../data/forecasting_features_val.pkl',road_feature=True,social_input=False,vehicle_input=True)#val dataset
# hist, hist_v, road_feature, fut=valSet[0]
# valDataloader = DataLoader(valSet,batch_size=128,shuffle=True,num_workers=8,collate_fn=valSet.collate_fn)
# hist_traj_batch, hist_v_batch, road_feature_batch, fut_batch=next(iter(valDataloader))
# print(hist_traj_batch.shape)
# print(hist_v_batch.shape)
# print(road_feature_batch.shape)
# print(fut_batch.shape)
# print(hist.shape)
# print(hist_v)
# print(road_feature.shape)
# print(fut.shape)