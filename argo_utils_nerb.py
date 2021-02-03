from torch.utils.data import Dataset, DataLoader
import pickle as pkl
import numpy as np
import torch
import random
class ArgoverseDataset(Dataset):
    def __init__(self, pkl_file, t_h=20, t_f=30, d_s=1,interaction=False,absolute=False,social_input=False,vehicle_input=False,val_metric=False,save_pkl=False):
        with open(pkl_file,"rb")as f:
            raw_file=pkl.load(f)
        self.interaction=interaction
        self.absolute=absolute
        self.social_input = social_input
        self.vehicle_input = vehicle_input
        self.val_metric = val_metric
        self.save_pkl = save_pkl
        self.traj = raw_file["FEATURES"].values
        if self.interaction:#考虑与周围车辆之间的交互关系
            self.nerb_info = raw_file["NEIGHBOUR"].values
        if self.val_metric:#用于在验证时将轨迹都转化回世界坐标系下进行评价
            self.centerline = raw_file["ORACLE_CENTERLINE"].values
        if self.save_pkl:#用于在验证时将所有轨迹都保存下来便于进行可视化
            self.sequence = raw_file["SEQUENCE"].values
        self.t_h = t_h  # length of track history
        self.t_f = t_f  # length of predicted trajectory
        self.d_s = d_s  # down sampling rate of all sequences

    def __len__(self):
        return len(self.traj)

    def __getitem__(self, idx):
        traj = self.traj[idx]
        #得到预测目标的历史轨迹和未来轨迹
        if self.absolute:#使用世界坐标系下的相对坐标
            hist=traj[0:self.t_h:self.d_s, [3,4]]
            hist_ref=hist[-1,:]
            hist=hist-hist_ref
            fut=traj[self.t_h::self.d_s,[3,4]]
            fut=fut-hist_ref
            return hist,fut

        if self.social_input:#直接将交互信息作为输入（分别与前后车的距离以及周围车的数量）
            # hist=traj[0:self.t_h:self.d_s,[9,10,6,7,8]]
            hist = traj[0:self.t_h:self.d_s, [9, 10]]
            social = traj[0:self.t_h:self.d_s, [6, 7, 8]]
        else:
            hist=traj[0:self.t_h:self.d_s,-2:]
        fut = traj[self.t_h::self.d_s,-2:]
        #将沿着车道线的坐标变化为相对坐标（这里只改变了y方向，即y方向的相对，x方向还是绝对）
        current_y_refer = hist[-1,1]
        hist[:,1]=hist[:,1]-current_y_refer
        fut[:,1]=fut[:,1]-current_y_refer

        #新添加的部分（周围交通参与者的运动状态）
        if self.interaction:
            target_pt=traj[self.t_h-self.d_s,[3,4]]
            nerb_traj_list=self.nerb_info[idx]["neighbour_traj"]
            nerb_pt_list=self.nerb_info[idx]["neighbour_pt"]
            all_traj_rel_list=[hist]
            all_pt_list=[target_pt]
            assert len(nerb_traj_list)==len(nerb_pt_list)
            for i in range(len(nerb_traj_list)):
                current_y_refer_nerb=nerb_traj_list[i][-1,1]
                nerb_traj_list[i][:,1]=nerb_traj_list[i][:,1]-current_y_refer_nerb
                all_traj_rel_list.append(nerb_traj_list[i])
                all_pt_list.append(nerb_pt_list[i])

            if self.val_metric:
                oracle_ct=self.centerline[idx]
                fut=traj[self.t_h::self.d_s,[3,4]]
                return all_traj_rel_list,all_pt_list,fut,current_y_refer,oracle_ct
            else:
                return all_traj_rel_list,all_pt_list,fut

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

        if self.interaction:
            if self.val_metric:
                centerline_batch=[]
                seq_hist_all = 0
                seq_list = []
                for all_traj_rel_list, all_pt_list, fut,_,_ in samples:
                    seq_hist_all += len(all_pt_list)
                    seq_list.append(len(all_pt_list))
                hist_traj_all_batch = torch.zeros(self.t_h, seq_hist_all, 2)
                hist_pt_all_batch = torch.zeros(seq_hist_all, 2)
                cum_start_idx = [0] + np.cumsum(np.asarray(seq_list)).tolist()

                for sampleId, (all_traj_rel_list, all_pt_list, fut, current_y_refer, oracle_ct) in enumerate(samples):
                    all_traj_rel = np.stack(all_traj_rel_list, 1)
                    all_pt = np.stack(all_pt_list, 0)

                    hist_traj_all_batch[:, cum_start_idx[sampleId]:cum_start_idx[sampleId + 1], :] = torch.from_numpy(
                        all_traj_rel[:, :, :].astype("float"))
                    hist_pt_all_batch[cum_start_idx[sampleId]:cum_start_idx[sampleId + 1], :] = torch.from_numpy(
                        all_pt[:, :].astype("float"))
                    fut_batch[:, sampleId, :] = torch.from_numpy(fut[:, :].astype("float"))

                    current_y_refer_batch[sampleId, :] = current_y_refer
                    centerline_batch.append(oracle_ct)
                return hist_traj_all_batch, hist_pt_all_batch, cum_start_idx, fut_batch, current_y_refer_batch, centerline_batch

            else:
                seq_hist_all = 0
                seq_list = []
                for all_traj_rel_list, all_pt_list, fut in samples:
                    seq_hist_all += len(all_pt_list)
                    seq_list.append(len(all_pt_list))
                hist_traj_all_batch = torch.zeros(self.t_h, seq_hist_all, 2)
                hist_pt_all_batch = torch.zeros(seq_hist_all, 2)
                cum_start_idx = [0] + np.cumsum(np.asarray(seq_list)).tolist()
                for sampleId,(all_traj_rel_list,all_pt_list,fut) in enumerate(samples):

                    all_traj_rel = np.stack(all_traj_rel_list, 1)
                    all_pt = np.stack(all_pt_list, 0)

                    hist_traj_all_batch[:, cum_start_idx[sampleId]:cum_start_idx[sampleId + 1], :] = torch.from_numpy(
                        all_traj_rel[:, :, :].astype("float"))
                    hist_pt_all_batch[cum_start_idx[sampleId]:cum_start_idx[sampleId + 1], :] = torch.from_numpy(
                        all_pt[:, :].astype("float"))
                    fut_batch[:, sampleId, :] = torch.from_numpy(fut[:, :].astype("float"))
                return hist_traj_all_batch,hist_pt_all_batch,cum_start_idx,fut_batch

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

# root_dir="/home/wangzehan/argoverse-forecast/data/data_nerb/forecasting_features_val.pkl"
# trSet=ArgoverseDataset(pkl_file=root_dir,interaction=True)
# trDataloader = DataLoader(trSet,batch_size=1,shuffle=True,num_workers=8,collate_fn=trSet.collate_fn)
# hist_batch, hist_pos, cum_start_idx, fut_batch=next(iter(trDataloader))
# print(hist_batch)
# print(hist_pos)
# print(cum_start_idx)
# print(fut_batch.shape)
# with open(root_dir,"rb")as f:
#     file=pkl.load(f)
# print(len(file["NEIGHBOUR"].values))
# print(file["NEIGHBOUR"].values[1]["neighbour_traj"][0])
# seq_list=[1,3,4,7]
# cum_start_idx = [0] + np.cumsum(np.asarray(seq_list)).tolist()
# print(cum_start_idx)
# print(torch.cuda.is_available())