from torch.utils.data import Dataset, DataLoader
import pickle as pkl
import numpy as np
import torch
# pkl_file='../data/data_nerb/forecasting_features_train.pkl'
# with open(pkl_file, "rb")as f:
#     raw_file = pkl.load(f)
# traj = raw_file["FEATURES"].values
# #考虑与周围车辆之间的交互关系
# nerb_info = raw_file["NEIGHBOUR"].values
# num_list=[]
# with torch.no_grad():
#     for idx in range(len(traj)):
#         target_pt = traj[idx][19, [3, 4]]
#         nerb_pt_list=nerb_info[idx]["neighbour_pt"]
#         #num_list.append(len(nerb_pt_list))
#         num_all=0
#         for nerb in nerb_pt_list:
#             if pow(nerb[0]-target_pt[0],2)+pow(nerb[1]-target_pt[1],2)<25*25:
#                 num_all+=1
#             else:
#                 pass
#         num_list.append(num_all)
# with open("./num_list.pkl","wb")as f:
#     pkl.dump(num_list,f)


import matplotlib.pyplot as plt
with open("./num_list.pkl","rb")as f:
    data=pkl.load(f)
# for i in range(len(data)):
#     data[i]-=10
plt.hist(x=data)
plt.show()