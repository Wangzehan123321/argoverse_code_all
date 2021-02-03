import matplotlib.pylab as plt
import math
import numpy as np
from typing import Any, Dict, List, Optional, Tuple, Union
from shapely.geometry import Point, Polygon, LineString, LinearRing
def get_xy_from_nt_seq(nt_seq: np.ndarray,
                       centerlines: List[np.ndarray]) -> np.ndarray:
    """Convert n-t coordinates to x-y, i.e., convert from centerline curvilinear coordinates to map coordinates.

    Args:
        nt_seq (numpy array): Array of shape (num_tracks x seq_len x 2) where last dimension has 'n' (offset from centerline) and 't' (distance along centerline)
        centerlines (list of numpy array): Centerline for each track
    Returns:
        xy_seq (numpy array): Array of shape (num_tracks x seq_len x 2) where last dimension contains coordinates in map frame

    """
    seq_len = nt_seq.shape[1]

    # coordinates obtained by interpolating distances on the centerline
    xy_seq = np.zeros(nt_seq.shape)
    for i in range(nt_seq.shape[0]):
        curr_cl = centerlines[i]
        line_string = LineString(curr_cl)
        for time in range(seq_len):

            # Project nt to xy
            offset_from_cl = nt_seq[i][time][0]
            dist_along_cl = nt_seq[i][time][1]
            x_coord, y_coord = get_xy_from_nt(offset_from_cl, dist_along_cl,
                                              curr_cl)
            xy_seq[i, time, 0] = x_coord
            xy_seq[i, time, 1] = y_coord

    return xy_seq


def get_xy_from_nt(n: float, t: float,
                   centerline: np.ndarray) -> Tuple[float, float]:
    """Convert a single n-t coordinate (centerline curvilinear coordinate) to absolute x-y.

    Args:
        n (float): Offset from centerline
        t (float): Distance along the centerline
        centerline (numpy array): Centerline coordinates
    Returns:
        x1 (float): x-coordinate in map frame
        y1 (float): y-coordinate in map frame

    """
    line_string = LineString(centerline)

    # If distance along centerline is negative, keep it to the start of line
    point_on_cl = line_string.interpolate(
        t) if t > 0 else line_string.interpolate(0)
    local_ls = None

    # Find 2 consective points on centerline such that line joining those 2 points
    # contains point_on_cl
    for i in range(len(centerline) - 1):
        pt1 = centerline[i]
        pt2 = centerline[i + 1]
        ls = LineString([pt1, pt2])
        if ls.distance(point_on_cl) < 1e-8:
            local_ls = ls
            break

    assert local_ls is not None, "XY from N({}) T({}) not computed correctly".format(
        n, t)

    pt1, pt2 = local_ls.coords[:]
    x0, y0 = point_on_cl.coords[0]

    # Determine whether the coordinate lies on left or right side of the line formed by pt1 and pt2
    # Find a point on either side of the line, i.e., (x1_1, y1_1) and (x1_2, y1_2)
    # If the ring formed by (pt1, pt2, (x1_1, y1_1)) is counter clockwise, then it lies on the left

    # Deal with edge cases
    # Vertical
    if pt1[0] == pt2[0]:
        m = 0
        x1_1, x1_2 = x0 + n, x0 - n
        y1_1, y1_2 = y0, y0
    # Horizontal
    elif pt1[1] == pt2[1]:
        m = float("inf")
        x1_1, x1_2 = x0, x0
        y1_1, y1_2 = y0 + n, y0 - n
    # General case
    else:
        ls_slope = (pt2[1] - pt1[1]) / (pt2[0] - pt1[0])
        m = -1 / ls_slope

        x1_1 = x0 + n / math.sqrt(1 + m**2)
        y1_1 = y0 + m * (x1_1 - x0)
        x1_2 = x0 - n / math.sqrt(1 + m**2)
        y1_2 = y0 + m * (x1_2 - x0)

    # Rings formed by pt1, pt2 and coordinates computed above
    lr1 = LinearRing([pt1, pt2, (x1_1, y1_1)])
    lr2 = LinearRing([pt1, pt2, (x1_2, y1_2)])

    # If ring is counter clockwise
    if lr1.is_ccw:
        x_ccw, y_ccw = x1_1, y1_1
        x_cw, y_cw = x1_2, y1_2
    else:
        x_ccw, y_ccw = x1_2, y1_2
        x_cw, y_cw = x1_1, y1_1

    # If offset is positive, coordinate on the left
    if n > 0:
        x1, y1 = x_ccw, y_ccw
    # Else, coordinate on the right
    else:
        x1, y1 = x_cw, y_cw

    return x1, y1


import torch
from model_SAIC import SAICNet,init_weights
from SAIC_utils import ArgoverseDataset,MSE,ADE,FDE,ADE_sequence,FDE_sequence
from torch.utils.data import DataLoader
import pickle as pkl

## Network Arguments
args = {}
args['use_cuda'] = True
args['encoder_size'] = 64
args['decoder_size'] = 64
args['in_length'] = 20
args['out_length'] = 30
args['input_embedding_size'] = 32
args['pool_embedding_size'] = 64
args['num_lon_classes'] = 3
args['use_maneuvers'] = True
#新加的是否使用交互模块
args["social_input"] = False
args['interaction'] = False
args["vehicle_input"]=False

# Initialize network
net = SAICNet(args)
net.load_state_dict(torch.load('./SAIC/trained_models/argo_lstm_multi_ade_pretrain_SAIC/cslstm_38.tar'))
if args['use_cuda']:
    net = net.cuda()

tsSet = ArgoverseDataset('../data/forecasting_features_test.pkl',save_pkl=True,test=True)
tsDataloader = DataLoader(tsSet,batch_size=1,shuffle=True,num_workers=8,collate_fn=tsSet.collate_fn)

with torch.no_grad():
    traj_all_dict={}
    for i, data in enumerate(tsDataloader):
        # st_time = time.time()
        hist_list, current_y_refer_list, seq_list, cand_centerlines = data

        for jj in range(len(hist_list)):
            current_hist=hist_list[jj]
            current_y_refer=current_y_refer_list[jj]
            # Initialize Variables
            if args['use_cuda']:
                current_hist = current_hist.cuda()
                current_y_refer = current_y_refer.cuda()

            traj_temp = []
            for kk in range(current_hist.shape[1]):
                hist=current_hist[:,kk:kk+1,:]
                y_refer=current_y_refer[kk:kk+1,:]
                centerline=cand_centerlines[jj][kk]
                fut_pred_all, lon_pred = net(hist)
                fut_1, fut_2, fut_3 = fut_pred_all
                fut_1_tmp = fut_1.permute(1, 0, 2)
                fut_1_tmp[:, :, 1] = fut_1_tmp[:, :, 1] + y_refer
                fut_1_abs = get_xy_from_nt_seq(fut_1_tmp.cpu().detach().numpy(), [centerline])
                traj_temp.append(fut_1_abs)
                fut_2_tmp = fut_2.permute(1, 0, 2)
                fut_2_tmp[:, :, 1] = fut_2_tmp[:, :, 1] + y_refer
                fut_2_abs = get_xy_from_nt_seq(fut_2_tmp.cpu().detach().numpy(), [centerline])
                traj_temp.append(fut_2_abs)

            traj_all_dict[seq_list[jj]]=traj_temp
            if i==100:
                with open("./100_multi_traj_2.pkl","wb") as f:
                    pkl.dump(traj_all_dict,f)
                print("finish")
                print(w)
    # with open("./all_multi_traj.pkl","wb") as f:
    #     pkl.dump(traj_all_dict,f)
    #     print("finish")


                # for traj in [fut_1_abs,fut_2_abs]:
                #     plt.plot(
                #         traj[0, :, 0],
                #         traj[0, :, 1],
                #         color="#007672",
                #         label="Predicted",
                #         alpha=1,
                #         linewidth=3,
                #         zorder=15,
                #     )
                #     plt.plot(
                #         traj[i, -1, 0],
                #         traj[i, -1, 1],
                #         "o",
                #         color="g",
                #         label="Predicted",
                #         alpha=1,
                #         linewidth=3,
                #         zorder=15,
                #         markersize=9,
                #     )


            # plt.show()
            # print(seq_list[i])
            # print(w)


    #         for current_cand_centerlines in cand_centerlines[num]:
    #             # print(fut_1)
    #             # print(current_y_refer[num:num+1,:])
    #             # print(current_cand_centerlines)
    #             print(current_cand_centerlines.shape)
    #             fut_1_abs = get_xy_from_nt_seq(fut_1_tmp.cpu().detach().numpy(), [current_cand_centerlines])
    #             traj_temp.append(fut_1_abs)
    #
    #             fut_2_abs = get_xy_from_nt_seq(fut_2_tmp.cpu().detach().numpy(), [current_cand_centerlines])
    #             traj_temp.append(fut_2_abs)
    #
    #         traj_all_dict[seq_list[num]] = traj_temp
    #
    #         for traj in traj_temp:
    #             plt.plot(
    #                 traj[0, :, 0],
    #                 traj[0, :, 1],
    #                 color="#007672",
    #                 label="Predicted",
    #                 alpha=1,
    #                 linewidth=3,
    #                 zorder=15,
    #             )
    #         plt.show()
    #         print(w)
    #
    # with open("./multi_test_result.pkl","wb")as f:
    #     pkl.dump(traj_all_dict,f)


