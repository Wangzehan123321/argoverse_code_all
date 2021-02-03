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
from model_argo_road import SAICNet,init_weights
from Argo_road_utils import ArgoverseDataset,MSE,ADE,FDE,ADE_sequence,FDE_sequence
from torch.utils.data import DataLoader
# import time
import copy

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
args["vehicle_input"] = True
args['interaction'] = False
args["road_feature"]=True

# Initialize network
net = SAICNet(args)
#net.apply(init_weights)
#net.load_state_dict(torch.load('./SAIC/trained_models/SAIC_lstm/cslstm_38.tar'))
#net.load_state_dict(torch.load('./SAIC/trained_models/SAIC_lstm_absolute/cslstm_49.tar'))
#net.load_state_dict(torch.load('./SAIC/trained_models/argo_lstm_pretrain_SAIC/cslstm_28.tar'))
#net.load_state_dict(torch.load('./SAIC/trained_models/argo_lstm_socialinput_pretrain_SAIC/cslstm_43.tar'))
#net.load_state_dict(torch.load('./SAIC/trained_models/argo_lstm_relative_speed/cslstm_55.tar'))
#net.load_state_dict(torch.load('./SAIC/trained_models/argo_lstm_relative_relative/cslstm_45.tar'))
#net.load_state_dict(torch.load('./SAIC/trained_models/argo_lstm_relative_speed_smoothl1/cslstm_97.tar'))
#net.load_state_dict(torch.load('./SAIC/trained_models/argo_lstm_relative_speed/cslstm_97.tar'))
#net.load_state_dict(torch.load('./SAIC/trained_models/argo_lstm_pretrain_SAIC/cslstm_96.tar'))
#net.load_state_dict(torch.load('./SAIC/trained_models/argo_lstm_relative_interaction/cslstm_96.tar'))
#net.load_state_dict(torch.load('./SAIC/trained_models/argo_lstm_relative_speed_nll/cslstm_97.tar'))
#net.load_state_dict(torch.load('./SAIC/trained_models/argo_lstm_relative_absolute/cslstm_97.tar'))
# net.load_state_dict(torch.load('./SAIC/trained_models/argo_lstm_relative_speed_smoothl1_step/cslstm_89.tar'))
net.load_state_dict(torch.load('./SAIC/trained_models/argo_roadfeature_multi/cslstm_31.tar'))

if args['use_cuda']:
    net = net.cuda()

# Evaluation metric:
metric = 'rmse'  #输出每个点的均方误差

tsSet = ArgoverseDataset('../data/forecasting_features_val.pkl',road_feature=args["road_feature"],social_input=args["social_input"],vehicle_input=args["vehicle_input"],val_metric=True)
tsDataloader = DataLoader(tsSet,batch_size=1024,shuffle=True,num_workers=8,collate_fn=tsSet.collate_fn)

lossVals_ade = 0#预测50个点，点间隔为0.1s
lossVals_fde = 0
counts = 0

for i, data in enumerate(tsDataloader):
    # st_time = time.time()
    hist,hist_v,road_feature,fut,current_refer,oracle_ct = data
    # hist, fut, current_refer, oracle_ct = data
    # hist,fut=data

    # Initialize Variables
    if args['use_cuda']:
        hist = hist.cuda()
        hist_v = hist_v.cuda()
        road_feature=road_feature.cuda()
        fut = fut.cuda()
        current_refer=current_refer.cuda()

    if args['use_maneuvers']:
        #TODO:多模态实现
        fut_pred_all,lon_pred = net(hist,hist_v=hist_v,road_feature=road_feature)
        fut_1, fut_2, fut_3 = fut_pred_all
        fut_1 = fut_1.permute(1, 0, 2)
        fut_1[:, :, 1] = fut_1[:, :, 1] + current_refer
        fut_1_abs = get_xy_from_nt_seq(fut_1.cpu().detach().numpy(), oracle_ct)
        fut_1_abs = torch.from_numpy(fut_1_abs).permute(1, 0, 2)

        fut_2 = fut_2.permute(1, 0, 2)
        fut_2[:, :, 1] = fut_2[:, :, 1] + current_refer
        fut_2_abs = get_xy_from_nt_seq(fut_2.cpu().detach().numpy(), oracle_ct)
        fut_2_abs = torch.from_numpy(fut_2_abs).permute(1, 0, 2)

        fut_3 = fut_3.permute(1, 0, 2)
        fut_3[:, :, 1] = fut_3[:, :, 1] + current_refer
        fut_3_abs = get_xy_from_nt_seq(fut_3.cpu().detach().numpy(), oracle_ct)
        fut_3_abs = torch.from_numpy(fut_3_abs).permute(1, 0, 2)

        l_ade_1 = ADE_sequence(fut_1_abs.float().cuda(), fut)
        l_ade_2 = ADE_sequence(fut_2_abs.float().cuda(), fut)
        l_ade_3 = ADE_sequence(fut_3_abs.float().cuda(), fut)
        l_ade_all = torch.stack((l_ade_1, l_ade_2, l_ade_3), 1)
        l_ade_min, number = l_ade_all.min(1)
        l_ade=torch.mean(l_ade_min)

        l_fde_1 = FDE_sequence(fut_1_abs.float().cuda(), fut)
        l_fde_2 = FDE_sequence(fut_2_abs.float().cuda(), fut)
        l_fde_3 = FDE_sequence(fut_3_abs.float().cuda(), fut)
        l_fde_all = torch.stack((l_fde_1, l_fde_2, l_fde_3), 1)
        l_fde_min, number = l_fde_all.min(1)
        l_fde = torch.mean(l_fde_min)
    else:
        #hist_2=copy.deepcopy(hist)
        fut_pred = net(hist, hist_v=hist_v, road_feature=road_feature)
        # fut_pred = net(hist)
        fut_pred = fut_pred.permute(1, 0, 2)
        fut_pred[:,:,1]=fut_pred[:,:,1]+current_refer
        fut_pred_abs=get_xy_from_nt_seq(fut_pred.cpu().detach().numpy(),oracle_ct)
        fut_pred_abs=torch.from_numpy(fut_pred_abs).permute(1, 0, 2)
        l_ade = ADE(fut_pred_abs.float().cuda(), fut)
        l_fde = FDE(fut_pred_abs.float().cuda(), fut)
        # l_ade = ADE(fut_pred, fut)
        # l_fde = FDE(fut_pred, fut)

    lossVals_ade +=l_ade.detach()
    lossVals_fde += l_fde.detach()
    counts += 1

print(lossVals_ade / counts)
print(lossVals_fde / counts)