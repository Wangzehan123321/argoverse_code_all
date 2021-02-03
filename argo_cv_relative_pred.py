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
from SAIC_utils import ArgoverseDataset,ADE,FDE
from torch.utils.data import DataLoader
tsSet = ArgoverseDataset('../data/forecasting_features_val.pkl',absolute=False,val_metric=True,save_pkl=False)
tsDataloader = DataLoader(tsSet,batch_size=1,shuffle=True,num_workers=8,collate_fn=tsSet.collate_fn)

lossVals_ade = 0
lossVals_fde = 0
counts = 0

# #结果：ADE：4.1588。FDE：8.5686
# for i, data in enumerate(tsDataloader):
#     # st_time = time.time()
#     hist, fut, hist_ref = data
#     hist = hist.cuda()
#     fut = fut.cuda()
#     hist_ref = hist_ref.cuda()
#
#     v_current=hist[-1,:,:]-hist[-2,:,:]
#     cv_fut_pred=torch.zeros_like(fut)
#     for i in range(fut.shape[0]):
#         cv_fut_pred[i,:,:]=hist[-1,:,:]+(i+1)*v_current
#
#     # cv_fut_pred[:, :, :] = cv_fut_pred[:, :, :] + hist_ref.unsqueeze(0)
#     # print(cv_fut_pred)
#     # print(fut)
#     # print(w)
#     l_ade = ADE(cv_fut_pred.float().cuda(), fut)
#     l_fde = FDE(cv_fut_pred.float().cuda(), fut)
#
#     lossVals_ade += l_ade.detach()
#     lossVals_fde += l_fde.detach()
#     counts += 1
#
# print(lossVals_ade / counts)
# print(lossVals_fde / counts)

#     cv_fut_pred = cv_fut_pred.permute(1, 0, 2)
#     cv_fut_pred[:, :, 1] = cv_fut_pred[:, :, 1] + current_y_refer
#     fut_pred_abs = get_xy_from_nt_seq(cv_fut_pred.cpu().detach().numpy(), oracle_ct)
#     fut_pred_abs = torch.from_numpy(fut_pred_abs).permute(1, 0, 2)
#     l_ade = ADE(fut_pred_abs.float().cuda(), fut)
#     l_fde = FDE(fut_pred_abs.float().cuda(), fut)
#
#     lossVals_ade += l_ade.detach()
#     lossVals_fde += l_fde.detach()
#     counts += 1
#
# print(lossVals_ade / counts)
# print(lossVals_fde / counts)




##结果：ADE:4.0258。FDE：7.9050。
for i, data in enumerate(tsDataloader):
    # st_time = time.time()
    hist, fut, current_y_refer, oracle_ct = data
    hist = hist.cuda()
    fut = fut.cuda()
    current_y_refer = current_y_refer.cuda()

    #只计算y方向速度
    v_current=hist[-1,:,:]-hist[-2,:,:]
    v_current[:,0]=0

    start_pt=torch.zeros_like(hist[-1,:,:])
    start_pt[:,0]=0

    cv_fut_pred=torch.zeros_like(fut)
    for i in range(fut.shape[0]):
        cv_fut_pred[i,:,:]=start_pt+(i+1)*v_current
    cv_fut_pred = cv_fut_pred.permute(1, 0, 2)
    cv_fut_pred[:, :, 1] = cv_fut_pred[:, :, 1] + current_y_refer
    fut_pred_abs = get_xy_from_nt_seq(cv_fut_pred.cpu().detach().numpy(), oracle_ct)
    fut_pred_abs = torch.from_numpy(fut_pred_abs).permute(1, 0, 2)
    l_ade = ADE(fut_pred_abs.float().cuda(), fut)
    l_fde = FDE(fut_pred_abs.float().cuda(), fut)

    lossVals_ade += l_ade.detach()
    lossVals_fde += l_fde.detach()
    counts += 1

print(lossVals_ade / counts)
print(lossVals_fde / counts)
print(1)