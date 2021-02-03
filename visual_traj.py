from argoverse.data_loading.argoverse_forecasting_loader import ArgoverseForecastingLoader

root_dir="/home/wangzehan/argoverse-api/data/val/data"
afl = ArgoverseForecastingLoader(root_dir)

from argoverse.visualization.visualize_sequences import viz_sequence
import os
seq_path =os.path.join(root_dir,os.listdir(root_dir)[5])

from typing import Dict, List, Union
import pickle as pkl
import numpy as np
import pandas as pd
from argoverse.map_representation.map_api import ArgoverseMap
from argoverse.utils.mpl_plotting_utils import draw_lane_polygons, plot_bbox_2D
from shapely.geometry.polygon import Polygon
#from demo_usage.visualize_30hz_benchmark_data_on_map import DatasetOnMapVisualizer

am = ArgoverseMap()

x_min = min(afl.get(seq_path).seq_df["X"])
x_max = max(afl.get(seq_path).seq_df["X"])
y_min = min(afl.get(seq_path).seq_df["Y"])
y_max = max(afl.get(seq_path).seq_df["Y"])
# # print(x_min,x_max,y_min,y_max)
local_das = am.find_local_driveable_areas([x_min,x_max,y_min,y_max], str(afl.get(seq_path).seq_df["CITY_NAME"][0]))
local_lane_polygons = am.find_local_lane_polygons([x_min,x_max,y_min,y_max], str(afl.get(seq_path).seq_df["CITY_NAME"][0]))

# with open("/home/wangzehan/argoverse-forecasting/groundtruth/val_gt.pkl", "rb") as f:
#     gt_trajectories: Dict[int, np.ndarray] = pkl.load(f)

with open("/home/wangzehan/argoverse-forecast/save_pkl/argo_multi_fde.pkl", "rb") as f:
    forecasted_trajectories: Dict[int, List[np.ndarray]] = pkl.load(f)

with open("/home/wangzehan/argoverse-forecast/data/forecasting_features_val.pkl", "rb") as f:
    features_df: pd.DataFrame = pkl.load(f)
#
import matplotlib.pyplot as plt
from eval_forecasting_helper import viz_predictions_helper
# # # fig = plt.figure(figsize=(10,10))
# # # ax = fig.add_subplot(111)
# # # ax.set_xlim([x_min, x_max])
# # # ax.set_ylim([y_min, y_max])
plt.style.use('dark_background')
viz_sequence(afl.get(seq_path).seq_df, show=False)
# # # draw_lane_polygons(ax,local_das)
for i, polygon in enumerate(local_lane_polygons):
    plt.plot(polygon[:, 0], polygon[:, 1], color="y", alpha=1, zorder=1,lw=3)
for i, polygon in enumerate(local_das):
    plt.plot(polygon[:, 0], polygon[:, 1], color="r", alpha=1, zorder=1,lw=3)
#
seq_id = int(os.listdir(root_dir)[5].split(".")[0])

# gt_trajectory = gt_trajectories[seq_id]

curr_features_df = features_df[features_df["SEQUENCE"] == seq_id]

print(curr_features_df)
print(curr_features_df["FEATURES"].values)

input_trajectory = (curr_features_df["FEATURES"].values[0])[:20, [3, 4]].astype("float")

gt_trajectory=(curr_features_df["FEATURES"].values[0])[20:, [3, 4]].astype("float")

output_trajectories = forecasted_trajectories[seq_id]

# gt_trajectory = np.expand_dims(gt_trajectory, 0)
input_trajectory = np.expand_dims(input_trajectory, 0)

# output_trajectories = np.expand_dims(np.array(output_trajectories), 0)

num_tracks = input_trajectory.shape[0]
obs_len = input_trajectory.shape[1]
# print(gt_trajectory)
# pred_len = gt_trajectory.shape[1]

# plt.figure(0, figsize=(8, 7))

avm = ArgoverseMap()
for i in range(num_tracks):
    plt.plot(
        input_trajectory[i, :, 0],
        input_trajectory[i, :, 1],
        color="#ECA154",
        label="Observed",
        alpha=1,
        linewidth=3,
        zorder=15,
    )

    plt.plot(
        input_trajectory[i, -1, 0],
        input_trajectory[i, -1, 1],
        "o",
        color="#ECA154",
        label="Observed",
        alpha=1,
        linewidth=3,
        zorder=15,
        markersize=9,
    )
    print(gt_trajectory.shape)
    gt_trajectory = gt_trajectory.reshape(1,gt_trajectory.shape[0],gt_trajectory.shape[1])
    plt.plot(
        gt_trajectory[i, :, 0],
        gt_trajectory[i, :, 1],
        color="#d33e4c",
        label="Target",
        alpha=1,
        linewidth=3,
        zorder=20,
    )
    plt.plot(
        gt_trajectory[i, -1, 0],
        gt_trajectory[i, -1, 1],
        "o",
        color="#d33e4c",
        label="Target",
        alpha=1,
        linewidth=3,
        zorder=20,
        markersize=9,
    )

    for j in range(len(output_trajectories)):
            plt.plot(
                np.array(output_trajectories[j][:, 0]),
                np.array(output_trajectories[j][:, 1]),
                color="#007672",
                label="Predicted",
                alpha=1,
                linewidth=3,
                zorder=15,
            )
            plt.plot(
                output_trajectories[j][-1, 0],
                output_trajectories[j][-1, 1],
                "o",
                color="#007672",
                label="Predicted",
                alpha=1,
                linewidth=3,
                zorder=15,
                markersize=9,
            )
            print(1)



# viz_predictions_helper(forecasted_trajectories, gt_trajectories,
#                                features_df, id_for_viz)
#
plt.show()