import pickle as pkl
pkl_file='../data/data_nerb/forecasting_features_train.pkl'
with open(pkl_file, "rb")as f:
    raw_file = pkl.load(f)
traj = raw_file["FEATURES"].values
nerb_info = raw_file["NEIGHBOUR"].values
for idx in range(len(traj)):
    target_pt=traj[idx][20-1,[3,4]]
    nerb_pt_list=nerb_info[idx]["neighbour_pt"]
    print(target_pt)
    print(nerb_pt_list)
    print(w)