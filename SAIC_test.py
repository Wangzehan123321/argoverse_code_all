import torch
from model_SAIC import SAICNet,init_weights
from SAIC_utils import SAICDataset,maskedMSE,maskedMSETest
from torch.utils.data import DataLoader
# import time

## Network Arguments
args = {}
args['use_cuda'] = True
args['encoder_size'] = 64
args['decoder_size'] = 64
args['in_length'] = 30
args['out_length'] = 50
args['input_embedding_size'] = 32
args['num_lon_classes'] = 3
args['use_maneuvers'] = False

# Initialize network
net = SAICNet(args)
#net.apply(init_weights)
#net.load_state_dict(torch.load('./SAIC/trained_models/SAIC_lstm/cslstm_38.tar'))
#net.load_state_dict(torch.load('./SAIC/trained_models/SAIC_lstm_absolute/cslstm_49.tar'))
net.load_state_dict(torch.load('./SAIC/trained_models/SAIC_lstm_absolute_argument/cslstm_99.tar'))
if args['use_cuda']:
    net = net.cuda()

# Evaluation metric:
metric = 'rmse'  #输出每个点的均方误差

tsSet = SAICDataset('./SAIC/train_new.pkl')
tsDataloader = DataLoader(tsSet,batch_size=128,shuffle=True,num_workers=8,collate_fn=tsSet.collate_fn)

lossVals = torch.zeros(50).cuda()#预测50个点，点间隔为0.1s
counts = torch.zeros(50).cuda()

for i, data in enumerate(tsDataloader):
    # st_time = time.time()
    hist, fut, op_mask = data

    # Initialize Variables
    if args['use_cuda']:
        hist = hist.cuda()
        fut = fut.cuda()
        op_mask = op_mask.cuda()

    if args['use_maneuvers']:
        #TODO:多模态实现
        fut_pred, lat_pred, lon_pred = net(hist)
        fut_pred_max = torch.zeros_like(fut_pred[0])
        for k in range(lat_pred.shape[0]):
            lat_man = torch.argmax(lat_pred[k, :]).detach()
            lon_man = torch.argmax(lon_pred[k, :]).detach()
            indx = lon_man*3 + lat_man
            fut_pred_max[:,k,:] = fut_pred[indx][:,k,:]
        l, c = maskedMSETest(fut_pred_max, fut, op_mask)
    else:
        fut_pred = net(hist)
        l, c = maskedMSETest(fut_pred, fut, op_mask)
    lossVals +=l.detach()
    counts += c.detach()

print(lossVals / counts)
#print(torch.pow(lossVals / counts,0.5))   # Calculate RMSE and convert from feet to meters