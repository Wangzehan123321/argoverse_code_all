import torch
import torch.nn as nn
import math

class ScaledDotProductAttention(nn.Module):

    def __init__(self,input_size=2,embed_size=32,dropout=0):
        super(ScaledDotProductAttention, self).__init__()
        self.q_linear=nn.Linear(input_size,embed_size)
        self.k_linear=nn.Linear(input_size,embed_size)
        self.v_linear=nn.Linear(input_size,embed_size)
        self.softmax = nn.Softmax(dim=-1)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, mask=None):
        """
        Q/K/V [batch_size, time_step, d_model]
        Args:
        Q: queue matrix
        K: key matrix
        V: value matrix
        QK^T:[batch_size, q_time_step, d_model]X[batch_size, d_model, k_time_step]
                        =[batch_size, q_time_step, k_time_step]
        """
        q=self.q_linear(x)
        k=self.k_linear(x)
        v=self.v_linear(x)
        attn = torch.bmm(q, k.transpose(1, 2)).div(math.sqrt(k.shape[-1]))

        if mask is not None:
            assert mask.size() == attn.size()
            attn.data.masked_fill_(mask, -float('inf'))

        attn_weights = self.softmax(attn)
        attn_weights = self.dropout(attn_weights)
        output = torch.bmm(attn_weights, v)
        #output = output+v
        output = torch.cat((v,output),2)

        # return output, attn_weights
        return output

#TODO:交互模块
class PoolHiddenNet(nn.Module):
    """Pooling module as proposed in social gan"""
    def __init__(self, embedding_dim=64, h_dim=64):
        super(PoolHiddenNet, self).__init__()
        self.h_dim = h_dim
        self.embedding_dim = embedding_dim
        self.relu=nn.ReLU(inplace=True)
        self.spatial_embedding = nn.Linear(2,self.embedding_dim)
        self.mlp_pre_pool =nn.Sequential(nn.Linear(self.embedding_dim+self.h_dim,self.h_dim),nn.ReLU(inplace=True))

    def repeat(self, tensor, num_reps):
        """
        Inputs:
        -tensor: 2D tensor of any shape
        -num_reps: Number of times to repeat each row
        Outpus:
        -repeat_tensor: Repeat each row such that: R1, R1, R2, R2
        """
        col_len = tensor.size(1)
        tensor = tensor.unsqueeze(dim=1).repeat(1, num_reps, 1)
        tensor = tensor.view(-1, col_len)
        return tensor

    def forward(self,hist_enc,hist_pos_batch,seq_start_end):#需要添加seq_start_end,周围车辆的hist_enc以及hist_pos
        pool_h = []
        for i in range(len(seq_start_end)-1):
            start = seq_start_end[i]
            end = seq_start_end[i+1]
            num_ped = end - start
            hist_batch_num = hist_enc[start:end]
            hist_pos_num = hist_pos_batch[start:end]
            # # Repeat -> H1, H2, H1, H2
            curr_hidden_1 = hist_batch_num.repeat(num_ped, 1)
            # # Repeat position -> P1, P2, P1, P2
            curr_end_pos_1 = hist_pos_num.repeat(num_ped, 1)
            # # Repeat position -> P1, P1, P2, P2
            curr_end_pos_2 = self.repeat(hist_pos_num, num_ped)
            curr_rel_pos = curr_end_pos_1 - curr_end_pos_2
            curr_rel_embedding = self.relu(self.spatial_embedding(curr_rel_pos))
            mlp_h_input = torch.cat([curr_hidden_1,curr_rel_embedding], dim=1)
            curr_pool_h = self.mlp_pre_pool(mlp_h_input)
            curr_pool_h = curr_pool_h.view(num_ped, num_ped, -1).max(1)[0]
            pool_h.append(curr_pool_h[0:1])
        pool_h = torch.cat(pool_h, dim=0)
        return pool_h

class SAICNet(nn.Module):

    ## Initialization
    def __init__(self,args):
        super(SAICNet, self).__init__()
        self.args = args
        self.use_cuda = args['use_cuda']#True
        # Flag for maneuver based (True) vs uni-modal decoder (False)
        self.use_maneuvers = args['use_maneuvers']#True

        ## Define size
        self.in_length=args['in_length']
        self.out_length=args['out_length']

        ## Sizes of network layers
        self.encoder_size = args['encoder_size']#64
        self.decoder_size = args['decoder_size']#64
        self.input_embedding_size = args['input_embedding_size']#32
        self.pool_embedding_size=args['pool_embedding_size']#64
        self.num_lon_classes = args['num_lon_classes']#3

        ## consider interaction or not
        self.social_input = args["social_input"]#default false
        self.vehicle_input = args["vehicle_input"]#default false
        #self.independ_lstm = args["independ_lstm"]#default false
        self.interaction = args['interaction']#default false
        self.road_feature=args["road_feature"]#default false
        #use self attention or not
        self.attention=args["self_attention"]#default false

        ## Define network weights

        # Input embedding layer
        if self.social_input:
            #self.ip_emb = torch.nn.Linear(5, self.input_embedding_size)
            if self.attention:
                self.atten_emb_s=ScaledDotProductAttention(3,self.input_embedding_size)
                self.enc_lstm_s = torch.nn.LSTM(2*self.input_embedding_size, self.encoder_size, 1)
            else:
                self.ip_emb_s = torch.nn.Linear(3, self.input_embedding_size)
                self.enc_lstm_s = torch.nn.LSTM(self.input_embedding_size, self.encoder_size, 1)

        if self.attention:
            self.atten_emb = ScaledDotProductAttention(2, self.input_embedding_size)
            self.enc_lstm = torch.nn.LSTM(2*self.input_embedding_size, self.encoder_size, 1)
        else:
            self.ip_emb = torch.nn.Linear(2, self.input_embedding_size)
            self.enc_lstm = torch.nn.LSTM(self.input_embedding_size, self.encoder_size, 1)

        if self.vehicle_input:
            if self.attention:
                self.atten_emb_v = ScaledDotProductAttention(2, self.input_embedding_size)
                self.enc_lstm_v = torch.nn.LSTM(2*self.input_embedding_size, self.encoder_size, 1)
            else:
                self.ip_emb_v=torch.nn.Linear(2,self.input_embedding_size)
                self.enc_lstm_v=torch.nn.LSTM(self.input_embedding_size,self.encoder_size,1)

        if self.road_feature:
            if self.attention:
                self.atten_emb_r = ScaledDotProductAttention(2, self.input_embedding_size)
                self.enc_lstm_r = torch.nn.LSTM(2*self.input_embedding_size, self.encoder_size, 1)
            else:
                self.ip_emb_r = torch.nn.Linear(2, self.input_embedding_size)
                self.enc_lstm_r = torch.nn.LSTM(self.input_embedding_size, self.encoder_size, 1)

        #Interaction Model
        if self.interaction:
            self.pool=PoolHiddenNet(embedding_dim=self.pool_embedding_size,h_dim=self.encoder_size)

        # Decoder LSTM
        if self.interaction:
            self.dec_hidden=self.encoder_size*2
        else:
            self.dec_hidden=self.encoder_size

        if self.vehicle_input:
            self.dec_hidden+=self.encoder_size

        if self.social_input:
            self.dec_hidden+=self.encoder_size

        if self.road_feature:
            self.dec_hidden+=self.encoder_size

        if self.use_maneuvers:
            self.dec_lstm=torch.nn.LSTM(self.dec_hidden+self.num_lon_classes,self.decoder_size,1)
            self.op_lon = torch.nn.Linear(self.dec_hidden,self.num_lon_classes)
            self.softmax = torch.nn.Softmax(dim=1)
        else:
            self.dec_lstm=torch.nn.LSTM(self.dec_hidden,self.decoder_size,1)

        # Output layers:
        self.op = torch.nn.Linear(self.decoder_size,2)#if MSE 2 else NLL 5

        # Activations:
        self.relu = torch.nn.ReLU()

    ## Forward Pass
    def forward(self,hist,hist_v=None,hist_s=None,road_feature=None,hist_pos=None,seq_start_end=None):
        if self.interaction:
            pass#TODO
            # 注意lstm的输入要求是(seq,batch,2)
            if self.attention:
                ip_embed_hist=self.atten_emb(hist.transpose(1,0)).transpose(1,0)
            else:
                ip_embed_hist=self.ip_emb(hist)
            _, (hist_enc, _) = self.enc_lstm(self.relu(ip_embed_hist))#注意这里的历史编码不再只包含自车，还包括周围车辆。
            hist_enc = hist_enc.view(hist_enc.shape[1], hist_enc.shape[2])
            soc_enc=self.pool(hist_enc,hist_pos,seq_start_end)
            enc = torch.cat((soc_enc, hist_enc[seq_start_end[0:-1], :]), 1)
            if self.use_maneuvers:
                lon_pred = self.softmax(self.op_lon(enc))
                fut_pred=[]
                for k in range(self.num_lon_classes):
                    lon_enc_tmp = torch.zeros(enc.shape[0],self.num_lon_classes)
                    if self.use_cuda:
                        lon_enc_tmp=lon_enc_tmp.cuda()
                    lon_enc_tmp[:, k] = 1
                    enc_tmp = torch.cat((enc, lon_enc_tmp), 1)
                    fut_pred.append(self.decode(enc_tmp))
                return fut_pred,lon_pred
            fut_pred = self.decode(enc)
            return fut_pred
        else:
            ## Forward pass hist:
            #注意lstm的输入要求是(seq,batch,2)
            if self.attention:
                ip_embed_hist = self.atten_emb(hist.transpose(1, 0)).transpose(1, 0)
            else:
                ip_embed_hist = self.ip_emb(hist)
            _,(hist_enc,_) = self.enc_lstm(self.relu(ip_embed_hist))
            hist_enc = hist_enc.view(hist_enc.shape[1], hist_enc.shape[2])
            if self.vehicle_input:
                if self.attention:
                    ip_embed_hist_v = self.atten_emb_v(hist_v.transpose(1, 0)).transpose(1, 0)
                else:
                    ip_embed_hist_v = self.ip_emb_v(hist_v)
                _, (hist_enc_v, _) = self.enc_lstm_v(self.relu(ip_embed_hist_v))
                hist_enc_v = hist_enc_v.view(hist_enc_v.shape[1], hist_enc_v.shape[2])
                hist_enc=torch.cat((hist_enc,hist_enc_v),1)

            if self.social_input:
                if self.attention:
                    ip_embed_hist_s = self.atten_emb_s(hist_s.transpose(1, 0)).transpose(1, 0)
                else:
                    ip_embed_hist_s = self.ip_emb_s(hist_s)
                _, (hist_enc_s, _) = self.enc_lstm_s(self.relu(ip_embed_hist_s))
                hist_enc_s = hist_enc_s.view(hist_enc_s.shape[1], hist_enc_s.shape[2])
                hist_enc = torch.cat((hist_enc, hist_enc_s), 1)

            if self.road_feature:
                if self.attention:
                    ip_embed_road_feat = self.atten_emb_r(road_feature.transpose(1, 0)).transpose(1, 0)
                else:
                    ip_embed_road_feat = self.ip_emb_r(road_feature)
                _, (road_enc, _) = self.enc_lstm_r(self.relu(ip_embed_road_feat))
                road_enc = road_enc.view(road_enc.shape[1], road_enc.shape[2])
                hist_enc = torch.cat((hist_enc, road_enc), 1)

            if seq_start_end is not None:
                hist_enc=hist_enc[seq_start_end[0:-1], :]
            if self.use_maneuvers:
                lon_pred = self.softmax(self.op_lon(hist_enc))
                fut_pred = []
                for k in range(self.num_lon_classes):
                    lon_enc_tmp = torch.zeros(hist_enc.shape[0], self.num_lon_classes)
                    if self.use_cuda:
                        lon_enc_tmp = lon_enc_tmp.cuda()
                    lon_enc_tmp[:, k] = 1
                    enc_tmp = torch.cat((hist_enc, lon_enc_tmp), 1)
                    fut_pred.append(self.decode(enc_tmp))
                return fut_pred, lon_pred
            fut_pred = self.decode(hist_enc)
            return fut_pred

    def decode(self,enc):
        enc = enc.repeat(self.out_length, 1, 1)
        h_dec, _ = self.dec_lstm(enc)
        h_dec = h_dec.permute(1, 0, 2)
        fut_pred = self.op(h_dec)
        fut_pred = fut_pred.permute(1, 0, 2)
        return fut_pred

def init_weights(m):
    classname=m.__class__.__name__
    if classname.find("Linear")!=-1:
        nn.init.kaiming_normal_(m.weight,a=0,mode="fan_out", nonlinearity="leaky_relu")
        if hasattr(m,"bias") and m.bias is not None:
            nn.init.constant_(m.bias,0)
    if classname.find("Conv")!=-1:
        nn.init.kaiming_normal_(m.weight,a=0,mode="fan_out",nonlinearity="leaky_relu")
        if hasattr(m,"bias") and m.bias is not None:
            nn.init.constant_(m.bias,0)
