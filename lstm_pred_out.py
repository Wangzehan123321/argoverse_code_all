# import torch as t
#
# def LSTM_result(input):
#     list_all=[]
#     a=t.Tensor(30,2).tolist()
#     b=t.Tensor(30,2).tolist()
#     list_all.append(a)
#     list_all.append(b)
#     return list_all
# import numpy as np
# a=np.array([[1,2],[3,4],[5,6]])
# s=list(np.argwhere(a[:,1]>2).squeeze(1))
# print(list(np.argwhere(a[:,1]>2).squeeze(1)))
# b=np.array([5,4,7,2])
# print(np.sort(b))
# print(np.argmax(a[s,0]))

import torch.nn as nn
import torch

input=torch.Tensor(64,1,7,7)
transconv=nn.ConvTranspose2d(1,16,4,2,1)
print(transconv(input).shape)

