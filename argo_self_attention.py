import math
import torch
import torch.nn as nn

class ScaledDotProductAttention(nn.Module):

    def __init__(self,dropout=0):
        super(ScaledDotProductAttention, self).__init__()
        self.q_linear=nn.Linear(2,32)
        self.k_linear=nn.Linear(2,32)
        self.v_linear=nn.Linear(2,32)
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
        output = output+v

        return output, attn_weights

# q_linear=nn.Linear(2,64)
# k_linear=nn.Linear(2,64)
# v_linear=nn.Linear(2,64)
input=torch.Tensor(32,128,2)

# input_q=q_linear(input)
# input_k=k_linear(input)
# input_v=v_linear(input)
# d_k=input_k.shape[-1]


self_attention=ScaledDotProductAttention()

output, attn_weights=self_attention(input)
print(output.shape)