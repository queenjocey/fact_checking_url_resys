"""
.. module:: attention
 
.. moduleauthor:: dyou
"""

import torch
import torch.nn as nn
from common import utils


class memAttention(nn.Module):
    """
    args: 
    
    """
   
    def __init__(self, x_dim=64, mem_dim=364, out_dim=64, attention_method='cos'):
        super(memAttention, self).__init__()
        self.method = attention_method
        ### x linear
        self.linear_x = nn.Linear(x_dim, mem_dim)
        utils.init_linear(self.linear_x)

        ### attention
        if self.method == 'cos':
            self.cos = torch.nn.CosineSimilarity(dim=2, eps=1e-6)
        else:
            self.linear_attention = nn.Linear(mem_dim, mem_dim)
            utils.init_linear(self.linear_attention)

        ### out
        self.linear_out = nn.Linear(mem_dim, out_dim)
        utils.init_linear(self.linear_out)


    def forward(self, x, mem):
        """
        args: 
            x   (ins_num, x_dim)
            mem (ins_num, slot_num, men_dim) 
        return:
            output tensor (ins_num, out_dim)
        """
        if self.method == 'cos':
            x = torch.relu(self.linear_x(x))  # (N, mem_dim)
        else:
            x = torch.tanh(self.linear_x(x))  # (N, mem_dim)
        ### attention
        x = x.unsqueeze(1)  # (N, 1, mem_dim)
        #print(x.shape, 'xxx')
        # method1
        if self.method == 'cos':
            mem_weight = self.cos(mem, x)  # (N, slot_num)
            mem_weight = torch.softmax(mem_weight, dim=1)  # (N, slot_num)
        else:
            # method2
            mem_weight = torch.tanh(self.linear_attention(mem)) * x           # (N, slot_num, hiddem_dim)
            mem_weight = torch.softmax(torch.mean(mem_weight, dim=2), dim=1)  # (N, slot_num)
        
        
        mem = mem * mem_weight.unsqueeze(2)  # (N, slot_num, mem_dim)
        mem = torch.mean(mem, dim=1) # (N, mem_dim)

        out = self.linear_out(mem * x.squeeze(1))  # (N, out_dim)
        out = torch.relu(out)   # (N, out_dim)
        return out
