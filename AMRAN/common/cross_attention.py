import torch
import torch.nn as nn
from torch.autograd import Variable
from common import utils


class crossAttention(nn.Module):
    def __init__(self, x1_dim=64, x2_dim=64):
        super(crossAttention, self).__init__()
        self.x1_dim = x1_dim
        self.x2_dim = x2_dim

        ### x linear
        '''self.linear_w1 = nn.Linear(x1_dim, 1, bias=False)
        utils.init_linear(self.linear_x1)
        self.linear_w2 = nn.Linear(x1_dim, 1, bias=False)
        utils.init_linear(self.linear_w2)
        self.linear_w12 = nn.Linear(x1_dim, 1, bias=False)
        utils.init_linear(self.linear_w12)'''

        self.w1 = nn.Parameter(torch.randn(x1_dim, x2_dim), requires_grad=True)
        torch.nn.init.normal_(self.w1, mean=0, std=0.1)
        self.w2 = nn.Parameter(torch.randn(x1_dim, x2_dim), requires_grad=True)
        torch.nn.init.normal_(self.w2, mean=0, std=0.1)
        self.w12 = nn.Parameter(torch.randn(x1_dim, x2_dim), requires_grad=True)
        torch.nn.init.normal_(self.w12, mean=0, std=0.1)



    def forward(self, x1, x2):
        '''
            input:
                x1  (N, x1_dim)
                x2  (N, x2_dim)
            out:
                f([x1, x2_update_1, x1_update_1]
        '''
        N = x1.shape[0]
        x12 = torch.bmm(x1.unsqueeze(2), x2.unsqueeze(1))   # (N, x1_dim, x2_dim)

        sim = self.w1 * x1.unsqueeze(2).expand(N, self.x1_dim, self.x2_dim) + \
            self.w2 * x2.unsqueeze(1).expand(N, self.x1_dim, self.x2_dim)  + \
            self.w12 * x12                                       # (N, x1_dim, x2_dim)

        x2_sim = torch.softmax(sim, dim=2)                  # (N, x1_dim, x2_dim)
        x1_sim = torch.max(sim, dim=2)[0]                   # (N, x1_dim)

        x2_update_1 = x2_sim * x2.view(N, 1, self.x2_dim)    # (N, x1_dim, x2_dim)
        x2_update_1 = torch.sum(x2_update_1, dim=2)     # (N, x1_dim)
        x1_update_1 = x1_sim * x1                       # (N, x1_dim)

        out = torch.cat([x1, x2_update_1, x1*x2_update_1, x1*x1_update_1], dim=1)  # (N, x1_dim*4)
        return out
