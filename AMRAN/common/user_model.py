import torch
import torch.nn as nn
from torch.autograd import Variable

import numpy as np

try:
    from common import hacnn
    from common import cnn_module
    from common import utils
except Exception as e:
    import hacnn
    import cnn_module
    import utils


class attenModel(nn.Module):
    def __init__(self, in_c, out_c, name, attend_f=True, out_conv=True, sa=False, h=1, w=1):#, k=3, s=1, p=0):
        super(attenModel, self).__init__()
        self.name = name
        self.attend_f = attend_f
        self.if_out_conv = out_conv
        self.sa = sa
        self.spatial_atten = hacnn.SpatialAttn()
        self.channel_atten = hacnn.ChannelAttn(in_channels=in_c, reduction_rate=in_c//2)
        if attend_f is True:
            self.atten_conv = cnn_module.SigmoidConv2d(in_c=in_c, out_c=out_c, k=1, s=1, p=0)
        if out_conv is True:
            self.out_conv = cnn_module.MaxConv2d(in_c=in_c, out_c=out_c, k=3, s=1, p=1)
        if sa is True:
            self.sa_model = SaModel(c=out_c, h=h, w=w, name=name)
        

    def forward(self, x):
        spatial = self.spatial_atten(x) # (N, 1, f_num, embed)
        channel = self.channel_atten(x) # (N, item/user_num, 1, 1)
        f_x = spatial * channel # (N, item/user_num, f_num, embed_dim)
        if self.attend_f is True:
            f_x_atten = self.atten_conv(f_x)
            #f_x = f_x * f_x_atten
            f_x = x * f_x_atten
        #if self.sa is True:
        #    f_x = self.sa_model(f_x)
        #    x = torch.sum(torch.sum(x, dim=1), dim=1)
        '''if debug and user_idxs.shape[0] == 15000:
            print('test friend: ------------------------')
            print(friend_user_features_embed[0:5, :, :, :])
            print('test friend f:')
            print(friend_users_f_x[0:5, :])'''
        if self.if_out_conv:
            x = self.out_conv(f_x) # (N, embed)
        else:
            x = x.view(x.shape[0], x.shape[1], x.shape[2]*x.shape[3])
            #print (self.name, x.shape)
        return x


class SaModel(nn.Module):
    def __init__(self, c, h, w, name):
        super(SaModel, self).__init__()
        self.name = name
        self.l1 = nn.Linear(w, h)
        utils.init_linear(self.l1)
        self.l2 = nn.Linear(h*w, c)
        utils.init_linear(self.l2)
        #self.lq = nn.Linear(w, w)
        #utils.init_linear(self.lq)
        #self.lk = nn.Linear(w, w)
        #utils.init_linear(self.lk)
        

    # x: (N, c, h, w)
    '''def forward(self, x):
        N, c, h, w = x.shape
        atten = torch.relu(self.l1(x)) # (N, c, h, h)
        atten_weight = torch.softmax((torch.sum(atten, dim=3) / np.sqrt(w*h)), dim=2).view((N, c, h, 1))
        x = x * atten_weight
        return x

    def forward(self, x):
        N, c, h, w = x.shape
        xq = torch.relu(self.lq(x)) # (n, c, h, w)
        xk = torch.relu(self.lk(x)) # (n, c, h, w)
        atten = torch.bmm(xq.view(N*c, h, w), xk.view(N*c, h, w).permute(0, 2, 1)) / np.sqrt(w)  # (N*c, h, h)
        atten_weight = torch.softmax(atten.view(N*c, h*h), dim=1).view(N*c, h, h)
        x = torch.bmm(atten_weight, x.view(N*c, h, w)).view(N, c, h, w)
        return x  '''

    def forward(self, x):
        N, c, h, w = x.shape
        atten = torch.relu(self.l1(x)) # (N, c, h, h)
        atten_weight = torch.softmax((torch.sum(atten, dim=3) / np.sqrt(h)), dim=2).view((N, c, h, 1))
        atten2 = torch.relu(self.l2(x.view(N, c, h*w)))  # (N, c, c)
        atten2_weight = torch.softmax(torch.sum(atten2, dim=2) / np.sqrt(c), dim=1).view(N, c, 1, 1)
        x = x * atten_weight * atten2_weight
        return x
