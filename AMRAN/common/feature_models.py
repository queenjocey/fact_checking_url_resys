import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim


import os
import sys
import math
import json
import random
import numpy as np
from datetime import datetime

from common import utils
from common import highway
from common import embed
from common import hacnn
from common import predict_test
from common import reader
from common import cnn_module
from common import user_model

CONSUM_NUM = 10
NEIGHB_URL_NUM = 10
FRIEND_NUM = 10#100
NEIGHT_USER_NUM = 10#40

F_DIM = 64


class FeatureModel(nn.Module):
    def __init__(self, out_conv=True, attend_f=True, sa=False, embed_drop_out=0.1
            ,num_limit_list=[CONSUM_NUM, NEIGHB_URL_NUM, FRIEND_NUM, NEIGHT_USER_NUM], F_DIM=F_DIM):
        super(FeatureModel, self).__init__()
        self.out_conv = out_conv
        self.attend_f = attend_f
        self.sa = sa

        self.user = nn.Linear(7*F_DIM, F_DIM)
        utils.init_linear(self.user)
        self.url = nn.Linear(5*F_DIM, F_DIM)
        utils.init_linear(self.url)
        #self.user_url = nn.Linear(F_DIM, F_DIM)
        #utils.init_linear(self.user_url)

        self.cos = nn.CosineSimilarity(dim=2, eps=1e-6)
        # url features: int(url_no), cate2idx[cate], site2idx[url_site], post_user_bin, post_freq_bin
        self.url_features_embed = nn.ModuleList()
        self.url_features_embed.append(embed.NormalEmbedding(4732+1, F_DIM, embed_drop_out))
        self.url_features_embed.append(embed.NormalEmbedding(20+1, F_DIM, embed_drop_out))
        self.url_features_embed.append(embed.NormalEmbedding(6+1, F_DIM, embed_drop_out))
        self.url_features_embed.append(embed.NormalEmbedding(10+1, F_DIM, embed_drop_out))
        self.url_features_embed.append(embed.NormalEmbedding(10+1, F_DIM, embed_drop_out))

        self.url_consume = user_model.attenModel(in_c=num_limit_list[0], out_c=num_limit_list[0], name='url_consume',
                    attend_f=self.attend_f, out_conv=self.out_conv, sa=self.sa, h=5, w=F_DIM)
        self.url_neighb = user_model.attenModel(in_c=num_limit_list[1], out_c=num_limit_list[1], name='url_neighb',
                    attend_f=self.attend_f, out_conv=self.out_conv, sa=self.sa, h=5, w=F_DIM)
        self.url_gate = nn.Linear(F_DIM*2, F_DIM)
        utils.init_linear(self.url_gate)


        # user features: int(uid_no), follow_cnt_bin, follow_in_cnt_bin, post_url_bin, post_cnt_bin ,favorite_cate_no, favorite_site_no
        self.user_features_embed = nn.ModuleList()
        self.user_features_embed.append(embed.NormalEmbedding(11576+1, F_DIM, embed_drop_out))
        self.user_features_embed.append(embed.NormalEmbedding(10+1, F_DIM, embed_drop_out))
        self.user_features_embed.append(embed.NormalEmbedding(10+1, F_DIM, embed_drop_out))
        self.user_features_embed.append(embed.NormalEmbedding(10+1, F_DIM, embed_drop_out))
        self.user_features_embed.append(embed.NormalEmbedding(10+1, F_DIM, embed_drop_out))
        self.user_features_embed.append(embed.NormalEmbedding(20+1, F_DIM, embed_drop_out))
        self.user_features_embed.append(embed.NormalEmbedding(6+1, F_DIM, embed_drop_out))
        
        self.user_friend = user_model.attenModel(in_c=num_limit_list[2], out_c=num_limit_list[2], name='user_friend', 
                        attend_f=self.attend_f, out_conv=self.out_conv, sa=self.sa, h=7, w=F_DIM)
        self.user_neighb = user_model.attenModel(in_c=num_limit_list[3], out_c=num_limit_list[3], name='user_neighb',
                        attend_f=self.attend_f, out_conv=self.out_conv, sa=self.sa, h=7, w=F_DIM)
        self.user_gate = nn.Linear(F_DIM*2, F_DIM)
        utils.init_linear(self.user_gate)

        #self.linear_out = nn.Linear(F_DIM*(2+0), 2)
        #utils.init_linear(self.linear_out)


    def get_user_or_url_embed(self, embed_module_list, f_list):
        N, f_num = f_list.shape
        features_embed = []
        for i in range(len(embed_module_list)):
            f_embed = embed_module_list[i](f_list[:, i].view(N))  # (N, embed_dim)
            #print(f_list.shape, f_list[:, i].shape, f_embed.shape, 'aaaaaaaaaa')
            features_embed.append(f_embed)
        features_embed = torch.cat(features_embed, dim=1) # (N, f_num*embed_dim)
        return features_embed

    
    def get_mul_channel_embed(self, embed_module_list, f_list):
        N, num, url_f_num = f_list.shape
        features_embed = []
        for i in range(len(embed_module_list)):
            f_embed = embed_module_list[i](f_list[:, :, i].view(N, num, 1))  # (N, item/user_num, 1, embed_dim)
            features_embed.append(f_embed)
        features_embed = torch.cat(features_embed, dim=2) # (N, item/user_num, f_num, embed_dim)
        return features_embed

    # 
    def forward(self, user_idxs, url_idxs, user_f_list, url_f_list
                ,consume_urls_f_list, neighb_urls_f_list
                ,friend_users_f_list, neighb_users_f_list):
                #user_ids, url_ids, url_word_idxs, url_char_idxs, tweets_word_idxs, tweets_char_idxs):
        '''
        user_idxs: torch.Size([128])
        url_idxs: torch.Size([128])
        labels: torch.Size([128])
        user_f_list: torch.Size([128, 7])
        url_f_list: torch.Size([128, 5])
        consume_urls_f_list: torch.Size([128, 3, 5])
        neighb_urls_f_list : torch.Size([128, 3, 5])
        friend_users_f_list: torch.Size([128, 3, 7])
        neighb_users_f_list: torch.Size([128, 3, 7]) 
        '''
        user_f = self.get_user_or_url_embed(self.user_features_embed, user_f_list)  # (N, 7*embed_dim)
        user_f = self.user(user_f)                                                  # (N, embed_dim)
        url_f = self.get_user_or_url_embed(self.url_features_embed, url_f_list)     # (N, 5*embed_dim)
        url_f = self.url(url_f)                                                     # (N, embed_dim)
        #user_url_f = torch.relu(self.user_url(user_f * url_f))
        user_f = torch.relu(user_f)
        url_f = torch.relu(url_f)


        if self.out_conv is True:
            consume_urls_features_embed = self.get_mul_channel_embed(self.url_features_embed, consume_urls_f_list) # (N, item/user_num, f_num, embed_dim)   
            consum_urls_f_x = self.url_consume(consume_urls_features_embed) # (N, embed)
        
            neighb_urls_features_embed = self.get_mul_channel_embed(self.url_features_embed, neighb_urls_f_list) # (N, item/user_num, f_num, embed_dim)
            neighb_urls_f_x = self.url_neighb(neighb_urls_features_embed)
           
            ### url feature add
            url_feature_x = torch.cat([consum_urls_f_x, neighb_urls_f_x], dim=1) # (N, dim*2)
            g_url = torch.sigmoid(self.url_gate(url_feature_x))
            url_feature_x = g_url * consum_urls_f_x + (1-g_url) * neighb_urls_f_x

            ###
            friend_user_features_embed = self.get_mul_channel_embed(self.user_features_embed, friend_users_f_list) # (N, item/user_num, f_num, embed_dim)
            friend_users_f_x = self.user_friend(friend_user_features_embed)

            ###
            neighb_user_features_embed = self.get_mul_channel_embed(self.user_features_embed, neighb_users_f_list) # (N, item/user_num, f_num, embed_dim)
            neighb_users_f_x = self.user_neighb(neighb_user_features_embed)

            # add user feature
            user_feature_x = torch.cat([friend_users_f_x, neighb_users_f_x], dim=1) # (N, dim*2)
            g_user = torch.sigmoid(self.user_gate(user_feature_x))
            user_feature_x = g_user * friend_users_f_x + (1-g_user) * neighb_users_f_x
        else:
            consume_urls_features_embed = self.get_mul_channel_embed(self.url_features_embed, consume_urls_f_list) # (N, item/user_num, f_num, embed_dim)   
            consume_urls_f_x = self.url_consume(consume_urls_features_embed) # (N, num, f*embed)
            consume_urls_f_x = self.url(consume_urls_f_x)  #(N, num, embed)
            consume_urls_f_x = torch.relu(consume_urls_f_x)
            url_weight1 = self.cos(consume_urls_f_x, url_f.unsqueeze(1))  # (N, num)
            #url_weight1 = torch.exp(url_weight)
            #url_weight1 = url_weight / torch.pow(torch.sum(url_weight, dim=1), 0.5).view(-1,1,1)
            url_weight1 = torch.softmax(url_weight1, dim=1).unsqueeze(2)  # (N, num, 1)
            #print(url_weight1[0:10, :, :])
            #print(consume_urls_f_x.shape, url_weight1.shape)
            consume_urls_f_x = consume_urls_f_x * url_weight1   # (N, num, embed)
            consume_urls_f_x = torch.sum(consume_urls_f_x, dim=1)


            neighb_urls_features_embed = self.get_mul_channel_embed(self.url_features_embed, neighb_urls_f_list) # (N, item/user_num, f_num, embed_dim)
            neighb_urls_f_x = self.url_neighb(neighb_urls_features_embed)
            neighb_urls_f_x = self.url(neighb_urls_f_x)
            neighb_urls_f_x = torch.relu(neighb_urls_f_x)
            url_weight2 = self.cos(neighb_urls_f_x, url_f.unsqueeze(1))  # (N, num)
            #url_weight2 = torch.exp(url_weight2)
            #url_weight2 = url_weight2 / torch.pow(torch.sum(url_weight2, dim=1), 0.5).view(-1,1,1)
            url_weight2 = torch.softmax(url_weight2, dim=1).unsqueeze(2)  # (N, num, 1)
            neighb_urls_f_x = neighb_urls_f_x * url_weight2   # (N, num, embed)
            neighb_urls_f_x = torch.sum(neighb_urls_f_x, dim=1)

            ### url feature add
            url_feature_x = consume_urls_f_x + neighb_urls_f_x  # (N, embed_dim)
            g_url = torch.sigmoid(self.url_gate(url_feature_x))
            url_feature_x = g_url * consume_urls_f_x + (1-g_url) * neighb_urls_f_x  # (N, embed_dim)


            ###
            friend_user_features_embed = self.get_mul_channel_embed(self.user_features_embed, friend_users_f_list) # (N, item/user_num, f_num, embed_dim)
            friend_users_f_x = self.user_friend(friend_user_features_embed)
            friend_users_f_x = self.user(friend_users_f_x)
            friend_users_f_x = torch.relu(friend_users_f_x)
            user_weight1 = self.cos(friend_users_f_x, user_f.unsqueeze(1)) # (N, num)
            #user_weight1 = torch.exp(user_weight1)
            #user_weight1 = user_weight / torch.pow(torch.sum(user_weight1, dim=1), 0.5).view(-1, 1, 1)
            user_weight1 = torch.softmax(user_weight1, dim=1).unsqueeze(2) 
            friend_users_f_x = friend_users_f_x * user_weight1
            friend_users_f_x = torch.sum(friend_users_f_x, dim=1)

            ###
            neighb_user_features_embed = self.get_mul_channel_embed(self.user_features_embed, neighb_users_f_list) # (N, item/user_num, f_num, embed_dim)
            neighb_users_f_x = self.user_neighb(neighb_user_features_embed)
            neighb_users_f_x = self.user(neighb_users_f_x)
            neighb_users_f_x = torch.relu(neighb_users_f_x)
            user_weight2 = self.cos(neighb_users_f_x, user_f.unsqueeze(1)) # (N, num)
            user_weight2 = torch.exp(user_weight2)
            #user_weight2 = user_weight2 / torch.pow(torch.sum(user_weight2, dim=1), 0.5).view(-1, 1, 1)
            user_weight2 = torch.softmax(user_weight2, dim=1).unsqueeze(2) 
            neighb_users_f_x = neighb_users_f_x * user_weight2
            neighb_users_f_x = torch.sum(neighb_users_f_x, dim=1)

            # add user feature
            user_feature_x = friend_users_f_x + neighb_users_f_x
            g_user = torch.sigmoid(self.user_gate(user_feature_x))
            user_feature_x = g_user * friend_users_f_x + (1-g_user) * neighb_users_f_x
        
        out = torch.cat([url_feature_x, user_feature_x, user_f, url_f], dim=1) # (N, embed*(2+2))
        return out
