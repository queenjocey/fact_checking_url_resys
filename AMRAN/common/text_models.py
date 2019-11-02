import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim


import os
import sys
import math
import random
import json
import numpy as np
from datetime import datetime

try:
    from common import utils
    from common import highway
    from common import embed
    from common import predict_test
except Exception as e:
    import utils
    import highway
    import embed
    import predict_test



class TextModel(nn.Module):
    def __init__(self, word_embed_mat, dim=64):
        super(TextModel, self).__init__()
        ### tweets and url content
        if word_embed_mat is None:
            self.word_embed = embed.NormalEmbedding(231172, 300, 0.1)
        else:
            self.word_embed = embed.FixedEmbedding(word_embed_mat, 0, False)
        
        self.char_embed = embed.CharacterEmbedding(num_embed=70, embed_dim=64, out_ch=64, kernel_size=3, padding=1)
        #self.char_embed = embed.CharacterDepthwiseEmbedding(num_embed=70, embed_dim=64, out_ch=64, kernel_size=3)
        #self.char_embed2 = embed.CharacterEmbedding(num_embed=70, embed_dim=64, out_ch=64, kernel_size=5, padding=2)

        #self.projection = nn.Conv1d(300+64, 128, 1)
        self.hw1 = highway.hw(300+64, 2)
        #self.hw2 = highway.hw(300+64, 2)
        '''self.lstm = nn.LSTM(         # if use nn.RNN(), it hardly learns
            input_size=128,
            hidden_size=64,         # rnn hidden unit
            num_layers=1,           # number of rnn layer
            batch_first=True,       # input & output will has batch size as 1s dimension. e.g. (batch, time_step, input_size)
            bidirectional=True
        )'''

        ### attention
        self.cos = torch.nn.CosineSimilarity(dim=2, eps=1e-6)
        #self.linear_attention = nn.Linear(300+64, 300+64)
        #utils.init_linear(self.linear_attention)

        #self.linear_attention_url = nn.Linear(300+64, 300+64)
        #utils.init_linear(self.linear_attention_url)

        self.linear = nn.Linear(300+64, 64)
        utils.init_linear(self.linear)


    # 
    def forward(self, url_word_idxs, url_char_idxs, tweets_word_idxs, tweets_char_idxs):
        '''
            user_embed              (1280,1)
            url_embed               (1280,1)
            url_word_idx            (1280, 200)
            url_char_idx            (1280, 200, 16)
            tweets_word_idx         (1280, 50, 30)
            tweets_char_idx         (1280, 50, 30, 16)
        '''
        debug = False

        ### url content
        if len(url_word_idxs.shape) == 3:
            url_word_idxs = url_word_idxs.view(url_word_idxs.shape[0], -1)
            url_char_idxs = url_char_idxs.view(url_char_idxs.shape[0], -1, 16)
        url_char_embeds = self.char_embed(url_char_idxs)     # (N, word_limit, char_dim)
        if debug:
            print('url_char_embeds:--------------------------------------')
            print(url_char_embeds[0, 2, :])
            print(url_char_embeds[1, 2, :])
        url_word_embeds = self.word_embed(url_word_idxs)     # (N, word_limit, word_dim)
        if debug:
            print('url_word_embeds:--------------------------------------')
            print(url_word_embeds[0, 2, :])
            print(url_word_embeds[1, 2, :])
        url_content_embeds = torch.cat([url_word_embeds, url_char_embeds], dim=2) # (N, word_limit, word_dim+char_dim)
        if debug:
            print('url_cat 5:--------------------------------------')
            print(url_content_embeds[0, 2:5, :])
            print(url_content_embeds[1, 2:5, :])
        #url_content_embeds = self.projection(url_content_embeds.permute(0, 2, 1))  # (N, hidden_dim, word_limit)
        url_content_embeds = torch.mean(url_content_embeds, dim=1)  # (N, hidden_dim)
        if debug:
            print('url_sum 2:--------------------------------------')
            print(url_content_embeds[0, :])
            print(url_content_embeds[1, :])
        #url_content_embeds, (h_n, h_c) = self.lstm(url_content_embeds.permute(0, 2, 1), None) # (N, word_limit, hidden_dim)
        #url_content_embeds = url_content_embeds[:, -1, :]  # (N, hidden_dim)
        url_content_embeds = self.hw1(url_content_embeds)   # (N, hidden_dim)


        ### tweets content
        tc_shape = tweets_char_idxs.shape
        tweets_char_idxs = tweets_char_idxs.view(tc_shape[0]*tc_shape[1], tc_shape[2], tc_shape[3])
        tweets_char_embeds = self.char_embed(tweets_char_idxs) #  (N * tweet_cnt, word_limit, char_dim)
        #tweets_char_embeds = tweets_char_embeds.view(tc_shape[0], tc_shape[1], tc_shape[2], -1) #  N * tweet_cnt * word_limit * char_dim
        #tweets_char_embeds = torch.sum(tweets_char_embeds, dim=3)   # 1 * tweet_cnt * word_limit  * char_dim
        tweets_word_embeds = self.word_embed(tweets_word_idxs).view(tc_shape[0]*tc_shape[1], tc_shape[2], -1)   # (N * tweet_cnt, word_limit, word_dim)
        tweets_content_embeds = torch.cat([tweets_word_embeds, tweets_char_embeds], dim=2) # (N * tweet_cnt, word_limit, word_dim+char_dim)
        #tweets_content_embeds = self.projection(tweets_content_embeds.permute(0, 2, 1))  # (N*tweet_cnt, hidden_dim, word_limit)
        tweets_content_embeds = torch.mean(tweets_content_embeds, dim=1) # (N * tweet_cnt, hidden_dim)
        #tweets_content_embeds, (h_n, h_c) = self.lstm(tweets_content_embeds.permute(0, 2, 1), None) #(N*tweet_cnt, word_limit, hidden_dim)
        #tweets_content_embeds = tweets_content_embeds[:, -1, :] # (N*tweet_cnt, hidden_dim)
        tweets_content_embeds = self.hw1(tweets_content_embeds)  # (N*tweet_cnt, hidden_dim)
        tweets_content_embeds = tweets_content_embeds.view(tc_shape[0], tc_shape[1], -1) # (N, tweet_cnt, hidden_dim)


        ### attention
        url_content_embeds = url_content_embeds.unsqueeze(1)  # (N, 1, hidden_dim)
        # method1
        
        url_tweets_weight = self.cos(tweets_content_embeds, url_content_embeds)  # (N, tweet_cnt)
        #url_tweets_weight = torch.exp(url_tweets_weight)
        #url_tweets_weight = url_tweets_weight / torch.pow(torch.sum(url_tweets_weight, dim=1), 0.5).view(-1,1)
        url_tweets_weight = torch.softmax(url_tweets_weight, dim=1)  # (N, tweet_cnt)
        '''
        # method2
        url_tweets_weight = torch.tanh(self.linear_attention(tweets_content_embeds)) \
                            * torch.tanh(self.linear_attention_url(url_content_embeds))  # (N, tweet_cnt, hiddem_dim)
        url_tweets_weight = torch.softmax(torch.mean(url_tweets_weight, dim=2), dim=1)  # (N, tweet_cnt)
        '''
        if debug:
            print('url tweet weight ---------------', url_tweets_weight.shape)
            print(url_tweets_weight[0, :])
            print(url_tweets_weight[1, :])

        tweets_content_embeds = tweets_content_embeds * url_tweets_weight.unsqueeze(2)  # (N, tweet_cnt, hidden_dim)
        tweets_content_embeds = torch.mean(tweets_content_embeds, dim=1)  # (N, hidden_dim)

        # linear
        tweets_url_content = self.linear(tweets_content_embeds*url_content_embeds.squeeze(1))    # (N, hidden_dim)
        #tweets_url_content = self.linear2(tweets_content_embeds.squeeze(1))    # (N, hidden_dim)

        return torch.relu(tweets_url_content)
