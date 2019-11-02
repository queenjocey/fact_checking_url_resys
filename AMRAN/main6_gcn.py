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
from common import embed
from common import predict_test
from common import reader


F_DIM = 64

SEED=98765
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

import global_constant
data_dir = global_constant.DATA_DIR

class Model6GCN(nn.Module):
    def __init__(self, url_f_embed_module=None, user_f_embed_module=None, embed_drop_out=0.1, if_one_layer=False):
        super(Model6GCN, self).__init__()
        self.if_one_layer = if_one_layer
        # int(uid_no), follow_cnt_bin, follow_in_cnt_bin, post_url_bin, post_cnt_bin
                # ,favorite_cate_no, favorite_site_no
        self.user_embed = embed.FeatureEmbed([11576+1, 10+1, 10+1, 10+1, 10+1, 20+1, 6+1], F_DIM, embed_drop_out)
        #int(url_no), cate2idx[cate], site2idx[url_site], post_user_bin, post_freq_bin
        self.url_embed = embed.FeatureEmbed([4732+1, 20+1, 6+1, 10+1, 10+1], F_DIM, embed_drop_out)

        if url_f_embed_module is not None:
            print(self.url_embed.features_embed_module[2].embed.weight[0:2, :])
            self.url_embed = url_f_embed_module
            print(self.url_embed.features_embed_module[2].embed.weight[0:2, :])
            '''for i in range(len(self.feature_model.url_features_embed)):
                #print(i, 'url', self.feature_model.url_features_embed[i].embed.weight.requires_grad)
                self.feature_model.url_features_embed[i].embed.weight.requires_grad = False
                #print(i, 'url', self.feature_model.url_features_embed[i].embed.weight.requires_grad)'''

        if user_f_embed_module is not None:
            #print(self.user_embed.features_embed_module[2].embed.weight[0:2, :])
            self.user_embed = user_f_embed_module
            #print(self.user_embed.features_embed_module[2].embed.weight[0:2, :])
            '''for i in range(len(self.feature_model.user_features_embed)):
                #print(i, 'user', self.feature_model.user_features_embed[i].embed.weight.requires_grad)
                self.feature_model.user_features_embed[i].embed.weight.requires_grad = False
                #print(i, 'user', self.feature_model.user_features_embed[i].embed.weight.requires_grad)'''

        self.cos2 = torch.nn.CosineSimilarity(dim=2, eps=1e-6)
        self.cos3 = torch.nn.CosineSimilarity(dim=3, eps=1e-6)

        self.linear_update2 = nn.Linear(F_DIM*2, F_DIM)
        utils.init_linear(self.linear_update2)

        self.linear_update1 = nn.Linear(F_DIM*2, F_DIM)
        utils.init_linear(self.linear_update1)

        self.linear_out = nn.Linear(F_DIM*2, 2)
        utils.init_linear(self.linear_out)


    def get_gragh_attention_embed2(self, f_embed, weight_list, center_embed):
        '''
        f_embed: [128, 3, 3, dim]
        center_embed:  (128, 3, dim)
        user_2_user_weight: [128, 3, 3]

        user_2_url_f_embed: [128, 3, 3, dim]
        user_2_url_weight: [128, 3, 3]
        '''
        N = f_embed.shape[0]
        cos_weight = self.cos3(f_embed, center_embed.unsqueeze(2))   # (N, 3, 3)
        cos_weight = torch.softmax(cos_weight, dim=2)                # (N, 3, 3)
        #cos_weight = torch.exp(cos_weight)
        #print('two 1', cos_weight[0, 0, :])
        #cos_weight = cos_weight / torch.pow(torch.sum(cos_weight, dim=2), 0.5).view(N, -1, 1)
        #print('two 2', cos_weight[0, 0, :])
        new_f_embed = torch.relu(f_embed * weight_list.unsqueeze(3)) * cos_weight.unsqueeze(3)  # (N, 3, 3, dim)
        new_f_embed = torch.sum(new_f_embed, dim=2)                                 # (N, 3, dim)
        #out_f_embed = torch.cat([center_embed, new_f_embed], dim=2)                 # (N, 3, 2*dim)
        #out_f_embed = torch.relu(self.linear_update(out_f_embed))                   # (N, 3, dim)
        
        #new_f_embed = torch.mean(f_embed, dim=2)

        return new_f_embed
        

    def get_gragh_attention_embed1(self, f_embed, weight_list, center_embed):
        '''
        f_embed: [128, 3, dim]
        center_embed:  (N, dim)
        user_1_user_weight: [128, 3]

        user_1_url_f_list: [128, 3, dim]
        user_1_url_weight: [128, 3]
        '''

        cos_weight = self.cos2(f_embed, center_embed.unsqueeze(1))   # (N, 3)
        cos_weight = torch.softmax(cos_weight, dim=1)                # (N, 3)
        #cos_weight = torch.exp(cos_weight)
        #print('one 1', cos_weight[0, :])
        #cos_weight = cos_weight / torch.pow(torch.sum(cos_weight, dim=1), 0.5).view(-1,1)
        #print('one 2', cos_weight[0, :])
        new_f_embed = torch.relu(f_embed * weight_list.unsqueeze(2)) * cos_weight.unsqueeze(2)   # (N, 3, dim)
        new_f_embed = torch.sum(new_f_embed, dim=1)                                 # (N, dim)
        #out_f_embed = torch.cat([center_embed, new_f_embed], dim=1)                 # (N, 2*dim)
        #out_f_embed = torch.relu(self.linear_update(out_f_embed))                   # (N, dim)
        #new_f_embed = torch.mean(f_embed, dim=1)

        return new_f_embed

    # 
    def forward(self, url_idxs, user_f_list, url_f_list
        ,user_1_user_f_list, user_1_user_weight
        ,user_1_url_f_list, user_1_url_weight
        ,user_2_user_f_list, user_2_user_weight
        ,user_2_url_f_list, user_2_url_weight
        ,url_2_user_f_list, url_2_user_weight
        ,url_2_url_f_list, url_2_url_weight):
        '''
        url_idxs: torch.Size([128])
        labels: torch.Size([128])
        user_f_list: torch.Size([128, 7])
        url_f_list: torch.Size([128, 5])

        user_1_user_f_list: [128, 3, 7]
        user_1_user_weight: [128, 3]

        user_1_url_f_list: [128, 3, 5]
        user_1_url_weight: [128, 3]
        
        user_2_user_f_list: [128, 3, 3, 7]
        user_2_user_weight: [128, 3, 3]

        user_2_url_f_list: [128, 3, 3, 5]
        user_2_url_weight: [128, 3, 3]
        
        url_2_user_f_list: [128, 3, 3, 7]
        url_2_user_weight: [128, 3, 3]
        
        url_2_url_f_list: [128, 3, 3, 5]
        url_2_url_weight: [128, 3, 3]
        '''
        if if_one_layer is False:
            ### net 2 url
            ori_user_1_url_embed = self.url_embed(user_1_url_f_list)       # (N, 3, dim)
            #print ('user_1_url', user_1_url_f_list[0], user_1_url_f_list.shape, ori_user_1_url_embed.shape)
            
            ori_url_2_user_embed = self.user_embed(url_2_user_f_list)      # (N, 3, 3, dim)
            #print ('url 2 user', url_2_user_f_list[0], url_2_user_f_list.shape, ori_url_2_user_embed.shape)
            url_2_user_new_embed = self.get_gragh_attention_embed2(ori_url_2_user_embed, url_2_user_weight
                    ,ori_user_1_url_embed)   # (N, 3, dim)
            
            ori_url_2_url_embed = self.url_embed(url_2_url_f_list)         # (N, 3, 3, dim)
            url_2_url_new_embed = self.get_gragh_attention_embed2(ori_url_2_url_embed, url_2_url_weight
                    ,ori_user_1_url_embed)    # (N, 3, dim)

            user_1_url_embed = (url_2_user_new_embed + url_2_url_new_embed) / 2
            user_1_url_embed_tmp = torch.cat([ori_user_1_url_embed, user_1_url_embed], dim=2)   # (N, 3 ,2*dim)
            # method1
            user_1_url_embed = torch.relu(self.linear_update2(user_1_url_embed_tmp))             # (N, 3, dim) 
            # method2
            #g = torch.sigmoid(self.linear_update2(user_1_url_embed_tmp))
            #user_1_url_embed = g * ori_user_1_url_embed + (1-g) * user_1_url_embed

            ### net 2 user
            ori_user_1_user_embed = self.user_embed(user_1_user_f_list)    # (N, 3, dim)

            ori_user_2_user_embed = self.user_embed(user_2_user_f_list)   # (N, 3, 3, dim)
            user_2_user_new_embed = self.get_gragh_attention_embed2(ori_user_2_user_embed, user_2_user_weight
                    ,ori_user_1_user_embed)   # (N, 3, dim)
            
            ori_user_2_url_embed = self.url_embed(user_2_url_f_list)      # (N, 3, 3, dim)
            user_2_url_new_embed = self.get_gragh_attention_embed2(ori_user_2_url_embed, user_2_url_weight
                    ,ori_user_1_user_embed)    # (N, 3, dim)
            user_1_user_embed = (user_2_user_new_embed + user_2_url_new_embed) / 2
            user_1_user_embed_tmp = torch.cat([ori_user_1_user_embed, user_1_user_embed], dim=2)   # (N, 3 ,dim)
            # method1
            user_1_user_embed = torch.relu(self.linear_update2(user_1_user_embed_tmp))             # (N, 3, dim) 
            # method2
            #g = torch.sigmoid(self.linear_update2(user_1_user_embed_tmp))
            #user_1_user_embed = g * ori_user_1_user_embed + (1-g) * user_1_user_embed

            # net 1 user
            ori_user_embed = self.user_embed(user_f_list)      # (N, dim)
            
            user_1_url_new_embed = self.get_gragh_attention_embed1(user_1_url_embed, user_1_url_weight, ori_user_embed)     # (N, dim)
            user_1_user_new_embed = self.get_gragh_attention_embed1(user_1_user_embed, user_1_user_weight, ori_user_embed)  # (N, dim)
            user_embed = (user_1_url_new_embed + user_1_user_new_embed) / 2        
            user_embed_tmp = torch.cat([ori_user_embed, user_embed], dim=1) # (N, 2*dim)
            # method1
            user_embed = torch.relu(self.linear_update1(user_embed_tmp))    # (N, dim)
            # method2
            #g = torch.sigmoid(self.linear_update1(user_embed_tmp))
            #user_embed = g * ori_user_embed + (1-g) * user_embed

            url_embed = self.url_embed(url_f_list)
            out = torch.cat([user_embed, url_embed], dim=1)
            out = self.linear_out(out)
            out = torch.softmax(out, dim=1)
        else:
            ################ only one layer
            ori_user_embed = self.user_embed(user_f_list)       # (N, dim)
            user_1_url_embed = self.url_embed(user_1_url_f_list)  # (N, 3, dim)
            user_1_user_embed = self.user_embed(user_1_user_f_list) # (N, 3 ,dim)
            
            user_1_url_new_embed = self.get_gragh_attention_embed1(user_1_url_embed, user_1_url_weight, ori_user_embed)     # (N, dim)
            user_1_user_new_embed = self.get_gragh_attention_embed1(user_1_user_embed, user_1_user_weight, ori_user_embed)  # (N, dim)
            user_embed = (user_1_url_new_embed + user_1_user_new_embed) / 2        
            user_embed_tmp = torch.cat([ori_user_embed, user_embed], dim=1) # (N, 2*dim)
            # method1
            user_embed = torch.relu(self.linear_update1(user_embed_tmp))    # (N, dim)
            # method2
            #g = torch.sigmoid(self.linear_update1(user_embed_tmp))
            #user_embed = g * ori_user_embed + (1-g) * user_embed

            url_embed = self.url_embed(url_f_list)
            out = torch.cat([user_embed, url_embed], dim=1)
            out = self.linear_out(out)
            out = torch.softmax(out, dim=1)
        return out


if __name__ == '__main__':
    ## read sample
    train_json_file = data_dir + os.sep + 'sample/train_dl_random.txt'
    test_file = data_dir + os.sep + 'sample/test.txt'
    #train_json_file = data_dir + os.sep + 'sample/train_dl_cate_random.txt'
    all_samples, all_test_samples = reader.read_train_and_test_sample(train_json_file, test_file)

    if_debug = False
    if_test = True
    if sys.argv[1] == 'gpu':
        IF_GPU = True
        device_id = int(sys.argv[2])
    else:
        IF_GPU = False
        device_id = None

    assert sys.argv[3] == 'lr'
    lr = float(sys.argv[4])

    assert sys.argv[5] == 'no'
    no_str = sys.argv[6]

    assert sys.argv[7] == 'l2'
    l2 = float(sys.argv[8])

    assert sys.argv[9] == 'top_ratio'
    v_ratio = float(sys.argv[10])

    assert sys.argv[11] == 'v_num'
    v_num = int(sys.argv[12])

    assert sys.argv[13] == 'layer'
    if int(sys.argv[14]) == 1: 
        if_one_layer = True
    else:
        if_one_layer = False

    assert sys.argv[15] == 'neg_num'
    neg_num = int(sys.argv[16])

    print('lr:', lr, 'l2:', l2, 'v_ratio:', v_ratio, 'v_num:', v_num, 'layer:', sys.argv[14], 'neg_num:', neg_num)
    
    ## read feature
    user_feature_json_file = data_dir + os.sep + 'feature/user_feature_list.json'
    url_feature_json_file = data_dir + os.sep + 'feature/url_feature_list.json'
    user_feature_list, url_feature_list = reader.read_feature(user_feature_json_file, url_feature_json_file
        ,device_id=None, if_tensor=True)

    ## read user url relation
    user_url_dict_file = data_dir + os.sep + 'feature/user_url_dict.json_weight'
    user_url_dict = reader.read_dict_json_by_ratio(user_url_dict_file, 20, v_ratio, name='user_urls')

    url_user_dict_file = data_dir + os.sep + 'feature/url_user_dict.json_weight'
    url_user_dict = reader.read_dict_json_by_ratio(url_user_dict_file, 20, v_ratio, name='url_users')

    ## read url neighb urls
    url_neighb_urls_dict_file = data_dir + os.sep + 'feature/url_neighb_urls.json_weight'
    url_neighb_urls = reader.read_dict_json_by_ratio(url_neighb_urls_dict_file, 20, v_ratio, 'url_neighb_urls')

    ## read user url neighb info
    user_neighb_users_dict_file = data_dir + os.sep + 'feature/user_neighb_users.json_weight'
    user_neighb_users = reader.read_dict_json_by_ratio(user_neighb_users_dict_file, 20, v_ratio, 'user_neighb_users')

    ## read weight tensor
    user_url_weight_json_file = data_dir + os.sep + 'gcn/user_url_weight.json'
    user_url_weight = reader.read_weights(user_url_weight_json_file, D=1.0, s=1.0, if_no_use=False, device_id=None)
    #user_user_weight 
    user_user_weight_json_file = data_dir + os.sep + 'gcn/user_user_weight.json'
    user_user_weight = reader.read_weights(user_user_weight_json_file, D=8899118, s=100, if_no_use=False, device_id=None)
    #url_url_weight
    url_url_weight_json_file = data_dir + os.sep + 'gcn/url_url_weight.json'
    url_url_weight = reader.read_weights(url_url_weight_json_file, D=289387, s=100, if_no_use=False, device_id=None)


    batch_size = 128
    test_total = 1157600
    test_batch_size = 100
    if if_test is True:
        batch_cnt = 100
        batch_log = 20
        test_batch_group = 10
    else:
        batch_cnt = 2431
        batch_log = 250
        test_batch_group = 11576

    model_dir = data_dir + os.sep + 'model_files/model6_gcn' + '_' + sys.argv[4].replace('.', '') + '_' + no_str
    if not os.path.isdir(model_dir):
        os.system('mkdir -p %s' % (model_dir))
    print('save model dir:', model_dir)

    ##### pretrain
    #epoch_no = 2
    #model_dir = data_dir + os.sep + 'model_files/model6_pre_0001_0'
    #user_f_embed_module = torch.load(model_dir + os.sep + 'user_embed_model.pkl' + str(epoch_no))
    #url_f_embed_module = torch.load(model_dir + os.sep + 'url_embed_model.pkl' + str(epoch_no))
    url_f_embed_module = None
    user_f_embed_module = None
    if url_f_embed_module is not None:
        embed_drop_out = 0.1
    else:
        embed_drop_out = 0.1
    model = Model6GCN(url_f_embed_module, user_f_embed_module, embed_drop_out, if_one_layer)
    mul_gpu = False
    if IF_GPU:
        #if torch.cuda.device_count() > 1:
        if mul_gpu:
            model = nn.DataParallel(model)
            model.cuda()
        else:
            model.cuda(device_id)
    print(model)
    #print('------------------------')
    #for name, m in model.named_children():
    #    print(name)
        #torch.save(m, model_dir + os.sep + 'f_model.pkl')
    #model.feature_model = torch.load(model_dir + os.sep + 'f_model.pkl')
    #print('--------------------------------')
    
    weight_p, bias_p = [], []
    for name, p in model.named_parameters():
        if 'bias' in name:  # no l2
            #print(name, 'no l2, lr 0.01')
            bias_p += [p]
        elif 'linear' in name:
            print(name, 'need l2')
            weight_p += [p]
        else:
            #print(name, 'default no l2, lr 0.01')
            pass

    optimizer = optim.Adam([
            {'params': weight_p, 'weight_decay': l2, 'lr': lr},
            {'params': bias_p, 'weight_decay': 0, 'lr': lr}
    ], lr=lr, weight_decay=0) #, momentum=0.9)

    #loss_weight = torch.tensor([1, 1], dtype=torch.float)
    #if IF_GPU:
    #    loss_weight = loss_weight.cuda(device_id)
    loss_func = torch.nn.NLLLoss()#weight=loss_weight) #
    #loss_func = torch.nn.CrossEntropyLoss()  # the target label is NOT an one-hotted

    test_sample_index = list(range(len(all_test_samples)))
    test_index_lists = utils.generate_batch_index(test_batch_size, 0, test_sample_index)
    print('test batch group:', len(test_index_lists))
    sample_index = list(range(len(all_samples)))
    random.shuffle(sample_index)

    for epoch in range(20):
        start_time = datetime.now()
        total_loss = 0
        model.train()
        print('%s start train epoch: %d' % (datetime.now(), epoch + 1))
        count = 0

        if neg_num > 0:
            train_pos_file = data_dir + os.sep + 'sample/train_dl.txt'
            all_samples = utils.build_train_sample_random(train_pos_file, neg_num=neg_num, total_url_cnt=4732)
            sample_index = list(range(len(all_samples)))
            print('neg num:', neg_num)

        random.shuffle(sample_index)
        indexs_list = utils.generate_batch_index(batch_size, 0, sample_index)
        print('train batch group:', len(indexs_list))
        for index_list in indexs_list[0: batch_cnt]:
            count += 1

            # Step 1. Prepare the inputs to be passed to the model (i.e, turn the words
            # into integer indices and wrap them in tensors)
            start = datetime.now()
            user_idxs, url_idxs, labels \
            ,user_f_list, url_f_list \
            ,user_1_user_f_list, user_1_user_weight \
            ,user_1_url_f_list, user_1_url_weight \
            ,user_2_user_f_list, user_2_user_weight \
            ,user_2_url_f_list, user_2_url_weight \
            ,url_2_user_f_list, url_2_user_weight \
            ,url_2_url_f_list, url_2_url_weight = reader.build_user_relation_by_sample(index_list, all_samples
                ,user_feature_list, url_feature_list
                ,user_url_weight, user_user_weight, url_url_weight
                ,user_url_dict, url_user_dict, user_neighb_users, url_neighb_urls
                ,mode='train', num=v_num)
            #print(count, 'build', (datetime.now() - start).total_seconds())
            if if_debug and epoch == 0:
                print('user_idxs', user_idxs.shape)
                print('url_idxs', url_idxs.shape)
                print('labels', labels.shape)
                print('user_f_list', user_f_list.shape)
                print('url_f_list', url_f_list.shape)
                print('user_1_user_f_list', user_1_user_f_list.shape)
                print('user_1_user_weight', user_1_user_weight.shape)
                #print(user_1_user_weight[0:2])
                print('user_1_url_f_list', user_1_url_f_list.shape)
                print('user_1_url_weight', user_1_url_weight.shape)
                print('user_2_user_f_list', user_2_user_f_list.shape)
                print('user_2_user_weight', user_2_user_weight.shape)
                print('user_2_url_f_list', user_2_url_f_list.shape)
                print('user_2_url_weight', user_2_url_weight.shape)
                print('url_2_user_f_list', url_2_user_f_list.shape)
                print('url_2_user_weight', url_2_user_weight.shape)
                print('url_2_url_f_list', url_2_url_f_list.shape)
                print('url_2_url_weight', url_2_url_weight.shape)

            start = datetime.now()
            if IF_GPU:
                if mul_gpu:
                    user_idxs = user_idxs.cuda()
                    url_idxs = url_idxs.cuda()
                    labels = labels.cuda()
                    user_f_list = user_f_list.cuda()
                    url_f_list = url_f_list.cuda()
                    user_1_user_f_list = user_1_user_f_list.cuda()
                    user_1_user_weight = user_1_user_weight.cuda()
                    user_1_url_f_list = user_1_url_f_list.cuda()
                    user_1_url_weight = user_1_url_weight.cuda()
                    user_2_user_f_list = user_2_user_f_list.cuda()
                    user_2_user_weight = user_2_user_weight.cuda()
                    user_2_url_f_list = user_2_url_f_list.cuda()
                    user_2_url_weight = user_2_url_weight.cuda()
                    url_2_user_f_list = url_2_user_f_list.cuda()
                    url_2_user_weight = url_2_user_weight.cuda()
                    url_2_url_f_list = url_2_url_f_list.cuda()
                    url_2_url_weight = url_2_url_weight.cuda()
                else:
                    user_idxs = user_idxs.cuda(device_id)
                    url_idxs = url_idxs.cuda(device_id)
                    labels = labels.cuda(device_id)
                    user_f_list = user_f_list.cuda(device_id)
                    url_f_list = url_f_list.cuda(device_id)
                    user_1_user_f_list = user_1_user_f_list.cuda(device_id)
                    user_1_user_weight = user_1_user_weight.cuda(device_id)
                    user_1_url_f_list = user_1_url_f_list.cuda(device_id)
                    user_1_url_weight = user_1_url_weight.cuda(device_id)
                    user_2_user_f_list = user_2_user_f_list.cuda(device_id)
                    user_2_user_weight = user_2_user_weight.cuda(device_id)
                    user_2_url_f_list = user_2_url_f_list.cuda(device_id)
                    user_2_url_weight = user_2_url_weight.cuda(device_id)
                    url_2_user_f_list = url_2_user_f_list.cuda(device_id)
                    url_2_user_weight = url_2_user_weight.cuda(device_id)
                    url_2_url_f_list = url_2_url_f_list.cuda(device_id)
                    url_2_url_weight = url_2_url_weight.cuda(device_id)
            #print(count, 'cuda', (datetime.now() - start).total_seconds())    
            # Step 2. Recall that torch *accumulates* gradients. Before passing in a
            # new instance, you need to zero out the gradients from the old instance
            start = datetime.now()
            model.zero_grad()

            # Step 3. Run the forward pass
            #output = nn.parallel.data_parallel(new_net, input, device_ids=[0, 1]
            out = model(url_idxs, user_f_list, url_f_list
                    ,user_1_user_f_list, user_1_user_weight
                    ,user_1_url_f_list, user_1_url_weight
                    ,user_2_user_f_list, user_2_user_weight
                    ,user_2_url_f_list, user_2_url_weight
                    ,url_2_user_f_list, url_2_user_weight
                    ,url_2_url_f_list, url_2_url_weight)

            # Step 4. Compute your loss function. (Again, Torch wants the target
            # word wrapped in a tensor)
            loss = loss_func(torch.log(out), labels)
            #loss = loss_func(out, labels)

            # Step 5. Do the backward pass and update the gradient
            loss.backward()
            optimizer.step()

            # Get the Python number from a 1-element Tensor by calling tensor.item()
            total_loss += loss.item()
            #print(count, 'train', (datetime.now() - start).total_seconds())   
            if if_debug:
                pass
                #print (model.linear_out.weight)
                #print ('grad---------------------------')
                #print (model.linear_out.weight.grad)
                #print ('----------------------------')
                #print (model.linear_out.bias)
                #print (model.linear_out.bias.grad)

            if (count % batch_log) == 0:
                prediction = torch.max(out, 1)[1]
                pred_y = prediction.cpu().data.numpy()
                target_y = labels.cpu().data.numpy()
                accuracy = float((pred_y == target_y).astype(int).sum()) / float(target_y.size)
                print (datetime.now(), 'batch:', count, ',loss:', loss.item(), ',accuracy:', accuracy, 
                    ',one case:', labels[10].cpu().data.numpy(), out[10, :].cpu().data.numpy())

        '''torch.save(model, model_dir + os.sep + 'model.pkl' + str(epoch))
        for name, m in model.named_children():
            if name == 'user_embed':
                torch.save(m, model_dir + os.sep + name + '_model.pkl' + str(epoch))
            elif name == 'url_embed':
                torch.save(m, model_dir + os.sep + name + '_model.pkl' + str(epoch))'''

        end_time = datetime.now()
        print('%s end train epoch %s, total cost %s' % (end_time, epoch+1, (end_time-start_time).total_seconds()))
       
        # one process
        all_test_res = []
        start_time = datetime.now()
        test_batch_cnt = 0
        total_test_case = 0
        for test_index_list in test_index_lists[0: test_batch_group]:
            total_test_case += len(test_index_list)
            res = predict_test.get_test_res6_gcn(model, IF_GPU
                    #,test_file_no, test_dir
                    ,test_index_list, all_test_samples
                    ,user_feature_list, url_feature_list
                    ,user_url_weight, user_user_weight, url_url_weight
                    ,user_url_dict, url_user_dict, user_neighb_users, url_neighb_urls
                    ,device_id=device_id, log_cnt=test_batch_cnt, mode='test', num=v_num)
            all_test_res.extend(res) 
            test_batch_cnt += 1

        end_time = datetime.now()
        print('%s, end predict total case: %d, test cost: %s' % (end_time, total_test_case, (end_time - start_time).total_seconds()))
        # two process
        #all_test_res = predict_test.get_test_res_by_pool(model, True, test_dir, test_index_list, proc_num=2, model_pkl=None)
        #print(len(all_test_res), 'start to evalu')
        hr, ncdg, avg_diff_cnt = utils.get_metric2(all_test_res)
        print (datetime.now(), 'epoch %d end, train loss: %s, test case: %s, test hr: %s, ncdg: %s, avg_diff_cnt: %s' 
            % (epoch+1, total_loss, len(all_test_res), hr[-1], ncdg[-1], avg_diff_cnt))

        print (datetime.now(), 'epoch %d end, hr1 loss: %s, test case: %s, test hr: %s, ncdg: %s, avg_diff_cnt: %s' 
            % (epoch+1, total_loss, len(all_test_res), hr[0], ncdg[0], avg_diff_cnt))

        print (datetime.now(), 'epoch %d end, hr3 loss: %s, test case: %s, test hr: %s, ncdg: %s, avg_diff_cnt: %s' 
            % (epoch+1, total_loss, len(all_test_res), hr[1], ncdg[1], avg_diff_cnt))

        print (datetime.now(), 'epoch %d end, hr5 loss: %s, test case: %s, test hr: %s, ncdg: %s, avg_diff_cnt: %s' 
            % (epoch+1, total_loss, len(all_test_res), hr[2], ncdg[2], avg_diff_cnt))
