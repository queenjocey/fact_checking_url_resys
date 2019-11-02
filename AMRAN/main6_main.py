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
from common import feature_models
from common import predict_test
from common import reader

SEED=98765
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

import global_constant
data_dir = global_constant.DATA_DIR


class Model6Main(nn.Module):
    def __init__(self, url_f_embed_module=None, user_f_embed_module=None, embed_drop_out=0.1
        ,if_one_layer=False, num_limit_list=[], F_DIM=64):
        super(Model6Main, self).__init__()
        self.if_one_layer = if_one_layer

        # feature model
        # num_limit_list=[CONSUM_NUM, NEIGHB_URL_NUM, FRIEND_NUM, NEIGHT_USER_NUM]
        self.feature_model = feature_models.FeatureModel(out_conv=True, attend_f=True, sa=False
            ,embed_drop_out=embed_drop_out, num_limit_list=num_limit_list, F_DIM=F_DIM)

        # int(uid_no), follow_cnt_bin, follow_in_cnt_bin, post_url_bin, post_cnt_bin
                # ,favorite_cate_no, favorite_site_no
        self.user_embed = embed.FeatureEmbed([11576+1, 10+1, 10+1, 10+1, 10+1, 20+1, 6+1], F_DIM, embed_drop_out)
        #int(url_no), cate2idx[cate], site2idx[url_site], post_user_bin, post_freq_bin
        self.url_embed = embed.FeatureEmbed([4732+1, 20+1, 6+1, 10+1, 10+1], F_DIM, embed_drop_out)

        self.cos2 = torch.nn.CosineSimilarity(dim=2, eps=1e-6)
        self.cos3 = torch.nn.CosineSimilarity(dim=3, eps=1e-6)

        self.linear_update2 = nn.Linear(F_DIM*2, F_DIM)
        utils.init_linear(self.linear_update2)

        self.linear_update1 = nn.Linear(F_DIM*2, F_DIM)
        utils.init_linear(self.linear_update1)

        self.linear_out = nn.Linear(F_DIM*(2+4), 2)
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

        return new_f_embed

    # 
    def forward(self, url_idxs, user_f_list, url_f_list
        ,user_1_user_f_list, user_1_user_weight
        ,user_1_url_f_list, user_1_url_weight
        ,user_2_user_f_list, user_2_user_weight
        ,user_2_url_f_list, user_2_url_weight
        ,url_2_user_f_list, url_2_user_weight
        ,url_2_url_f_list, url_2_url_weight
        ,consume_urls_f_list, neighb_urls_f_list
        ,friend_users_f_list, neighb_users_f_list):
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


        f_out = self.feature_model(user_idxs, url_idxs, user_f_list, url_f_list
                ,consume_urls_f_list, neighb_urls_f_list
                ,friend_users_f_list, neighb_users_f_list)  # (N, 4*dim)

        out = torch.cat([f_out, user_embed, url_embed], dim=1)  # (N, 6*dim)
        out = self.linear_out(out)
        out = torch.softmax(out, dim=1)
        return out


if __name__ == '__main__':
    ## read sample
    train_json_file = data_dir + os.sep + 'sample/train_dl_random.txt'
    test_file = data_dir + os.sep + 'sample/test.txt'
    all_samples, all_test_samples = reader.read_train_and_test_sample(train_json_file, test_file)
    train_dict = {}
    for item in all_samples:
        if item[0] not in train_dict:
            train_dict[item[0]] = [[], []]
        if item[2] == 1:
            #print(train_dict[item[0]][0], item[1])
            train_dict[item[0]][0].append(item[1])
        else:
            train_dict[item[0]][1].append(item[1])

    if_debug = False
    if_test = True
    print('if test:', if_test)
    if sys.argv[1] == 'gpu':
        IF_GPU = True
        device_id = int(sys.argv[2])
        torch.cuda.manual_seed(SEED)
    else:
        IF_GPU = False
        device_id = None

    assert sys.argv[3] == 'lr'
    lr = float(sys.argv[4])

    assert sys.argv[5] == 'dim'
    f_dim = int(sys.argv[6])

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

    assert sys.argv[17] == 'neighb_num'
    # num_limit_list=[CONSUM_NUM, NEIGHB_URL_NUM, FRIEND_NUM, NEIGHT_USER_NUM]
    num_limit_list = [int(sys.argv[18]), int(sys.argv[18]), int(sys.argv[18]), int(sys.argv[18])]


    print('lr:', lr, 'f_dim', f_dim, 'l2:', l2, 'v_ratio:', v_ratio, 'v_num:', v_num, 
        'layer:', sys.argv[14], 'neg_num:', neg_num, 'num_limit_list', num_limit_list)
    ## read feature
    user_feature_json_file = data_dir + os.sep + 'feature/user_feature_list.json'
    url_feature_json_file = data_dir + os.sep + 'feature/url_feature_list.json'
    user_feature_list, url_feature_list = reader.read_feature(user_feature_json_file, url_feature_json_file
        ,device_id=None, if_tensor=True)

    ## read user url relation
    user_url_dict_file = data_dir + os.sep + 'feature/user_url_dict.json_weight'
    user_url_dict = reader.read_dict_json_by_ratio(user_url_dict_file, 20, v_ratio, 'user_urls')

    url_user_dict_file = data_dir + os.sep + 'feature/url_user_dict.json_weight'
    url_user_dict = reader.read_dict_json_by_ratio(url_user_dict_file, 20, v_ratio, 'url_users')

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

    # for feature
    # read user follow users
    uid_no_follower_dict_file = data_dir + os.sep + 'feature/uid_no_follower_dict.json'
    uid_no_follower_dict = reader.read_dict_json_by_weight(uid_no_follower_dict_file, user_feature_list, 2, 20, v_ratio, 'uid_no_follower_dict')


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

    model_dir = data_dir + os.sep + 'model_files/model6_main' + '_' + str(f_dim) + '_' + str(neg_num)
    if not os.path.isdir(model_dir):
        os.system('mkdir -p %s' % (model_dir))
    print('save model dir:', model_dir)

    epoch_no = 2
    url_f_embed_module = None #torch.load(model_dir + os.sep + 'url_features_embed_model.pkl' + str(epoch_no))
    user_f_embed_module = None #torch.load(model_dir + os.sep + 'user_features_embed_model.pkl' + str(epoch_no))
    if url_f_embed_module is not None:
        embed_drop_out = 0
    else:
        embed_drop_out = 0.1
    model = Model6Main(url_f_embed_module, user_f_embed_module, embed_drop_out, if_one_layer, num_limit_list, f_dim)
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
            print('neg num:', neg_num)
            #method1
            train_pos_file = data_dir + os.sep + 'sample/train_dl.txt'
            all_samples = utils.build_train_sample_random(train_pos_file, neg_num=neg_num, total_url_cnt=4732)

            # method2
            '''all_samples = []
            for uid in train_dict.keys():
                pos_samples = train_dict[uid][0]
                for url_id in pos_samples:
                    all_samples.append([int(uid), int(url_id), 1])
                random.shuffle(train_dict[uid][1])
                for neg_url_id in train_dict[uid][1][0:neg_num*len(pos_samples)]:
                    all_samples.append([int(uid), int(neg_url_id), 0])'''

        sample_index = list(range(len(all_samples)))
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

            user_idxs1, url_idxs1, labels1, user_f_list1, url_f_list1, consume_urls_f_list, neighb_urls_f_list \
                ,friend_users_f_list, neighb_users_f_list = reader.read_batch_f_data_by_sample(index_list, all_samples
                    ,user_feature_list, url_feature_list, user_url_dict, url_neighb_urls
                    ,uid_no_follower_dict, user_neighb_users
                    ,num_limit_list, mode='train')
            '''
            user_idxs, url_idxs, labels \
            ,user_f_list, url_f_list \
            ,user_1_user_f_list, user_1_user_weight \
            ,user_1_url_f_list, user_1_url_weight \
            ,user_2_user_f_list, user_2_user_weight \
            ,user_2_url_f_list, user_2_url_weight \
            ,url_2_user_f_list, url_2_user_weight \
            ,url_2_url_f_list, url_2_url_weight \
            ,consume_urls_f_list, neighb_urls_f_list \
            ,friend_users_f_list, neighb_users_f_list = reader.build_gcn_and_f_by_sample(index_list, all_samples
                ,user_feature_list, url_feature_list
                ,user_url_weight, user_user_weight, url_url_weight
                ,user_url_dict, url_user_dict, user_neighb_users, url_neighb_urls
                ,uid_no_follower_dict, [CONSUM_NUM, NEIGHB_URL_NUM, FRIEND_NUM, NEIGHT_USER_NUM]
                ,mode='train', num=v_num)'''
            

            '''assert torch.equal(user_idxs, user_idxs1)
            assert torch.equal(url_idxs, url_idxs1)
            assert torch.equal(labels, labels1)
            assert torch.equal(user_f_list, user_f_list1)
            assert torch.equal(url_f_list, url_f_list1)
            print('check train done')'''
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

                consume_urls_f_list = consume_urls_f_list.cuda(device_id)
                neighb_urls_f_list = neighb_urls_f_list.cuda(device_id)
                friend_users_f_list = friend_users_f_list.cuda(device_id)
                neighb_users_f_list = neighb_users_f_list.cuda(device_id)

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
                    ,url_2_url_f_list, url_2_url_weight
                    ,consume_urls_f_list, neighb_urls_f_list
                    ,friend_users_f_list, neighb_users_f_list)

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

            if (count % batch_log) == 0:
                prediction = torch.max(out, 1)[1]
                pred_y = prediction.cpu().data.numpy()
                target_y = labels.cpu().data.numpy()
                accuracy = float((pred_y == target_y).astype(int).sum()) / float(target_y.size)
                print (datetime.now(), 'batch:', count, ',loss:', loss.item(), ',accuracy:', accuracy, 
                    ',one case:', labels[10].cpu().data.numpy(), out[10, :].cpu().data.numpy())

        torch.save(model, model_dir + os.sep + 'model.pkl' + str(epoch))
        '''for name, m in model.named_children():
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
            res = predict_test.get_test_res6_main(model, IF_GPU
                    ,test_index_list, all_test_samples
                    ,user_feature_list, url_feature_list
                    ,user_url_weight, user_user_weight, url_url_weight
                    ,user_url_dict, url_user_dict, user_neighb_users, url_neighb_urls
                    ,device_id=device_id, log_cnt=test_batch_cnt, mode='test', num=v_num
                    ,uid_no_follower_dict=uid_no_follower_dict
                    ,num_limit_list=num_limit_list)
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