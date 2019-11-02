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
from common import feature_models

CONSUM_NUM = 3
#NEIGHB_URL_NUM = 20
#FRIEND_NUM = 20
#NEIGHT_USER_NUM = 20

F_DIM = 64

SEED=98765
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)


import global_constant
data_dir = global_constant.DATA_DIR


class Model6Features(nn.Module):
    def __init__(self, url_f_embed_module=None, user_f_embed_module=None, embed_drop_out=0.1, num_limit_list=[]):
        super(Model6Features, self).__init__()
        self.feature_model = feature_models.FeatureModel(out_conv=True, attend_f=False, 
                sa=False, embed_drop_out=embed_drop_out, num_limit_list=num_limit_list)
        if url_f_embed_module is not None:
            self.feature_model.url_features_embed = url_f_embed_module.features_embed_module
            self.feature_model.url = url_f_embed_module.linear_out

        if user_f_embed_module is not None:
            self.feature_model.user_features_embed = user_f_embed_module.features_embed_module
            self.feature_model.user = user_f_embed_module.linear_out

        self.linear_out = nn.Linear(F_DIM*(2+2), 2)
        utils.init_linear(self.linear_out)

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
        out = self.feature_model(user_idxs, url_idxs, user_f_list, url_f_list
                ,consume_urls_f_list, neighb_urls_f_list
                ,friend_users_f_list, neighb_users_f_list)
        out = self.linear_out(out)
        out = torch.softmax(out, dim=1)
        return out


if __name__ == '__main__':
    ## read sample
    train_json_file = data_dir + os.sep + 'sample/train_dl_random.txt'
    test_file=data_dir + os.sep + 'sample/test.txt'
    all_samples, all_test_samples = reader.read_train_and_test_sample(train_json_file, test_file)

    if_debug = False
    if_test = True
    print('if_test:', if_test)
    if sys.argv[1] == 'gpu':
        IF_GPU = True
        device_id = int(sys.argv[2])
        torch.cuda.manual_seed(SEED)
    else:
        IF_GPU = False
        device_id = 0
    assert sys.argv[3] == 'lr'
    lr = float(sys.argv[4])

    assert sys.argv[5] == 'neg_num'
    neg_num = int(sys.argv[6])

    assert sys.argv[7] == 'neighb_num'
    # num_limit_list=[CONSUM_NUM, NEIGHB_URL_NUM, FRIEND_NUM, NEIGHT_USER_NUM]
    num_limit_list = [3, int(sys.argv[8]), int(sys.argv[8]), int(sys.argv[8])]


    ## read feature
    user_feature_json_file = data_dir + os.sep + 'feature/user_feature_list.json'
    url_feature_json_file = data_dir + os.sep + 'feature/url_feature_list.json'
    user_feature_list, url_feature_list = reader.read_feature(user_feature_json_file, url_feature_json_file, 
        device_id=None, if_tensor=True)

    ## read user url relation
    user_url_dict_file = data_dir + os.sep + 'feature/user_url_dict.json'
    #user_url_dict = reader.read_dict_json(user_url_dict_file, 'user_url_dict')
    user_url_dict = reader.read_dict_json_by_weight(user_url_dict_file, url_feature_list, 3, 20, 0.25, 'consume_urls')

    ## read url neighb urls
    url_neighb_urls_dict_file = data_dir + os.sep + 'feature/url_neighb_urls.json_weight'
    url_neighb_urls = reader.read_dict_json_by_ratio(url_neighb_urls_dict_file, 20, 0.25, 'url_neighb_urls')

    # read user follow users
    uid_no_follower_dict_file = data_dir + os.sep + 'feature/uid_no_follower_dict.json'
    uid_no_follower_dict = reader.read_dict_json_by_weight(uid_no_follower_dict_file, user_feature_list, 2, 20, 0.25, 'uid_no_follower_dict')

    ## read user url neighb info
    user_neighb_users_dict_file = data_dir + os.sep + 'feature/user_neighb_users.json_weight'
    user_neighb_users = reader.read_dict_json_by_ratio(user_neighb_users_dict_file, 20, 0.25, 'user_neighb_users')

    # read url neighb urls
    #url_neighb_urls_dict_file = data_dir + os.sep + 'feature/url_neighb_urls.json'
    #url_neighb_urls = reader.read_dict_json_by_weight(url_neighb_urls_dict_file, url_feature_list, 3, NEIGHB_URL_NUM, 0.25, 'url_neighb_urls')
    ## read user url neighb info
    #user_neighb_users_dict_file = data_dir + os.sep + 'feature/user_neighb_users.json'
    #user_neighb_users = reader.read_dict_json_by_weight(user_neighb_users_dict_file, user_feature_list, 2, NEIGHT_USER_NUM, 0.25, 'user_neighb')
    

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

    #model_dir = '/home/dyou/url_recsys/dl_code/model_files/model6_feature' + '_' + sys.argv[4].replace('.', '') + '_' + no_str
    #if not os.path.isdir(model_dir):
    #    os.system('mkdir -p %s' % (model_dir))
    #print('save model dir:', model_dir)

    ### pretrain
    #epoch_no = 2
    #model_dir = '/home/dyou/url_recsys/dl_code/model_files/model6_pre_0001_0'
    #user_f_embed_module = torch.load(model_dir + os.sep + 'user_embed_model.pkl' + str(epoch_no))
    #url_f_embed_module = torch.load(model_dir + os.sep + 'url_embed_model.pkl' + str(epoch_no))
    
    url_f_embed_module = None
    user_f_embed_module = None
    if url_f_embed_module is not None:
        embed_drop_out = 0.1
    else:
        embed_drop_out = 0.1
    model = Model6Features(url_f_embed_module, user_f_embed_module, embed_drop_out, num_limit_list)
    if IF_GPU:
        model.cuda(device_id)
    print(model)
    '''print('------------------------')
    for name, m in model.named_children():
        if 'feature_model' in name:
            print(m)
        torch.save(m, model_dir + os.sep + 'f_model.pkl')
    model.feature_model = torch.load(model_dir + os.sep + 'f_model.pkl')
    print('--------------------------------')'''
    print('lr:', lr, 'neg_num:', neg_num, 'neighbs:', num_limit_list)
    weight_p, bias_p = [], []
    for name, p in model.named_parameters():
        if 'bias' in name:  # no l2
            bias_p += [p]
        elif 'linear' in name:
            weight_p += [p]
        else:
            pass

    optimizer = optim.Adam([
            {'params': weight_p, 'weight_decay': 0, 'lr': lr},
            {'params': bias_p, 'weight_decay': 0, 'lr': lr}
    ], lr=lr, weight_decay=0)

    loss_func = torch.nn.NLLLoss()#weight=loss_weight) #
    #loss_func = torch.nn.CrossEntropyLoss()  # the target label is NOT an one-hotted

    test_sample_index = list(range(len(all_test_samples)))
    test_indexs_list = utils.generate_batch_index(test_batch_size, 0, test_sample_index)
    print('test batch group:', len(test_indexs_list))
    sample_index = list(range(len(all_samples)))

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

            # Step 1. Prepare the inputs to be passed to the model
            user_idxs, url_idxs, labels, user_f_list, url_f_list, consume_urls_f_list, neighb_urls_f_list \
                ,friend_users_f_list, neighb_users_f_list = reader.read_batch_f_data_by_sample(index_list, all_samples
                    ,user_feature_list, url_feature_list, user_url_dict, url_neighb_urls
                    ,uid_no_follower_dict, user_neighb_users
                    ,num_limit_list, mode='train')

            if if_debug:
                print('user_idxs:', user_idxs.shape)
                print('url_idxs:', url_idxs.shape)
                print('labels:', labels.shape)
                print('user_f_list:', user_f_list.shape)
                print('url_f_list:', url_f_list.shape)
                print('consume_urls_f_list:', consume_urls_f_list.shape)
                print('neighb_urls_f_list :', neighb_urls_f_list.shape)
                print('friend_users_f_list:', friend_users_f_list.shape)
                print('neighb_users_f_list:', neighb_users_f_list.shape)
                #print(user_idxs)
                #print([all_samples[index] for index in index_list])
                #sys.exit()

            if IF_GPU:
                user_idxs = user_idxs.cuda(device_id)
                url_idxs = url_idxs.cuda(device_id)
                labels = labels.cuda(device_id)
                user_f_list = user_f_list.cuda(device_id)
                url_f_list = url_f_list.cuda(device_id)
                consume_urls_f_list = consume_urls_f_list.cuda(device_id)
                neighb_urls_f_list = neighb_urls_f_list.cuda(device_id)
                friend_users_f_list = friend_users_f_list.cuda(device_id)
                neighb_users_f_list = neighb_users_f_list.cuda(device_id)

            # Step 2. Recall that torch *accumulates* gradients
            model.zero_grad()

            # Step 3. Run the forward pass
            out = model(user_idxs, url_idxs, user_f_list, url_f_list
                ,consume_urls_f_list, neighb_urls_f_list
                ,friend_users_f_list, neighb_users_f_list)

            # Step 4. Compute your loss function.
            loss = loss_func(torch.log(out), labels)
            #loss = loss_func(out, labels)

            # Step 5. Do the backward pass and update the gradient
            loss.backward()
            optimizer.step()

            # Get the Python number from a 1-element Tensor by calling tensor.item()
            total_loss += loss.item()

            if if_debug:
                #print (model.linear_out.weight)
                #print ('grad---------------------------')
                #print (model.linear_out.weight.grad)
                #print ('----------------------------')
                #print (model.linear_out.bias)
                #print (model.linear_out.bias.grad)
                print ('train res:--------------------------------------------')
                print (out[0:10])
                print ('train url conv --------------------------')
                print (model.url_consume_conv.conv.weight)
                print ('train url conv grad ----------')
                print (model.url_consume_conv.conv.weight.grad)
                print ('train user conv --------------------------')
                print (model.user_friend_conv.conv.weight)
                print ('train user conv grad ----------')
                print (model.user_friend_conv.conv.weight.grad)


            if (count % batch_log) == 0:
                prediction = torch.max(out, 1)[1]
                pred_y = prediction.cpu().data.numpy()
                target_y = labels.cpu().data.numpy()
                accuracy = float((pred_y == target_y).astype(int).sum()) / float(target_y.size)
                print (datetime.now(), 'batch:', count, ',loss:', loss.item(), ',accuracy:', accuracy, 
                    ',one case:', labels[10].cpu().data.numpy(), out[10, :].cpu().data.numpy())

        '''torch.save(model, model_dir + os.sep + 'model.pkl' + str(epoch))
        for name, m in model.named_children():
            if name == 'feature_model':
                feature_model = m
                break
        for name, m in feature_model.named_children():
            for need_name in ['url_features_embed', 'user_features_embed']:
                if need_name in name:
                    print (epoch, name, 'save')
                    torch.save(m, model_dir + os.sep + need_name + '_model.pkl' + str(epoch))'''

        end_time = datetime.now()
        print('%s end train epoch %s, total cost %s' % (end_time, epoch+1, (end_time-start_time).total_seconds()))
       
        # one process
        all_test_res = []
        start_time = datetime.now()
        test_batch_cnt = 0
        total_test_case = 0
        for test_indexs in test_indexs_list[0: test_batch_group]:
            total_test_case += len(test_indexs)
            res = predict_test.get_test_res6_f(model, IF_GPU, test_indexs, all_test_samples
                    #all_tweets_word_idx, all_tweets_char_idx, all_url_word_idx, all_url_char_idx, 
                    ,user_feature_list, url_feature_list, user_url_dict, url_neighb_urls
                    ,uid_no_follower_dict, user_neighb_users
                    ,num_limit_list
                    ,device_id=device_id, log_cnt=test_batch_cnt)
            all_test_res.extend(res) 
            test_batch_cnt += 1

        end_time = datetime.now()
        print('%s, end predict total case: %d, test cost: %s' % (end_time, total_test_case, (end_time - start_time).total_seconds()))
        hr, ncdg, avg_diff_cnt = utils.get_metric2(all_test_res)
        print (datetime.now(), 'epoch %d end, train loss: %s, test case: %s, test hr: %s, ncdg: %s, avg_diff_cnt: %s' 
            % (epoch+1, total_loss, len(all_test_res), hr[-1], ncdg[-1], avg_diff_cnt))

        print (datetime.now(), 'epoch %d end, hr1 loss: %s, test case: %s, test hr: %s, ncdg: %s, avg_diff_cnt: %s' 
            % (epoch+1, total_loss, len(all_test_res), hr[0], ncdg[0], avg_diff_cnt))

        print (datetime.now(), 'epoch %d end, hr3 loss: %s, test case: %s, test hr: %s, ncdg: %s, avg_diff_cnt: %s' 
            % (epoch+1, total_loss, len(all_test_res), hr[1], ncdg[1], avg_diff_cnt))

        print (datetime.now(), 'epoch %d end, hr5 loss: %s, test case: %s, test hr: %s, ncdg: %s, avg_diff_cnt: %s' 
            % (epoch+1, total_loss, len(all_test_res), hr[2], ncdg[2], avg_diff_cnt))
