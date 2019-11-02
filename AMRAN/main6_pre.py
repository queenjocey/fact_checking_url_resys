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

class Model6Pre(nn.Module):
    def __init__(self, url_f_embed_module=None, user_f_embed_module=None, embed_drop_out=0.1, if_easy=True):
        super(Model6Pre, self).__init__()
        self.if_easy = if_easy
        
        if if_easy:
            self.user_embed = embed.FeatureEmbed([11576+1], F_DIM, 0.1)
            self.url_embed = embed.FeatureEmbed([4732+1], F_DIM, 0.1)
        else:
            # int(uid_no), follow_cnt_bin, follow_in_cnt_bin, post_url_bin, post_cnt_bin
                # ,favorite_cate_no, favorite_site_no
            self.user_embed = embed.FeatureEmbed([11576+1, 10+1, 10+1, 10+1, 10+1, 20+1, 6+1], F_DIM, 0.1)
            #int(url_no), cate2idx[cate], site2idx[url_site], post_user_bin, post_freq_bin
            self.url_embed = embed.FeatureEmbed([4732+1, 20+1, 6+1, 10+1, 10+1], F_DIM, 0.1)

        self.linear_out = nn.Linear(F_DIM*2, 2)
        utils.init_linear(self.linear_out)

    # 
    def forward(self, user_f_list, url_f_list):
        '''
        url_idxs: torch.Size([128])
        labels: torch.Size([128])
        user_f_list: torch.Size([128, 7])
        url_f_list: torch.Size([128, 5])
        '''
        if self.if_easy:
            user_f_list = user_f_list[:, 0].view(-1, 1)
            url_f_list = url_f_list[:, 0].view(-1, 1)
        user_embed = self.user_embed(user_f_list)     # (N, dim)
        url_embed = self.url_embed(url_f_list)
        out = torch.cat([user_embed, url_embed], dim=1)
        out = self.linear_out(out)
        out = torch.softmax(out, dim=1)
        return out


if __name__ == '__main__':
    data_dir = '../data'

    if_easy = False
    if_test = False
    ## read sample
    train_json_file = data_dir + os.sep + 'sample/train_dl_random.txt'
    test_file=data_dir + os.sep + 'sample/test.txt'
    all_samples, all_test_samples = reader.read_train_and_test_sample(train_json_file, test_file)

    if_debug = False
    if sys.argv[1] == 'gpu':
        IF_GPU = True
        device_id = int(sys.argv[2])
        torch.cuda.manual_seed(SEED)
    else:
        IF_GPU = False
        device_id = None

    ## read feature
    user_feature_json_file = data_dir + os.sep + 'feature/user_feature_list.json'
    url_feature_json_file = data_dir + os.sep + 'feature/url_feature_list.json'
    user_feature_list, url_feature_list = reader.read_feature(user_feature_json_file, url_feature_json_file, device_id=None, if_tensor=True)

    batch_size = 128
    test_total = 1157600
    test_batch_size = 100
    if if_test is True:
        batch_cnt = 100
        batch_log = 20
        test_batch_group = 100
    else:
        batch_cnt = 2431
        batch_log = 250
        test_batch_group = 11576

    assert sys.argv[3] == 'lr'
    lr = float(sys.argv[4])

    assert sys.argv[5] == 'no'
    no_str = sys.argv[6]

    assert sys.argv[7] == 'l2'
    l2 = float(sys.argv[8])

    model_dir = data_dir + os.sep + 'model_files/model6_pre' + '_' + sys.argv[4].replace('.', '') + '_' + no_str
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
    model = Model6Pre(url_f_embed_module, user_f_embed_module, embed_drop_out, if_easy)
    if IF_GPU:
        model.cuda(device_id)
    print(model)
    print('lr:', lr, 'l2:', l2)
    weight_p, bias_p = [], []
    for name, p in model.named_parameters():
        if 'bias' in name:  # no l2
            bias_p += [p]
        elif 'linear' in name:
            print(name, 'need l2')
            weight_p += [p]
        else:
            pass

    optimizer = optim.Adam([
            {'params': weight_p, 'weight_decay': l2, 'lr': lr},
            {'params': bias_p, 'weight_decay': 0, 'lr': lr}
    ], lr=lr, weight_decay=0)

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

        random.shuffle(sample_index)
        indexs_list = utils.generate_batch_index(batch_size, 0, sample_index)
        print('train batch group:', len(indexs_list))
        for index_list in indexs_list[0: batch_cnt]:
            count += 1

            # Step 1. Prepare the inputs to be passed to the model
            user_idxs, url_idxs, labels \
            ,user_f_list, url_f_list = reader.read_user_url_f_tensor_by_sample(index_list, all_samples
                , user_feature_list, url_feature_list)

            if if_debug:
                print('user_idxs', user_idxs.shape)
                print('url_idxs', url_idxs.shape)
                print('labels', labels.shape)
                print('user_f_list', user_f_list.shape)
                print('url_f_list', url_f_list.shape)

            if IF_GPU:
                user_idxs = user_idxs.cuda(device_id)
                url_idxs = url_idxs.cuda(device_id)
                labels = labels.cuda(device_id)
                user_f_list = user_f_list.cuda(device_id)
                url_f_list = url_f_list.cuda(device_id)

            # Step 2. Recall that torch *accumulates* gradients.
            model.zero_grad()

            # Step 3. Run the forward pass
            out = model(user_f_list, url_f_list)

            # Step 4. Compute your loss function
            loss = loss_func(torch.log(out), labels)

            # Step 5. Do the backward pass and update the gradient
            loss.backward()
            optimizer.step()

            # Get the Python number from a 1-element Tensor by calling tensor.item()
            total_loss += loss.item()

            if (count % batch_log) == 0:
                prediction = torch.max(out, 1)[1]
                pred_y = prediction.cpu().data.numpy()
                target_y = labels.cpu().data.numpy()
                accuracy = float((pred_y == target_y).astype(int).sum()) / float(target_y.size)
                print (datetime.now(), 'batch:', count, ',loss:', loss.item(), ',accuracy:', accuracy, 
                    ',one case:', labels[10].cpu().data.numpy(), out[10, :].cpu().data.numpy())

        torch.save(model, model_dir + os.sep + 'model.pkl' + str(epoch))
        for name, m in model.named_children():
            if name == 'user_embed':
                torch.save(m, model_dir + os.sep + name + '_model.pkl' + str(epoch))
            elif name == 'url_embed':
                torch.save(m, model_dir + os.sep + name + '_model.pkl' + str(epoch))

        end_time = datetime.now()
        print('%s end train epoch %s, total cost %s' % (end_time, epoch+1, (end_time-start_time).total_seconds()))
       
        # predict
        all_test_res = []
        start_time = datetime.now()
        test_batch_cnt = 0
        total_test_case = 0
        for test_index_list in test_index_lists[0: test_batch_group]:
            total_test_case += len(test_index_list)
            res = predict_test.get_test_res6_pre(model, IF_GPU
                    #,test_file_no, test_dir
                    ,test_index_list, all_test_samples
                    ,user_feature_list, url_feature_list
                    ,device_id=device_id, log_cnt=test_batch_cnt)
            all_test_res.extend(res) 
            test_batch_cnt += 1

        end_time = datetime.now()
        print('%s, end predict total case: %d, test cost: %s' % (end_time, total_test_case, (end_time - start_time).total_seconds()))

        hr, ncdg, avg_diff_cnt = utils.get_metric(all_test_res)
        print (datetime.now(), 'epoch %d end, train loss: %s, test case: %s, test hr: %s, ncdg： %s, avg_diff_cnt: %s' 
            % (epoch+1, total_loss, len(all_test_res), hr, ncdg, avg_diff_cnt))
