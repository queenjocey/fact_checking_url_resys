import os
import torch
from datetime import datetime

from common import utils
from common import reader

from torch.multiprocessing import Pool
torch.multiprocessing.set_start_method('spawn', force=True)


import global_constant
DATA_DIR = global_constant.DATA_DIR


def get_test_res1(model, if_gpu, test_dir, test_file_no, device_id=0, if_url_short=True):
    start_time = datetime.now()
    proc_id = os.getpid()
    if test_file_no % 1500 == 0:
        print('%s, [%s] start predict test' % (start_time, proc_id))
    test_res = []
    user_idxs, url_idxs, url_word_idxs, url_char_idxs, tweets_word_idxs, tweets_char_idxs, labels = reader.read_batch_text_data(test_file_no, test_dir)
    if if_url_short:
        url_word_idxs = url_word_idxs[:, 0:4, :]
        url_char_idxs = url_char_idxs[:, 0:4, :, :]
    if if_gpu is True:
        #model.cuda(device_id)
        user_idxs = user_idxs.cuda(device_id)
        url_idxs = url_idxs.cuda(device_id)
        url_word_idxs = url_word_idxs.cuda(device_id)
        url_char_idxs = url_char_idxs.cuda(device_id)
        tweets_word_idxs = tweets_word_idxs.cuda(device_id)
        tweets_char_idxs = tweets_char_idxs.cuda(device_id)
        labels = labels.cuda(device_id)
    else:
        model.cpu()

    if test_file_no == 0:
        print('test url word char:', url_word_idxs.shape, url_char_idxs.shape)

    model.eval()
    #out = model(user_idxs, url_idxs, url_word_idxs, url_char_idxs, tweets_word_idxs, tweets_char_idxs)
    out = model(user_idxs, url_idxs, url_word_idxs, url_char_idxs, tweets_word_idxs, tweets_char_idxs)
    test_res.extend(list(out.cpu().data.numpy()[:, 1]))
    if test_file_no % 1500 == 0:
        end_time = datetime.now()
        print('%s, [%s] end predict test cost %s' % (end_time, proc_id, (end_time - start_time).total_seconds()))
    return test_res  


# only user and url embeding
def get_test_res1_base(model, if_gpu, test_dir, test_file_nos, device_id=0):
    start_time = datetime.now()
    proc_id = os.getpid()
    print('%s, [%s] start predict test %s * 100' % (start_time, proc_id, len(test_file_nos)))
    test_res = []
    for test_file_no in test_file_nos:
        user_idxs, url_idxs, labels = reader.read_batch_user_url_label(test_file_no, test_dir)
        if if_gpu is True:
            #model.cuda(device_id)
            user_idxs = user_idxs.cuda(device_id)
            url_idxs = url_idxs.cuda(device_id)
            labels = labels.cuda(device_id)
        else:
            model.cpu()
        model.eval()
        out = model(user_idxs, url_idxs)
        test_res.extend(list(out.cpu().data.numpy()[:, 1]))
    end_time = datetime.now()
    print('%s, [%s] end predict test cost %s' % (end_time, proc_id, (end_time - start_time).total_seconds()))
    return test_res


# 
def get_test_res3(model, if_gpu, index_list, all_samples
                ,all_tweets_word_idx, all_tweets_char_idx, all_url_word_idx, all_url_char_idx
                ,device_id=0, log_cnt=1):
    start_time = datetime.now()
    proc_id = os.getpid()
    test_res = []
    if log_cnt % 1000 == 0:
        print('%s, [%s] start predict test %d, %s' % (start_time, proc_id, len(index_list), index_list[0]))
    user_idxs, url_idxs, url_word_idxs, url_char_idxs, tweets_word_idxs, \
    tweets_char_idxs, labels = utils.chose_batch_case(index_list, all_samples, 
                all_tweets_word_idx, all_tweets_char_idx, all_url_word_idx, all_url_char_idx)
    if if_gpu is True:
        #model.cuda(device_id)
        user_idxs = user_idxs.cuda(device_id)
        url_idxs = url_idxs.cuda(device_id)
        url_word_idxs = url_word_idxs.cuda(device_id)
        url_char_idxs = url_char_idxs.cuda(device_id)
        tweets_word_idxs = tweets_word_idxs.cuda(device_id)
        tweets_char_idxs = tweets_char_idxs.cuda(device_id)
        labels = labels.cuda(device_id)
    else:
        model.cpu()
    model.eval()
    out = model(user_idxs, url_idxs, url_word_idxs, url_char_idxs, tweets_word_idxs, tweets_char_idxs)
    test_res.extend(list(out.cpu().data.numpy()[:, 1]))
    end_time = datetime.now()
    if log_cnt % 1000 == 0:
        print('%s, [%s] end predict test cost %s' % (end_time, proc_id, (end_time - start_time).total_seconds()))
    return test_res


def get_test_res6_f(model, if_gpu, index_list, all_samples
    ,user_feature_list, url_feature_list, user_url_dict, url_neighb_urls
    ,uid_no_follower_dict, user_neighb_users, num_limit_list, device_id=0, log_cnt=0):
    if log_cnt % 1500 == 0:
        start_time = datetime.now()
        proc_id = os.getpid()
        print('%s, [%s] start predict test %d, %s' % (start_time, proc_id, len(index_list), index_list[0]))
    
    target_dir =  DATA_DIR + os.sep + 'f/test/'
    user_idxs, url_idxs, labels, user_f_list, url_f_list, consume_urls_f_list, neighb_urls_f_list \
        ,friend_users_f_list, neighb_users_f_list = reader.read_batch_f_data(log_cnt, target_dir, num_limit_list)
    '''reader.read_batch_f_data_by_sample(index_list, all_samples
                    ,user_feature_list, url_feature_list, user_url_dict, url_neighb_urls
                    ,uid_no_follower_dict, user_neighb_users, num_limit_list, 'test')
    '''
    if if_gpu is True:
        #model.cuda(device_id)
        user_idxs = user_idxs.cuda(device_id)
        url_idxs = url_idxs.cuda(device_id)
        labels = labels.cuda(device_id)
        user_f_list = user_f_list.cuda(device_id)
        url_f_list = url_f_list.cuda(device_id)
        consume_urls_f_list = consume_urls_f_list.cuda(device_id)
        neighb_urls_f_list = neighb_urls_f_list.cuda(device_id)
        friend_users_f_list = friend_users_f_list.cuda(device_id)
        neighb_users_f_list = neighb_users_f_list.cuda(device_id)
    else:
        model.cpu()
    model.eval()

    test_res = []
    out = model(user_idxs, url_idxs, user_f_list, url_f_list
                ,consume_urls_f_list, neighb_urls_f_list
                ,friend_users_f_list, neighb_users_f_list)
    test_res.extend(list(out.cpu().data.numpy()[:, 1]))
    
    if log_cnt % 1500 == 0:
        end_time = datetime.now()
        print('%s, [%s] end predict test cost %s' % (end_time, proc_id, (end_time - start_time).total_seconds()))
    return test_res


def get_test_res6(model, if_gpu, test_file_no, test_dir
    #, index_list, all_samples
    ,user_feature_list, url_feature_list, user_url_dict, url_neighb_urls
    ,uid_no_follower_dict, user_neighb_users, num_limit_list, if_url_short=True, device_id=0):
    start_time = datetime.now()
    proc_id = os.getpid()
    if test_file_no % 1500 == 0:
        print('%s, [%s] start predict test' % (start_time, proc_id))
    test_res = []
    user_idxs, url_idxs, url_word_idxs, url_char_idxs, tweets_word_idxs \
        , tweets_char_idxs, labels = reader.read_batch_text_data(test_file_no, test_dir)
    
    user_idxs, url_idxs, labels, user_f_list, url_f_list, consume_urls_f_list, neighb_urls_f_list \
        ,friend_users_f_list, neighb_users_f_list = reader.read_batch_f_data_by_idx(user_idxs, url_idxs, labels
                        ,user_feature_list, url_feature_list, user_url_dict, url_neighb_urls
                        ,uid_no_follower_dict, user_neighb_users, num_limit_list, 'test')
    if if_url_short:
        url_word_idxs = url_word_idxs[:, 0:4, :]
        url_char_idxs = url_char_idxs[:, 0:4, :, :]

    if test_file_no == 0:
        print('test url word char:', url_word_idxs.shape, url_char_idxs.shape)

    #assert torch.equal(user_idxs, user_idxs1)
    #assert torch.equal(url_idxs, url_idxs1)
    #assert torch.equal(labels, labels1)

    if if_gpu is True:
        #model.cuda(device_id)
        user_idxs = user_idxs.cuda(device_id)
        url_idxs = url_idxs.cuda(device_id)
        labels = labels.cuda(device_id)
        user_f_list = user_f_list.cuda(device_id)
        url_f_list = url_f_list.cuda(device_id)
        consume_urls_f_list = consume_urls_f_list.cuda(device_id)
        neighb_urls_f_list = neighb_urls_f_list.cuda(device_id)
        friend_users_f_list = friend_users_f_list.cuda(device_id)
        neighb_users_f_list = neighb_users_f_list.cuda(device_id)

        url_word_idxs = url_word_idxs.cuda(device_id)
        url_char_idxs = url_char_idxs.cuda(device_id)
        tweets_word_idxs = tweets_word_idxs.cuda(device_id)
        tweets_char_idxs = tweets_char_idxs.cuda(device_id)
    else:
        model.cpu()
    model.eval()
    out = model(user_idxs, url_idxs, user_f_list, url_f_list
                ,consume_urls_f_list, neighb_urls_f_list
                ,friend_users_f_list, neighb_users_f_list
                ,url_word_idxs, url_char_idxs, tweets_word_idxs, tweets_char_idxs)
    test_res.extend(list(out.cpu().data.numpy()[:, 1]))
    end_time = datetime.now()
    if test_file_no % 1500 == 0:
        print('%s, [%s] end predict test cost %s' % (end_time, proc_id, (end_time - start_time).total_seconds()))
    return test_res


def get_test_res6_gcn(model, if_gpu
    #,test_file_no, test_dir
    ,index_list, all_samples
    ,user_feature_list, url_feature_list
    ,user_url_weight, user_user_weight, url_url_weight
    ,user_url_dict, url_user_dict, user_neighb_users, url_neighb_urls, device_id=0, log_cnt=0
    ,mode='train', num=3):
    if log_cnt % 1500 == 0:
        start_time = datetime.now()
        proc_id = os.getpid()
        print('%s, [%s] start predict test %d, %s' % (start_time, proc_id, len(index_list), index_list[0]))

    target_dir =  DATA_DIR + os.sep + 'gcn/test/'
    test_res = []
    user_idxs, url_idxs, labels \
    ,user_f_list, url_f_list \
    ,user_1_user_f_list, user_1_user_weight \
    ,user_1_url_f_list, user_1_url_weight \
    ,user_2_user_f_list, user_2_user_weight \
    ,user_2_url_f_list, user_2_url_weight \
    ,url_2_user_f_list, url_2_user_weight \
    ,url_2_url_f_list, url_2_url_weight = reader.read_batch_gcn_data(log_cnt, target_dir, v_num=num)
    '''reader.build_user_relation_by_sample(index_list, all_samples
        ,user_feature_list, url_feature_list
        ,user_url_weight, user_user_weight, url_url_weight
        ,user_url_dict, url_user_dict, user_neighb_users, url_neighb_urls
        ,mode=mode, num=num)'''
    #
    if if_gpu:
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

    model.eval()
    out = model(url_idxs, user_f_list, url_f_list
                ,user_1_user_f_list, user_1_user_weight
                ,user_1_url_f_list, user_1_url_weight
                ,user_2_user_f_list, user_2_user_weight
                ,user_2_url_f_list, user_2_url_weight
                ,url_2_user_f_list, url_2_user_weight
                ,url_2_url_f_list, url_2_url_weight)
    test_res.extend(list(out.cpu().data.numpy()[:, 1]))

    if log_cnt % 1500 == 0:
        end_time = datetime.now()
        print('%s, [%s] end predict test cost %s' % (end_time, proc_id, (end_time - start_time).total_seconds()))
    return test_res


def get_test_res6_main(model, if_gpu
    ,index_list, all_samples
    ,user_feature_list, url_feature_list
    ,user_url_weight, user_user_weight, url_url_weight
    ,user_url_dict, url_user_dict, user_neighb_users, url_neighb_urls, device_id, log_cnt
    ,mode, num
    ,uid_no_follower_dict, num_limit_list):
    
    if log_cnt % 1500 == 0:
        start_time = datetime.now()
        proc_id = os.getpid()
        print('%s, [%s] start predict test %d, %s' % (start_time, proc_id, len(index_list), index_list[0]))
    
    target_dir =  DATA_DIR + os.sep + 'gcn/test/'
    user_idxs, url_idxs, labels \
    ,user_f_list, url_f_list \
    ,user_1_user_f_list, user_1_user_weight \
    ,user_1_url_f_list, user_1_url_weight \
    ,user_2_user_f_list, user_2_user_weight \
    ,user_2_url_f_list, user_2_url_weight \
    ,url_2_user_f_list, url_2_user_weight \
    ,url_2_url_f_list, url_2_url_weight = reader.read_batch_gcn_data(log_cnt, target_dir, v_num=num)

    target_dir =  DATA_DIR + os.sep + 'f/test/'
    user_idxs1, url_idxs1, labels1, user_f_list1, url_f_list1, consume_urls_f_list, neighb_urls_f_list \
        ,friend_users_f_list, neighb_users_f_list = reader.read_batch_f_data(log_cnt, target_dir, num_limit_list)

    '''assert torch.equal(user_idxs, user_idxs1)
    assert torch.equal(url_idxs, url_idxs1)
    assert torch.equal(labels, labels1)
    assert torch.equal(user_f_list, user_f_list1)
    assert torch.equal(url_f_list, url_f_list1)
    print('check test done1')
    
    user_idxs = torch.tensor([all_samples[index][0] for index in index_list], dtype=torch.long) # (N, 1)
    url_idxs = torch.tensor([all_samples[index][1] for index in index_list], dtype=torch.long)  # (N, 1)
    labels = torch.tensor([all_samples[index][2] for index in index_list], dtype=torch.long)    # (N, 1)
    assert torch.equal(user_idxs, user_idxs1)
    assert torch.equal(url_idxs, url_idxs1)
    assert torch.equal(labels, labels1)
    print('check test done2')'''

    if if_gpu:
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

    model.eval()
    test_res = []
    out = model(url_idxs, user_f_list, url_f_list
                ,user_1_user_f_list, user_1_user_weight
                ,user_1_url_f_list, user_1_url_weight
                ,user_2_user_f_list, user_2_user_weight
                ,user_2_url_f_list, user_2_url_weight
                ,url_2_user_f_list, url_2_user_weight
                ,url_2_url_f_list, url_2_url_weight
                ,consume_urls_f_list, neighb_urls_f_list
                ,friend_users_f_list, neighb_users_f_list)
    test_res.extend(list(out.cpu().data.numpy()[:, 1]))

    if log_cnt % 1500 == 0:
        end_time = datetime.now()
        print('%s, [%s] end predict test cost %s' % (end_time, proc_id, (end_time - start_time).total_seconds()))
    return test_res


def get_test_res6_pre(model, if_gpu
    #,test_file_no, test_dir
    ,index_list, samples
    ,user_feature_list, url_feature_list
    ,device_id=0, log_cnt=0):
    if log_cnt % 1500 == 0:
        start_time = datetime.now()
        proc_id = os.getpid()
        print('%s, [%s] start predict test %d, %s' % (start_time, proc_id, len(index_list), index_list[0]))

    test_res = []
    user_idxs, url_idxs, labels \
    ,user_f_list, url_f_list = reader.read_user_url_f_tensor_by_sample(index_list, samples
                ,user_feature_list, url_feature_list)

    if if_gpu:
        user_idxs = user_idxs.cuda(device_id)
        url_idxs = url_idxs.cuda(device_id)
        labels = labels.cuda(device_id)
        user_f_list = user_f_list.cuda(device_id)
        url_f_list = url_f_list.cuda(device_id)

    model.eval()
    out = model(user_f_list, url_f_list)
    test_res.extend(list(out.cpu().data.numpy()[:, 1]))

    if log_cnt % 1500 == 0:
        end_time = datetime.now()
        print('%s, [%s] end predict test cost %s' % (end_time, proc_id, (end_time - start_time).total_seconds()))
    return test_res
