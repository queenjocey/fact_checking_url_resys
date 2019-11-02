import os
import json
import copy
import random
import torch
import multiprocessing
from multiprocessing import Manager

import numpy as np

from common import utils


# train
def read_train_and_test_sample(train_json_file, test_file='/home/dyou/url_recsys/dl_data/test.txt'):
    with open(train_json_file) as f:
        all_samples = json.loads(f.read())
    print('train case:', len(all_samples), all_samples[0])
    
    # test
    with open(test_file) as f:
        lines = f.readlines()
    # uid, url_id, freq, ts
    all_test_samples = []
    for line in lines:
        uid, url_id, freq, ts = line.strip().split()
        all_test_samples.append([int(uid), int(url_id), 0 if int(freq) == 0 else 1])
    print('test case:', len(all_test_samples), all_test_samples[0])

    return all_samples, all_test_samples


# read fix batch case by no
def read_batch_text_data(no, file_path):
    #print (datetime.now(), 'batch size:', len(index_list), 'batch start:', no)
    user_idxs = torch.load(file_path + os.sep + str(no) + '_user_idxs')
    url_idxs = torch.load(file_path + os.sep + str(no) + '_url_idxs')

    url_word_idxs = torch.load(file_path + os.sep + str(no) + '_url_word_idxs')
    #url_word_inverse_idxs = get_inverse_idx(url_word_idxs, 'url', 'word')

    url_char_idxs = torch.load(file_path + os.sep + str(no) + '_url_char_idxs')
    #url_char_inverse_idxs = get_inverse_idx(url_char_idxs, 'url', 'char')

    tweets_word_idxs = torch.load(file_path + os.sep + str(no) + '_tweets_word_idxs')
    #tweets_word_inverse_idxs = get_inverse_idx(tweets_word_idxs, 'tweet', 'word')

    tweets_char_idxs = torch.load(file_path + os.sep + str(no) + '_tweets_char_idxs')
    #tweets_char_inverse_idxs = get_inverse_idx(tweets_char_idxs, 'tweet', 'char')

    labels = torch.load(file_path + os.sep + str(no) + '_labels')
    #print(type(labels))
    if type(labels) != torch.Tensor:
        labels = torch.tensor(labels)
    #print (datetime.now(), 'batch size:', len(index_list), 'batch end:', no)
    return user_idxs, url_idxs, url_word_idxs, url_char_idxs, tweets_word_idxs, tweets_char_idxs, labels
    '''user_idxs, url_idxs, url_word_idxs, url_word_inverse_idxs, url_char_idxs, url_char_inverse_idxs \
            , tweets_word_idxs, tweets_word_inverse_idxs, tweets_char_idxs, tweets_char_inverse_idxs, labels'''


# only read user url labels
def read_batch_user_url_label(no, file_path):
    user_idxs = torch.load(file_path + os.sep + str(no) + '_user_idxs')
    url_idxs = torch.load(file_path + os.sep + str(no) + '_url_idxs')
    labels = torch.load(file_path + os.sep + str(no) + '_labels')
    if type(labels) != torch.Tensor:
        labels = torch.tensor(labels)
    
    return user_idxs, url_idxs, labels


def read_batch_f_data(i, target_dir, num_limit_list):#=[CONSUM_NUM, NEIGHB_URL_NUM, FRIEND_NUM, NEIGHT_USER_NUM]):
    user_idxs = torch.load(target_dir + os.sep + 'user_idxs_' + str(i))
    url_idxs = torch.load(target_dir + os.sep + 'url_idxs_' + str(i))
    labels = torch.load(target_dir + os.sep + 'labels_' + str(i))
    user_f_list = torch.load(target_dir + os.sep + 'user_f_list_' + str(i))
    url_f_list = torch.load(target_dir + os.sep + 'url_f_list_' + str(i))
    consume_urls_f_list = torch.load(target_dir + os.sep + 'consume_urls_f_list_' + str(i))
    neighb_urls_f_list = torch.load(target_dir + os.sep + 'neighb_urls_f_list_' + str(i))
    friend_users_f_list = torch.load(target_dir + os.sep + 'friend_users_f_list_' + str(i))
    neighb_users_f_list = torch.load(target_dir + os.sep + 'neighb_users_f_list_' + str(i))

    consume_urls_f_list = consume_urls_f_list[:, 0:num_limit_list[0], :]
    neighb_urls_f_list = neighb_urls_f_list[:, 0:num_limit_list[1], :]
    friend_users_f_list = friend_users_f_list[:, 0:num_limit_list[2], :]
    neighb_users_f_list = neighb_users_f_list[:, 0:num_limit_list[3], :]

    return user_idxs, url_idxs, labels, user_f_list, url_f_list, consume_urls_f_list, neighb_urls_f_list \
        ,friend_users_f_list, neighb_users_f_list


def read_batch_gcn_data(i, target_dir, v_num=3):
    user_idxs = torch.load(target_dir + os.sep + 'user_idxs_' + str(i))
    url_idxs = torch.load(target_dir + os.sep + 'url_idxs_' + str(i))
    labels = torch.load(target_dir + os.sep + 'labels_' + str(i)) 
    user_f_list = torch.load(target_dir + os.sep + 'user_f_list_' + str(i))
    url_f_list = torch.load(target_dir + os.sep + 'url_f_list_' + str(i)) 
    user_1_user_f_list = torch.load(target_dir + os.sep + 'user_1_user_f_list_' + str(i))
    user_1_user_weight = torch.load(target_dir + os.sep + 'user_1_user_weight_' + str(i)) 
    user_1_url_f_list = torch.load(target_dir + os.sep + 'user_1_url_f_list_' + str(i))
    user_1_url_weight = torch.load(target_dir + os.sep + 'user_1_url_weight_' + str(i)) 
    user_2_user_f_list = torch.load(target_dir + os.sep + 'user_2_user_f_list_' + str(i))
    user_2_user_weight = torch.load(target_dir + os.sep + 'user_2_user_weight_' + str(i)) 
    user_2_url_f_list = torch.load(target_dir + os.sep + 'user_2_url_f_list_' + str(i))
    user_2_url_weight = torch.load(target_dir + os.sep + 'user_2_url_weight_' + str(i)) 
    url_2_user_f_list = torch.load(target_dir + os.sep + 'url_2_user_f_list_' + str(i))
    url_2_user_weight = torch.load(target_dir + os.sep + 'url_2_user_weight_' + str(i)) 
    url_2_url_f_list = torch.load(target_dir + os.sep + 'url_2_url_f_list_' + str(i))
    url_2_url_weight = torch.load(target_dir + os.sep + 'url_2_url_weight_' + str(i))

    user_1_user_f_list = user_1_user_f_list[:, 0:v_num, :]
    user_1_user_weight = user_1_user_weight[:, 0:v_num]
    user_1_url_f_list = user_1_url_f_list[:, 0:v_num, :]
    user_1_url_weight = user_1_url_weight[:, 0:v_num]
    user_2_user_f_list = user_2_user_f_list[:, 0:v_num, 0:v_num, :]
    user_2_user_weight = user_2_user_weight[:, 0:v_num, 0:v_num]
    user_2_url_f_list = user_2_url_f_list[:, 0:v_num, 0:v_num, :]
    user_2_url_weight = user_2_url_weight[:, 0:v_num, 0:v_num]
    url_2_user_f_list = url_2_user_f_list[:, 0:v_num, 0:v_num, :]
    url_2_user_weight = url_2_user_weight[:, 0:v_num, 0:v_num]
    url_2_url_f_list = url_2_url_f_list[:, 0:v_num, 0:v_num, :]
    url_2_url_weight = url_2_url_weight[:, 0:v_num, 0:v_num]

    return user_idxs, url_idxs, labels \
        ,user_f_list, url_f_list \
        ,user_1_user_f_list, user_1_user_weight \
        ,user_1_url_f_list, user_1_url_weight \
        ,user_2_user_f_list, user_2_user_weight \
        ,user_2_url_f_list, user_2_url_weight \
        ,url_2_user_f_list, url_2_user_weight \
        ,url_2_url_f_list, url_2_url_weight



def read_user_url_f_tensor_by_sample(index_list, samples, user_feature_list, url_feature_list):
    # user features
    # [int(uid_no), follow_cnt_bin, follow_in_cnt_bin, post_url_bin, post_cnt_bin
    #            ,favorite_cate_no, favorite_site_no, uid, name]

    ## url features
    # # [int(url_no), cate2idx[cate], site2idx[url_site], post_user_bin, post_freq_bin, url]
    user_idxs = torch.tensor([samples[index][0] for index in index_list], dtype=torch.long) # (N, 1)
    url_idxs = torch.tensor([samples[index][1] for index in index_list], dtype=torch.long)  # (N, 1)
    labels = torch.tensor([samples[index][2] for index in index_list], dtype=torch.long)    # (N, 1)

    user_f_list = []        # (N, f_num, 1)
    url_f_list = []         # (N, f_num, 1)
    count = 0
    for uid_no, url_no, label in [samples[index] for index in index_list]:
        user_f = build_user_f_tensor(user_feature_list, uid_no)
        user_f_list.append(user_f)

        url_f = build_url_f_tensor(url_feature_list, url_no)
        url_f_list.append(url_f)

    user_f_list = torch.stack(user_f_list, dim=0)
    url_f_list = torch.stack(url_f_list, dim=0)

    return user_idxs, url_idxs, labels, \
        user_f_list, url_f_list



# chose batch text data, the result is bad
def chose_batch_text_data(index_list, samples, all_tweets_word_idx, all_tweets_char_idx, all_url_word_idx, all_url_char_idx, if_url_short=True):
    user_idxs = torch.tensor([samples[index][0] for index in index_list], dtype=torch.long)
    url_idxs = torch.tensor([samples[index][1] for index in index_list], dtype=torch.long)
    labels = torch.tensor([samples[index][2] for index in index_list], dtype=torch.long)
    if if_url_short is True:
        url_word_idxs = all_url_word_idx[url_idxs][:, 0:4, :]
        url_char_idxs = all_url_char_idx[url_idxs][:, 0:4, :, :]
    else:
        url_word_idxs = all_url_word_idx[url_idxs]
        url_char_idxs = all_url_char_idx[url_idxs]

    tweets_word_idxs = all_tweets_word_idx[user_idxs]
    tweets_char_idxs = all_tweets_char_idx[user_idxs]
    return user_idxs, url_idxs, url_word_idxs, url_char_idxs, tweets_word_idxs, tweets_char_idxs, labels


# chose batch text data by pre cal tfidf
def chose_batch_text_data_by_tfidf(user_idxs, url_idxs, all_url_word_idx, all_url_char_idx, tweets_dir):
    url_word_idxs = all_url_word_idx[url_idxs]
    url_char_idxs = all_url_char_idx[url_idxs]

    tweets_word_list = []
    tweets_char_list = []
    for i in range(user_idxs.shape[0]):
        uid = user_idxs[i].data.numpy()
        url_id = url_idxs[i].data.numpy()
        tweets_word_idx = torch.load(tweets_dir + os.sep + 'tweets_word' + os.sep + str(uid) + '_' + str(url_id) + '_tweets_word_idx')
        tweets_word_list.append(tweets_word_idx)
        tweets_char_idx = torch.load(tweets_dir + os.sep + 'tweets_char' + os.sep + str(uid) + '_' + str(url_id) + '_tweets_char_idx')
        tweets_char_list.append(tweets_char_idx)

    tweets_word_idxs = torch.stack(tweets_word_list, dim=0)
    tweets_char_idxs = torch.stack(tweets_char_list, dim=0)
    return url_word_idxs, url_char_idxs, tweets_word_idxs, tweets_char_idxs


# read feature, sort by no
def read_feature(user_feature_json_file, url_feature_json_file, device_id=None, if_tensor=False):
    user_feature_list = json.load(open(user_feature_json_file, 'r'))
    user_feature_list.sort(key=lambda x: int(x[0]), reverse=False)
    user_feature_list = [f_list[0:7] for f_list in user_feature_list]
    #
    
    url_feature_list = json.load(open(url_feature_json_file, 'r'))
    url_feature_list.sort(key=lambda x: int(x[0]), reverse=False)
    url_feature_list = [f_list[0:5] for f_list in url_feature_list]

    if if_tensor:
        user_feature_list = torch.tensor(user_feature_list, dtype=torch.long)
        url_feature_list = torch.tensor(url_feature_list, dtype=torch.long)
        print('user_feature_list:', user_feature_list.shape, user_feature_list[0])
        print('url feature list:', url_feature_list.shape, url_feature_list[0])
    else:
        print('user_feature_list:', len(user_feature_list), user_feature_list[0])
        print('url feature list:', len(url_feature_list), url_feature_list[0])

    if device_id is not None:
        user_feature_list = user_feature_list.cuda(device_id)
        url_feature_list = url_feature_list.cuda(device_id)
        
    return user_feature_list, url_feature_list


# 
def read_dict_json(json_dict_file, name='json_dict'):
    json_dict = json.load(open(json_dict_file, 'r'))
    json_dict = {int(key): [int(item) for item in item_list] for key, item_list in json_dict.items()}
    #print(name, len(json_dict), list(json_dict.keys())[0:3])
    return json_dict


# 
def read_dict_json_by_weight(json_dict_file, weight_list, weight_index, num_limit, top_ratio=0.25, name='json_dict'):
    json_dict = json.load(open(json_dict_file, 'r'))
    #print (weight_list[0])
    json_dict = {
        int(key): [
            [int(weight_list[int(item)][weight_index].data.numpy())+1, int(item)] 
            for item in item_list if int(item) != int(key)
        ] 
        for key, item_list in json_dict.items()
    }
    #print (json_dict[0])
    new_dict = {}
    for key in json_dict:
        value = json_dict[key]
        value.sort(key=lambda x: x[0], reverse=True)
        max_num = max(num_limit, int(len(value) * top_ratio))
        new_dict[key] = value[0:max_num]
    print(name, len(new_dict))
    return new_dict


def read_dict_json_by_ratio(json_dict_file, num_limit, top_ratio=0.25, name='json_dict'):
    json_dict = json.load(open(json_dict_file, 'r'))
    new_dict = {}
    for key in json_dict:
        value = json_dict[key]
        max_num = max(num_limit, int(len(value) * top_ratio))
        new_dict[int(key)] = value[0:max_num]
    #print(new_dict[10])
    print(name, len(new_dict))
    return new_dict


# consum url_neighb friend user_neighb
# item_list = [[weight, no], [weight, no]], sorted
def chose_items_by_random(item_list, num=3, debug=False):
    weight_list = []
    while len(weight_list) < num and len(item_list) > 0:
        i = random.randint(0, len(item_list)-1)
        weight_list.append(item_list[i])
        item_list.remove(item_list[i])

    weight_list.sort(key=lambda x: x[0], reverse=True)
    new_list = [item[1] for item in weight_list]
    if debug:
        print(weight_list, new_list, 'url weight')
    while len(new_list) < num:
        new_list.append(-1)
    return new_list


# item_list = [[weight, no], [weight, no]]  sorted
def chose_top_items(item_list, num=3, debug=False):
    new_list = []
    new_list.extend([item[1] for item in item_list[0:num]])
    while len(new_list) < num:
        new_list.append(-1)
    if debug:
        print(item_list, new_list, 'test mode, weight2')
    return new_list


'''#############################################################
def chose_items_by_weight1_old(item_list, item_weights, weight_index, num=3, debug=False):
    if len(item_list) > num:
        new_list = []
        for i in range(num):
            sum_weight = np.sum([item_weights[index][weight_index] + 1 for index in item_list])
            random_weight = random.randint(1, sum_weight)
            cum_weight = 0
            for j in range(len(item_list)):
                cum_weight += item_weights[item_list[j]][weight_index]
                if random_weight <= cum_weight:
                    break
            new_list.append(item_list[j])
            item_list.remove(item_list[j])
        if debug:
            print(new_list, [item_weights[index][weight_index] for index in new_list], 'url weight')
        return new_list
    else:
        unknown_index = len(item_weights)
        item_list.extend([-1 for i in range(num)])
        #print(item_list, 'url no 3')
        return item_list[0:num]


def chose_items_by_random_old(item_list, item_weights, weight_index, num=3, debug=False):
    new_list = []
    while len(new_list) < num and len(item_list) > 0:
        i = random.randint(0, len(item_list)-1)
        new_list.append(item_list[i])
        item_list.remove(item_list[i])

    weight_list = [[item_weights[index][weight_index], index] for index in new_list]
    #weight_list.sort(key=lambda x: x[0], reverse=True)
    new_list = [item[1] for item in weight_list]
    if debug:
        print(new_list, [item_weights[index][weight_index] for index in new_list], 'url weight')
    while len(new_list) < num:
        new_list.append(-1)
    return new_list

def chose_top_items_old(item_list, item_weights, weight_index, num=3, debug=False):
    new_list = []
    #for i in range(num):
    weight_list = [[item_weights[index][weight_index], index] for index in item_list]
    weight_list.sort(key=lambda x: x[0], reverse=True)
    new_list.extend([item[1] for item in weight_list[0:num]])
    #print(new_list, [item_weights[index][weight_index] for index in new_list], 'url weight')
    while len(new_list) < num:
        new_list.append(-1)
    return new_list

################################################################## '''


def build_url_f(url_feature_list, url_no):
    # [int(url_no), cate2idx[cate], site2idx[url_site], post_user_bin, post_freq_bin, url]
    if url_no != -1:
        url_feature = url_feature_list[url_no]
        return [url_feature[0], url_feature[1], url_feature[2], url_feature[3], url_feature[4]]
    else:
        return [len(url_feature_list), 20, 6, 10, 10]
    

def build_user_f(user_feature_list, user_no):
    # [int(uid_no), follow_cnt_bin, follow_in_cnt_bin, post_url_bin, post_cnt_bin
    # ,favorite_cate_no, favorite_site_no, uid, name]
    if user_no != -1:
        user_feature = user_feature_list[user_no]
        return [user_feature[0], user_feature[1], user_feature[2], user_feature[3], user_feature[4], user_feature[5], user_feature[6]]
    else:
        return [len(user_feature_list), 10, 10, 10, 10, 20, 6]


#def read_train_batch_case 
def read_batch_f_data_by_sample(index_list, samples, user_feature_list, url_feature_list
    ,user_url_dict, url_neighb_urls, uid_no_follower_dict, user_neighb_users
    ,num_limit_list, mode='train'):
    # user features
    # [int(uid_no), follow_cnt_bin, follow_in_cnt_bin, post_url_bin, post_cnt_bin
    #            ,favorite_cate_no, favorite_site_no, uid, name]

    ## url features
    # # [int(url_no), cate2idx[cate], site2idx[url_site], post_user_bin, post_freq_bin, url]
    user_idxs = torch.tensor([samples[index][0] for index in index_list], dtype=torch.long) # (N, 1)
    url_idxs = torch.tensor([samples[index][1] for index in index_list], dtype=torch.long)  # (N, 1)
    labels = torch.tensor([samples[index][2] for index in index_list], dtype=torch.long)    # (N, 1)

    user_f_list = []        # (N, f_num, 1)
    url_f_list = []         # (N, f_num, 1)
    consume_urls_f_list = []  # (N, item_num, f_num, 1)
    neighb_urls_f_list = []  # (N, item_num, f_num, 1)
    friend_users_f_list = [] # (N, user_item_num, f_num, 1)
    neighb_users_f_list = [] # (N, user_item_num, f_num, 1)
    count = 0
    for uid_no, url_no, label in [samples[index] for index in index_list]:
        user_f = build_user_f(user_feature_list, uid_no)
        user_f_list.append(user_f)

        url_f = build_url_f(url_feature_list, url_no)
        url_f_list.append(url_f)

        # url consume urls [[weight, no], [weight, no]]
        url_consume_items = []
        for items in user_url_dict[uid_no]:
            if items[1] != url_no:
                url_consume_items.append(items)
        #print(url_consume_items)
        
        if mode == 'train':
            consume_urls = chose_items_by_random(url_consume_items, num=num_limit_list[0], debug=False)
        else:
            consume_urls = chose_top_items(url_consume_items, num=num_limit_list[0], debug=False)
        consume_urls_f = [] # (item_num, f_num, 1)
        for consume_url_no in consume_urls:
            one_f_list = build_url_f(url_feature_list, consume_url_no)
            consume_urls_f.append(one_f_list)
            #url_idx, cate_idx, site_idx, post_user_bin, post_freq_bin = build_url_f(url_feature_list, consume_url_no)
            #consume_urls_f.append([url_idx, cate_idx, site_idx, post_user_bin, post_freq_bin])
        consume_urls_f_list.append(consume_urls_f)

        # url neighb urls
        url_neighb_items = copy.deepcopy(url_neighb_urls.get(url_no, []))
        if mode == 'train':
            neighb_urls = chose_items_by_random(url_neighb_items, num=num_limit_list[1], debug=False)
        else:
            neighb_urls = chose_top_items(url_neighb_items, num=num_limit_list[1], debug=False)
        neighb_urls_f = [] # (item_num, f_num, 1)
        for neighb_url_no in neighb_urls:
            one_f_list = build_url_f(url_feature_list, neighb_url_no)
            neighb_urls_f.append(one_f_list)
        neighb_urls_f_list.append(neighb_urls_f)

        # user friend-level 
        user_follow_users = copy.deepcopy(uid_no_follower_dict[uid_no])
        if mode == 'train':
            friend_users = chose_items_by_random(user_follow_users, num=num_limit_list[2])
        else:
            friend_users = chose_top_items(user_follow_users, num=num_limit_list[2])
        follow_users_f = [] # (user_num, f_num, 1)
        for friend_no in friend_users:
            one_f_list = build_user_f(user_feature_list, friend_no)
            follow_users_f.append(one_f_list)
        friend_users_f_list.append(follow_users_f)

        # user neighb users
        user_neighb_items = copy.deepcopy(user_neighb_users[uid_no])
        if mode == 'train':
            neighb_users = chose_items_by_random(user_neighb_items, num=num_limit_list[3])
        else:
            neighb_users = chose_top_items(user_neighb_items, num=num_limit_list[3])
        neight_users_f = [] # (user_num, f_num, 1)
        for neighb_no in neighb_users:
            one_f_list = build_user_f(user_feature_list, neighb_no)
            neight_users_f.append(one_f_list)
        neighb_users_f_list.append(neight_users_f)

    user_f_list = torch.tensor(user_f_list, dtype=torch.long)
    url_f_list = torch.tensor(url_f_list, dtype=torch.long)
    consume_urls_f_list = torch.tensor(consume_urls_f_list, dtype=torch.long)
    neighb_urls_f_list = torch.tensor(neighb_urls_f_list, dtype=torch.long)
    friend_users_f_list = torch.tensor(friend_users_f_list, dtype=torch.long)
    neighb_users_f_list = torch.tensor(neighb_users_f_list, dtype=torch.long)

    return user_idxs, url_idxs, labels, \
        user_f_list, url_f_list, \
        consume_urls_f_list, neighb_urls_f_list,\
        friend_users_f_list, neighb_users_f_list


# 
def read_batch_f_data_by_idx(user_idxs, url_idxs, labels, user_feature_list, url_feature_list
    ,user_url_dict, url_neighb_urls, uid_no_follower_dict, user_neighb_users
    ,num_limit_list, mode='train'):
    # user features
    # [int(uid_no), follow_cnt_bin, follow_in_cnt_bin, post_url_bin, post_cnt_bin
    #            ,favorite_cate_no, favorite_site_no, uid, name]

    ## url features
    # # [int(url_no), cate2idx[cate], site2idx[url_site], post_user_bin, post_freq_bin, url]
    user_f_list = []        # (N, f_num, 1)
    url_f_list = []         # (N, f_num, 1)
    consume_urls_f_list = []  # (N, item_num, f_num, 1)
    neighb_urls_f_list = []  # (N, item_num, f_num, 1)
    friend_users_f_list = [] # (N, user_item_num, f_num, 1)
    neighb_users_f_list = [] # (N, user_item_num, f_num, 1)
    count = 0
    for i in range(user_idxs.shape[0]):
        uid_no, url_no = int(user_idxs[i].data.numpy()), int(url_idxs[i].data.numpy())
        #, labels[i].data.numpy()
        user_f = build_user_f(user_feature_list, uid_no)
        user_f_list.append(user_f)

        url_f = build_url_f(url_feature_list, url_no)
        url_f_list.append(url_f)

        # url consume urls [[weight, no], [weight, no]]
        url_consume_items = []
        for items in user_url_dict[uid_no]:
            if items[1] != url_no:
                url_consume_items.append(items)
        
        if mode == 'train':
            consume_urls = chose_items_by_random(url_consume_items, num=num_limit_list[0], debug=False)
        else:
            consume_urls = chose_top_items(url_consume_items, num=num_limit_list[0], debug=False)
        consume_urls_f = [] # (item_num, f_num, 1)
        for consume_url_no in consume_urls:
            one_f_list = build_url_f(url_feature_list, consume_url_no)
            consume_urls_f.append(one_f_list)
            #url_idx, cate_idx, site_idx, post_user_bin, post_freq_bin = build_url_f(url_feature_list, consume_url_no)
            #consume_urls_f.append([url_idx, cate_idx, site_idx, post_user_bin, post_freq_bin])
        consume_urls_f_list.append(consume_urls_f)

        # url neighb urls
        url_neighb_items = copy.deepcopy(url_neighb_urls.get(url_no, []))
        if mode == 'train':
            neighb_urls = chose_items_by_random(url_neighb_items, num=num_limit_list[1], debug=False)
        else:
            neighb_urls = chose_top_items(url_neighb_items, num=num_limit_list[1], debug=False)
        neighb_urls_f = [] # (item_num, f_num, 1)
        for neighb_url_no in neighb_urls:
            one_f_list = build_url_f(url_feature_list, neighb_url_no)
            neighb_urls_f.append(one_f_list)
        neighb_urls_f_list.append(neighb_urls_f)

        # user friend-level 
        user_follow_users = copy.deepcopy(uid_no_follower_dict[uid_no])
        if mode == 'train':
            friend_users = chose_items_by_random(user_follow_users, num=num_limit_list[2])
        else:
            friend_users = chose_top_items(user_follow_users, num=num_limit_list[2])
        follow_users_f = [] # (user_num, f_num, 1)
        for friend_no in friend_users:
            one_f_list = build_user_f(user_feature_list, friend_no)
            follow_users_f.append(one_f_list)
        friend_users_f_list.append(follow_users_f)

        # user neighb users
        user_neighb_items = copy.deepcopy(user_neighb_users[uid_no])
        if mode == 'train':
            neighb_users = chose_items_by_random(user_neighb_items, num=num_limit_list[3])
        else:
            neighb_users = chose_top_items(user_neighb_items, num=num_limit_list[3])
        neight_users_f = [] # (user_num, f_num, 1)
        for neighb_no in neighb_users:
            one_f_list = build_user_f(user_feature_list, neighb_no)
            neight_users_f.append(one_f_list)
        neighb_users_f_list.append(neight_users_f)

    user_f_list = torch.tensor(user_f_list, dtype=torch.long)
    url_f_list = torch.tensor(url_f_list, dtype=torch.long)
    consume_urls_f_list = torch.tensor(consume_urls_f_list, dtype=torch.long)
    neighb_urls_f_list = torch.tensor(neighb_urls_f_list, dtype=torch.long)
    friend_users_f_list = torch.tensor(friend_users_f_list, dtype=torch.long)
    neighb_users_f_list = torch.tensor(neighb_users_f_list, dtype=torch.long)

    return user_idxs, url_idxs, labels, \
        user_f_list, url_f_list, \
        consume_urls_f_list, neighb_urls_f_list,\
        friend_users_f_list, neighb_users_f_list



########## gcn
def chose_item_by_mode(item_list, num=3, debug=False, mode='train'):
    copy_list = copy.deepcopy(item_list)
    if mode == 'train':
        return chose_items_by_random(copy_list, num, debug)
    else:
        return chose_top_items(copy_list, num, debug)


def build_url_f_tensor(url_feature_list, url_no):
    # [int(url_no), cate2idx[cate], site2idx[url_site], post_user_bin, post_freq_bin, url]
    if url_no != -1:
        url_feature = url_feature_list[url_no]#[0:5]
    else:
        url_feature = torch.tensor([len(url_feature_list), 20, 6, 10, 10], dtype=torch.long)
        if str(url_feature_list.device) != 'cpu':
            device_id = int(str(url_feature_list.device).split(':')[1])
            url_feature = url_feature.cuda(device_id)
    return url_feature


def build_user_f_tensor(user_feature_list, user_no):
    # [int(uid_no), follow_cnt_bin, follow_in_cnt_bin, post_url_bin, post_cnt_bin
    # ,favorite_cate_no, favorite_site_no, uid, name]
    if user_no != -1:
        user_feature = user_feature_list[user_no]#[0:7]
    else:
        user_feature = torch.tensor([len(user_feature_list), 10, 10, 10, 10, 20, 6], dtype=torch.long)
        if str(user_feature_list.device) != 'cpu':
            device_id = int(str(user_feature_list.device).split(':')[1])
            user_feature = user_feature.cuda(device_id)
    return user_feature


def read_weights(weight_json_file, D=1.0, s=1.0, if_no_use=False, device_id=0):
    #print('start', weight_json_file)
    weight_list = json.load(open(weight_json_file))
    weight_list = torch.tensor(weight_list, dtype=torch.float)
    #print('read done', weight_json_file)
    #max(0, pmi - log(s)), pmi = log(ij_cnt * D / i_cnt * j_cnt)
    if D > 1.0:
        weight_list = torch.relu(torch.log(D * weight_list) - np.log(s))
    if device_id is not None:
        weight_list = weight_list.cuda(device_id)
    if if_no_use:
        weight_list[weight_list > 0] = 1.0
    weight_cnt = torch.sum(weight_list > 0)
    weight_sum = torch.sum(weight_list)
    print(weight_json_file, weight_list.shape, weight_cnt, weight_sum, weight_sum / weight_cnt)
    return weight_list


def build_f_and_weight_tensor(center_no, center_type, item_list, item_type, features_tensor, weight_tensor):
    item_f_list = [] # (num, f_num)
    item_weight_list = [] # (num)
    for item_no in item_list:
        if item_type == 'user':
            f_tensor = build_user_f_tensor(features_tensor, item_no)
        else:
            f_tensor = build_url_f_tensor(features_tensor, item_no)
        item_f_list.append(f_tensor)
        if center_type == 'url' and item_type == 'user':
            item_weight_list.append(weight_tensor[item_no][center_no])
        else:
            item_weight_list.append(weight_tensor[center_no][item_no])
    item_f_list = torch.stack(item_f_list, dim=0)
    item_weight_list = torch.stack(item_weight_list, dim=0)
    return item_f_list, item_weight_list


def build_user_relation_by_sample(index_list, samples
    ,user_feature_list, url_feature_list
    ,user_url_weight, user_user_weight, url_url_weight
    ,user_url_dict, url_user_dict, user_neighb_users, url_neighb_urls
    ,mode='train', num=3, return_dict=None, key=0):
    # user features
    # [int(uid_no), follow_cnt_bin, follow_in_cnt_bin, post_url_bin, post_cnt_bin
    #            ,favorite_cate_no, favorite_site_no, uid, name]

    ## url features
    # # [int(url_no), cate2idx[cate], site2idx[url_site], post_user_bin, post_freq_bin, url]
    user_idxs = torch.tensor([samples[index][0] for index in index_list], dtype=torch.long) # (N, 1)
    url_idxs = torch.tensor([samples[index][1] for index in index_list], dtype=torch.long)  # (N, 1)
    labels = torch.tensor([samples[index][2] for index in index_list], dtype=torch.long)    # (N, 1)

    user_f_list, url_f_list = [], []  # (N, f_num)
    user_1_user_f_list, user_1_user_weight = [], []     # (N, num, f_num), (N, num)
    user_1_url_f_list, user_1_url_weight = [], []   # (N, num, f_num), (N, num)
    user_2_user_f_list, user_2_user_weight = [], []     # (N, num, num, f_num), , (N, num, num)
    user_2_url_f_list, user_2_url_weight = [], []   # (N, num, num, f_num), (N, num, num)
    url_2_user_f_list, url_2_user_weight = [], []   # (N, num, num, f_num), (N, num, num)
    url_2_url_f_list, url_2_url_weight = [], []     # (N, num, num, f_num), (N, num, num)


    for uid_no, url_no, label in [samples[index] for index in index_list]:
        user_f = build_user_f_tensor(user_feature_list, uid_no)
        user_f_list.append(user_f)

        url_f = build_url_f_tensor(url_feature_list, url_no)
        url_f_list.append(url_f)

        # user 1 user
        user_1_users = chose_item_by_mode(user_neighb_users[uid_no], num, False, mode)
        # (num, f_num), (num)
        one_graph_fs, one_graph_weights = build_f_and_weight_tensor(uid_no, 'user', user_1_users, 'user', user_feature_list, user_user_weight)
        user_1_user_f_list.append(one_graph_fs)   # (N, num, f_num)
        user_1_user_weight.append(one_graph_weights) # (N, num)

        # user 2 user
        user_2_users_fs, user_2_users_weights = [], []
        # user 2 url
        user_2_urls_fs, user_2_urls_weights = [], []
        for uid in user_1_users:
            user_2_users = chose_item_by_mode(user_neighb_users.get(uid, []), num, False, mode)
            # (num, f_num), (num)
            one_graph_fs, one_graph_weights = build_f_and_weight_tensor(uid, 'user', user_2_users, 'user', user_feature_list, user_user_weight)
            user_2_users_fs.append(one_graph_fs)
            user_2_users_weights.append(one_graph_weights)

            user_2_urls = chose_item_by_mode(user_url_dict.get(uid, []), num, False, mode)
            one_graph_fs, one_graph_weights = build_f_and_weight_tensor(uid, 'user', user_2_urls, 'url', url_feature_list, user_url_weight)
            user_2_urls_fs.append(one_graph_fs)
            user_2_urls_weights.append(one_graph_weights)
        user_2_users_fs = torch.stack(user_2_users_fs, dim=0) # (num, num, f_num)
        user_2_users_weights = torch.stack(user_2_users_weights, dim=0) # (num, num)
        user_2_urls_fs = torch.stack(user_2_urls_fs, dim=0)
        user_2_urls_weights = torch.stack(user_2_urls_weights, dim=0)

        user_2_user_f_list.append(user_2_users_fs)   # (N, num, num, f_num)
        user_2_user_weight.append(user_2_users_weights)  #(N, num, num)
        user_2_url_f_list.append(user_2_urls_fs)
        user_2_url_weight.append(user_2_urls_weights)

        # user 1 url
        user_1_urls = chose_item_by_mode(user_url_dict[uid_no], num, False, mode)
        # (num, f_num), (num)
        one_graph_fs, one_graph_weights = build_f_and_weight_tensor(uid_no, 'user', user_1_urls, 'url', url_feature_list, user_url_weight)
        user_1_url_f_list.append(one_graph_fs)  # (N, num, f_num)
        user_1_url_weight.append(one_graph_weights) # (N, num)

        # url 2 user
        url_2_users_fs, url_2_users_weights = [], []
        # url 2 url
        url_2_urls_fs, url_2_urls_weights = [], []
        for url_id in user_1_urls:
            url_2_users = chose_item_by_mode(url_user_dict.get(url_id, []), num, False, mode)
            one_graph_fs, one_graph_weights = build_f_and_weight_tensor(url_id, 'url', url_2_users, 'user', user_feature_list, user_url_weight)
            url_2_users_fs.append(one_graph_fs) # (num, num, f_num)
            url_2_users_weights.append(one_graph_weights) # (num, num)

            url_2_urls = chose_item_by_mode(url_neighb_urls.get(url_id, []), num, False, mode)
            one_graph_fs, one_graph_weights = build_f_and_weight_tensor(url_id, 'url', url_2_urls, 'url', url_feature_list, url_url_weight)
            url_2_urls_fs.append(one_graph_fs)
            url_2_urls_weights.append(one_graph_weights)
        url_2_users_fs = torch.stack(url_2_users_fs, dim=0) # (num, num, f_num)
        url_2_users_weights = torch.stack(url_2_users_weights, dim=0) # (num, num)
        url_2_urls_fs = torch.stack(url_2_urls_fs, dim=0) # (num, num, f_num)
        url_2_urls_weights = torch.stack(url_2_urls_weights, dim=0) # (num, num)

        url_2_user_f_list.append(url_2_users_fs)   # (N, num, num, f_num)
        url_2_user_weight.append(url_2_users_weights) # (N, num, num)
        url_2_url_f_list.append(url_2_urls_fs)
        url_2_url_weight.append(url_2_urls_weights)

    user_f_list = torch.stack(user_f_list, dim=0)
    url_f_list = torch.stack(url_f_list, dim=0)

    user_1_user_f_list = torch.stack(user_1_user_f_list, dim=0)     # (N, num, f_num)
    user_1_user_weight = torch.stack(user_1_user_weight, dim=0)     # (N, num)
    user_2_user_f_list = torch.stack(user_2_user_f_list, dim=0)     # (N, num, num, f_num)
    user_2_user_weight = torch.stack(user_2_user_weight, dim=0)     # (N, num, num)
    user_2_url_f_list = torch.stack(user_2_url_f_list, dim=0)       # (N, num, num, f_num)
    user_2_url_weight = torch.stack(user_2_url_weight, dim=0)       # (N, num, num)
    
    user_1_url_f_list = torch.stack(user_1_url_f_list, dim=0)
    user_1_url_weight = torch.stack(user_1_url_weight, dim=0)
    url_2_user_f_list = torch.stack(url_2_user_f_list, dim=0)
    url_2_user_weight = torch.stack(url_2_user_weight, dim=0)
    url_2_url_f_list = torch.stack(url_2_url_f_list, dim=0)
    url_2_url_weight = torch.stack(url_2_url_weight, dim=0)

    if return_dict is not None:
        return_dict[key] = (user_idxs, url_idxs, labels \
        ,user_f_list, url_f_list \
        ,user_1_user_f_list, user_1_user_weight \
        ,user_1_url_f_list, user_1_url_weight \
        ,user_2_user_f_list, user_2_user_weight \
        ,user_2_url_f_list, user_2_url_weight \
        ,url_2_user_f_list, url_2_user_weight \
        ,url_2_url_f_list, url_2_url_weight)
    else:
        return (user_idxs, url_idxs, labels \
        ,user_f_list, url_f_list \
        ,user_1_user_f_list, user_1_user_weight \
        ,user_1_url_f_list, user_1_url_weight \
        ,user_2_user_f_list, user_2_user_weight \
        ,user_2_url_f_list, user_2_url_weight \
        ,url_2_user_f_list, url_2_user_weight \
        ,url_2_url_f_list, url_2_url_weight)


def build_user_relation_by_sample_queue(index_lists, samples
    ,user_feature_list, url_feature_list
    ,user_url_weight, user_user_weight, url_url_weight
    ,user_url_dict, url_user_dict, user_neighb_users, url_neighb_urls
    ,mode, num, queue):
    # user features
    # [int(uid_no), follow_cnt_bin, follow_in_cnt_bin, post_url_bin, post_cnt_bin
    #            ,favorite_cate_no, favorite_site_no, uid, name]

    ## url features
    # # [int(url_no), cate2idx[cate], site2idx[url_site], post_user_bin, post_freq_bin, url]
    for index_list in index_lists:
        res = build_user_relation_by_sample(index_list, samples
            ,user_feature_list, url_feature_list
            ,user_url_weight, user_user_weight, url_url_weight
            ,user_url_dict, url_user_dict, user_neighb_users, url_neighb_urls
            ,mode, num, return_dict=None, key=0)
        queue.put(res, block=True)


def build_user_relation_by_sample_mul(index_list, samples
    ,user_feature_list, url_feature_list
    ,user_url_weight, user_user_weight, url_url_weight
    ,user_url_dict, url_user_dict, user_neighb_users, url_neighb_urls
    ,mode='train', num=3):
    # user features
    # [int(uid_no), follow_cnt_bin, follow_in_cnt_bin, post_url_bin, post_cnt_bin
    #            ,favorite_cate_no, favorite_site_no, uid, name]

    ## url features
    # # [int(url_no), cate2idx[cate], site2idx[url_site], post_user_bin, post_freq_bin, url]
    user_idxs = torch.tensor([samples[index][0] for index in index_list], dtype=torch.long) # (N, 1)
    url_idxs = torch.tensor([samples[index][1] for index in index_list], dtype=torch.long)  # (N, 1)
    labels = torch.tensor([samples[index][2] for index in index_list], dtype=torch.long)    # (N, 1)

    user_f_list, url_f_list = [], []  # (N, f_num)
    user_1_user_f_list, user_1_user_weight = [], []     # (N, num, f_num), (N, num)
    user_1_url_f_list, user_1_url_weight = [], []   # (N, num, f_num), (N, num)
    user_2_user_f_list, user_2_user_weight = [], []     # (N, num, num, f_num), , (N, num, num)
    user_2_url_f_list, user_2_url_weight = [], []   # (N, num, num, f_num), (N, num, num)
    url_2_user_f_list, url_2_user_weight = [], []   # (N, num, num, f_num), (N, num, num)
    url_2_url_f_list, url_2_url_weight = [], []     # (N, num, num, f_num), (N, num, num)


    index_lists = utils.generate_batch_index(32, 0, index_list)
    '''pool = multiprocessing.Pool(processes=4) # 创建4个进程
    results = []
    for one_list in index_lists:
        results.append(pool.apply_async(build_user_relation_by_sample, (one_list, samples
                            ,user_feature_list, url_feature_list
                            ,user_url_weight, user_user_weight, url_url_weight
                            ,user_url_dict, url_user_dict
                            ,user_neighb_users, url_neighb_urls
                            ,mode, num)
                        )               
        )
    pool.close()
    pool.join()
    for res in results:
        one_res = res.get()
        user_f_list.append(one_res[3])
        url_f_list.append(one_res[4])
        user_1_user_f_list.append(one_res[5])
        user_1_user_weight.append(one_res[6])
        user_1_url_f_list.append(one_res[7])
        user_1_url_weight.append(one_res[8])
        user_2_user_f_list.append(one_res[9])
        user_2_user_weight.append(one_res[10])
        user_2_url_f_list.append(one_res[11])
        user_2_url_weight.append(one_res[12])
        url_2_user_f_list.append(one_res[13])
        url_2_user_weight.append(one_res[14])
        url_2_url_f_list.append(one_res[15])
        url_2_url_weight.append(one_res[16])
    '''
    p_list = []
    manager = Manager()
    count = 0
    return_dict = manager.dict()
    for one_list in index_lists:
        p = multiprocessing.Process(target=build_user_relation_by_sample, args=(one_list, samples
                            ,user_feature_list, url_feature_list
                            ,user_url_weight, user_user_weight, url_url_weight
                            ,user_url_dict, url_user_dict
                            ,user_neighb_users, url_neighb_urls
                            ,mode, num, return_dict, count))
        count += 1
        p_list.append(p)
        p.start()

    for p in p_list:
        p.join()
    print(len(p_list), 'done')
    start = datetime.now()
    for i in range(count):
        user_f_list.append(return_dict[i][3])
        url_f_list.append(return_dict[i][4])
        user_1_user_f_list.append(return_dict[i][5])
        user_1_user_weight.append(return_dict[i][6])
        user_1_url_f_list.append(return_dict[i][7])
        user_1_url_weight.append(return_dict[i][8])
        user_2_user_f_list.append(return_dict[i][9])
        user_2_user_weight.append(return_dict[i][10])
        user_2_url_f_list.append(return_dict[i][11])
        user_2_url_weight.append(return_dict[i][12])
        url_2_user_f_list.append(return_dict[i][13])
        url_2_user_weight.append(return_dict[i][14])
        url_2_url_f_list.append(return_dict[i][15])
        url_2_url_weight.append(return_dict[i][16])

    user_f_list = torch.cat(user_f_list, dim=0)
    url_f_list = torch.cat(url_f_list, dim=0)
    user_1_user_f_list = torch.cat(user_1_user_f_list, dim=0)
    user_1_user_weight = torch.cat(user_1_user_weight, dim=0)
    user_1_url_f_list = torch.cat(user_1_url_f_list, dim=0)
    user_1_url_weight = torch.cat(user_1_url_weight, dim=0)
    user_2_user_f_list = torch.cat(user_2_user_f_list, dim=0)
    user_2_user_weight = torch.cat(user_2_user_weight, dim=0)
    user_2_url_f_list = torch.cat(user_2_url_f_list, dim=0)
    user_2_url_weight = torch.cat(user_2_url_weight, dim=0)
    url_2_user_f_list = torch.cat(url_2_user_f_list, dim=0)
    url_2_user_weight = torch.cat(url_2_user_weight, dim=0)
    url_2_url_f_list = torch.cat(url_2_url_f_list, dim=0)
    url_2_url_weight = torch.cat(url_2_url_weight, dim=0)
    print('done2', (datetime.now()-start).total_seconds())

    return user_idxs, url_idxs, labels \
        ,user_f_list, url_f_list \
        ,user_1_user_f_list, user_1_user_weight \
        ,user_1_url_f_list, user_1_url_weight \
        ,user_2_user_f_list, user_2_user_weight \
        ,user_2_url_f_list, user_2_url_weight \
        ,url_2_user_f_list, url_2_user_weight \
        ,url_2_url_f_list, url_2_url_weight


#### for main
def build_gcn_and_f_by_sample(
    index_list, samples
    ,user_feature_list, url_feature_list
    ,user_url_weight, user_user_weight, url_url_weight
    ,user_url_dict, url_user_dict, user_neighb_users, url_neighb_urls
    ,uid_no_follower_dict
    ,num_limit_list
    ,mode='train', num=3, return_dict=None, key=0
    ):
    ## url features
    # # [int(url_no), cate2idx[cate], site2idx[url_site], post_user_bin, post_freq_bin, url]
    user_idxs = torch.tensor([samples[index][0] for index in index_list], dtype=torch.long) # (N, 1)
    url_idxs = torch.tensor([samples[index][1] for index in index_list], dtype=torch.long)  # (N, 1)
    labels = torch.tensor([samples[index][2] for index in index_list], dtype=torch.long)    # (N, 1)

    user_f_list, url_f_list = [], []  # (N, f_num)
    user_1_user_f_list, user_1_user_weight = [], []     # (N, num, f_num), (N, num)
    user_1_url_f_list, user_1_url_weight = [], []   # (N, num, f_num), (N, num)
    user_2_user_f_list, user_2_user_weight = [], []     # (N, num, num, f_num), , (N, num, num)
    user_2_url_f_list, user_2_url_weight = [], []   # (N, num, num, f_num), (N, num, num)
    url_2_user_f_list, url_2_user_weight = [], []   # (N, num, num, f_num), (N, num, num)
    url_2_url_f_list, url_2_url_weight = [], []     # (N, num, num, f_num), (N, num, num)

    consume_urls_f_list = []  # (N, item_num, f_num, 1)
    neighb_urls_f_list = []  # (N, item_num, f_num, 1)
    friend_users_f_list = [] # (N, user_item_num, f_num, 1)
    neighb_users_f_list = [] # (N, user_item_num, f_num, 1)

    for uid_no, url_no, label in [samples[index] for index in index_list]:
        # gcn
        user_f = build_user_f_tensor(user_feature_list, uid_no)
        user_f_list.append(user_f)

        url_f = build_url_f_tensor(url_feature_list, url_no)
        url_f_list.append(url_f)

        # user 1 user
        user_1_users = chose_item_by_mode(user_neighb_users[uid_no], num, False, mode)
        # (num, f_num), (num)
        one_graph_fs, one_graph_weights = build_f_and_weight_tensor(uid_no, 'user', user_1_users, 'user', user_feature_list, user_user_weight)
        user_1_user_f_list.append(one_graph_fs)   # (N, num, f_num)
        user_1_user_weight.append(one_graph_weights) # (N, num)

        # user 2 user
        user_2_users_fs, user_2_users_weights = [], []
        # user 2 url
        user_2_urls_fs, user_2_urls_weights = [], []
        for uid in user_1_users:
            user_2_users = chose_item_by_mode(user_neighb_users.get(uid, []), num, False, mode)
            # (num, f_num), (num)
            one_graph_fs, one_graph_weights = build_f_and_weight_tensor(uid, 'user', user_2_users, 'user', user_feature_list, user_user_weight)
            user_2_users_fs.append(one_graph_fs)
            user_2_users_weights.append(one_graph_weights)

            user_2_urls = chose_item_by_mode(user_url_dict.get(uid, []), num, False, mode)
            one_graph_fs, one_graph_weights = build_f_and_weight_tensor(uid, 'user', user_2_urls, 'url', url_feature_list, user_url_weight)
            user_2_urls_fs.append(one_graph_fs)
            user_2_urls_weights.append(one_graph_weights)
        user_2_users_fs = torch.stack(user_2_users_fs, dim=0) # (num, num, f_num)
        user_2_users_weights = torch.stack(user_2_users_weights, dim=0) # (num, num)
        user_2_urls_fs = torch.stack(user_2_urls_fs, dim=0)
        user_2_urls_weights = torch.stack(user_2_urls_weights, dim=0)

        user_2_user_f_list.append(user_2_users_fs)   # (N, num, num, f_num)
        user_2_user_weight.append(user_2_users_weights)  #(N, num, num)
        user_2_url_f_list.append(user_2_urls_fs)
        user_2_url_weight.append(user_2_urls_weights)

        # user 1 url
        user_1_urls = chose_item_by_mode(user_url_dict[uid_no], num, False, mode)
        # (num, f_num), (num)
        one_graph_fs, one_graph_weights = build_f_and_weight_tensor(uid_no, 'user', user_1_urls, 'url', url_feature_list, user_url_weight)
        user_1_url_f_list.append(one_graph_fs)  # (N, num, f_num)
        user_1_url_weight.append(one_graph_weights) # (N, num)

        # url 2 user
        url_2_users_fs, url_2_users_weights = [], []
        # url 2 url
        url_2_urls_fs, url_2_urls_weights = [], []
        for url_id in user_1_urls:
            url_2_users = chose_item_by_mode(url_user_dict.get(url_id, []), num, False, mode)
            one_graph_fs, one_graph_weights = build_f_and_weight_tensor(url_id, 'url', url_2_users, 'user', user_feature_list, user_url_weight)
            url_2_users_fs.append(one_graph_fs) # (num, num, f_num)
            url_2_users_weights.append(one_graph_weights) # (num, num)

            url_2_urls = chose_item_by_mode(url_neighb_urls.get(url_id, []), num, False, mode)
            one_graph_fs, one_graph_weights = build_f_and_weight_tensor(url_id, 'url', url_2_urls, 'url', url_feature_list, url_url_weight)
            url_2_urls_fs.append(one_graph_fs)
            url_2_urls_weights.append(one_graph_weights)
        url_2_users_fs = torch.stack(url_2_users_fs, dim=0) # (num, num, f_num)
        url_2_users_weights = torch.stack(url_2_users_weights, dim=0) # (num, num)
        url_2_urls_fs = torch.stack(url_2_urls_fs, dim=0) # (num, num, f_num)
        url_2_urls_weights = torch.stack(url_2_urls_weights, dim=0) # (num, num)

        url_2_user_f_list.append(url_2_users_fs)   # (N, num, num, f_num)
        url_2_user_weight.append(url_2_users_weights) # (N, num, num)
        url_2_url_f_list.append(url_2_urls_fs)
        url_2_url_weight.append(url_2_urls_weights)

        # feature
        #user_f = build_user_f(user_feature_list, uid_no)
        #user_f_list.append(user_f)

        #url_f = build_url_f(url_feature_list, url_no)
        #url_f_list.append(url_f)

        # url consume urls [[weight, no], [weight, no]]
        url_consume_items = []
        for items in user_url_dict[uid_no]:
            if items[1] != url_no:
                url_consume_items.append(items)
        #print(url_consume_items)
        
        if mode == 'train':
            consume_urls = chose_items_by_random(url_consume_items, num=num_limit_list[0], debug=False)
        else:
            consume_urls = chose_top_items(url_consume_items, num=num_limit_list[0], debug=False)
        consume_urls_f = [] # (item_num, f_num, 1)
        for consume_url_no in consume_urls:
            one_f_list = build_url_f(url_feature_list, consume_url_no)
            consume_urls_f.append(one_f_list)
            #url_idx, cate_idx, site_idx, post_user_bin, post_freq_bin = build_url_f(url_feature_list, consume_url_no)
            #consume_urls_f.append([url_idx, cate_idx, site_idx, post_user_bin, post_freq_bin])
        consume_urls_f_list.append(consume_urls_f)

        # url neighb urls
        url_neighb_items = copy.deepcopy(url_neighb_urls.get(url_no, []))
        if mode == 'train':
            neighb_urls = chose_items_by_random(url_neighb_items, num=num_limit_list[1], debug=False)
        else:
            neighb_urls = chose_top_items(url_neighb_items, num=num_limit_list[1], debug=False)
        neighb_urls_f = [] # (item_num, f_num, 1)
        for neighb_url_no in neighb_urls:
            one_f_list = build_url_f(url_feature_list, neighb_url_no)
            neighb_urls_f.append(one_f_list)
        neighb_urls_f_list.append(neighb_urls_f)

        # user friend-level 
        user_follow_users = copy.deepcopy(uid_no_follower_dict[uid_no])
        if mode == 'train':
            friend_users = chose_items_by_random(user_follow_users, num=num_limit_list[2])
        else:
            friend_users = chose_top_items(user_follow_users, num=num_limit_list[2])
        follow_users_f = [] # (user_num, f_num, 1)
        for friend_no in friend_users:
            one_f_list = build_user_f(user_feature_list, friend_no)
            follow_users_f.append(one_f_list)
        friend_users_f_list.append(follow_users_f)

        # user neighb users
        user_neighb_items = copy.deepcopy(user_neighb_users[uid_no])
        if mode == 'train':
            neighb_users = chose_items_by_random(user_neighb_items, num=num_limit_list[3])
        else:
            neighb_users = chose_top_items(user_neighb_items, num=num_limit_list[3])
        neight_users_f = [] # (user_num, f_num, 1)
        for neighb_no in neighb_users:
            one_f_list = build_user_f(user_feature_list, neighb_no)
            neight_users_f.append(one_f_list)
        neighb_users_f_list.append(neight_users_f)


    user_f_list = torch.stack(user_f_list, dim=0)
    url_f_list = torch.stack(url_f_list, dim=0)

    user_1_user_f_list = torch.stack(user_1_user_f_list, dim=0)     # (N, num, f_num)
    user_1_user_weight = torch.stack(user_1_user_weight, dim=0)     # (N, num)
    user_2_user_f_list = torch.stack(user_2_user_f_list, dim=0)     # (N, num, num, f_num)
    user_2_user_weight = torch.stack(user_2_user_weight, dim=0)     # (N, num, num)
    user_2_url_f_list = torch.stack(user_2_url_f_list, dim=0)       # (N, num, num, f_num)
    user_2_url_weight = torch.stack(user_2_url_weight, dim=0)       # (N, num, num)
    
    user_1_url_f_list = torch.stack(user_1_url_f_list, dim=0)
    user_1_url_weight = torch.stack(user_1_url_weight, dim=0)
    url_2_user_f_list = torch.stack(url_2_user_f_list, dim=0)
    url_2_user_weight = torch.stack(url_2_user_weight, dim=0)
    url_2_url_f_list = torch.stack(url_2_url_f_list, dim=0)
    url_2_url_weight = torch.stack(url_2_url_weight, dim=0)

    consume_urls_f_list = torch.tensor(consume_urls_f_list, dtype=torch.long)
    neighb_urls_f_list = torch.tensor(neighb_urls_f_list, dtype=torch.long)
    friend_users_f_list = torch.tensor(friend_users_f_list, dtype=torch.long)
    neighb_users_f_list = torch.tensor(neighb_users_f_list, dtype=torch.long)

    if return_dict is not None:
        return_dict[key] = (user_idxs, url_idxs, labels \
        ,user_f_list, url_f_list \
        ,user_1_user_f_list, user_1_user_weight \
        ,user_1_url_f_list, user_1_url_weight \
        ,user_2_user_f_list, user_2_user_weight \
        ,user_2_url_f_list, user_2_url_weight \
        ,url_2_url_f_list, url_2_url_weight \
        ,consume_urls_f_list, neighb_urls_f_list\
        ,friend_users_f_list, neighb_users_f_list)
    else:
        return (user_idxs, url_idxs, labels \
        ,user_f_list, url_f_list \
        ,user_1_user_f_list, user_1_user_weight \
        ,user_1_url_f_list, user_1_url_weight \
        ,user_2_user_f_list, user_2_user_weight \
        ,user_2_url_f_list, user_2_url_weight \
        ,url_2_user_f_list, url_2_user_weight \
        ,url_2_url_f_list, url_2_url_weight \
        ,consume_urls_f_list, neighb_urls_f_list\
        ,friend_users_f_list, neighb_users_f_list)
