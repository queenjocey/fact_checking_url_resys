import os
import copy
import json
import math
import random
from collections import Counter
import numpy as np

# 废弃
def get_url_site_and_cate_statis(url_ori_feature_file):
    with open(url_ori_feature_file) as f:
        # url_no,url,url_site,cate,post_users,post_freq
        lines = f.readlines()

    site_set = set()
    cate_set = set()
    cate_list = []
    for line in lines[1:]:
        url_no,url,url_site,cate,post_users,post_freq = [item.strip() for item in line.strip().split(',')]
        site_set.add(url_site)
        cate_set.add(cate)
        cate_list.append(cate)
    print(len(site_set))
    print(len(cate_set))
    print(Counter(cate_list))


def get_value_bin(value, value_bins = [2, 4, 8, 16, 32, 64, 128, 256, 512, 1024]):
    value_index = len(value_bins) - 1
    for index in range(len(value_bins)):
        if int(value) <= value_bins[index]:
            value_index = index
            break
    return value_index


def process_url_feature(url_ori_feature_file, out_dir):
    if not os.path.isdir(out_dir):
        os.system('mkdir -p %s' % (out_dir))

    with open(url_ori_feature_file) as f:
        # url_no,url,url_site,cate,post_users,post_freq
        lines = f.readlines()
    
    cate_lists = [
        ['hurricane', 'environment'], ['soapbox', 'junknews', 'horror', 'food'],    
        ['entertainment', 'crittercountry'], ['history', 'education'],    
        ['immigration', 'terrorism'], ['medical', 'science'], ['racial', 'religion'], ['crime', 'legal']
    ]

    cate_info = {}
    site_info = {}
    value_bins = [2, 4, 8, 16, 32, 64, 128, 256, 512, 1024]
    for line in lines[1:]:
        url_no,url,url_site,cate,post_users,post_freq = [item.strip() for item in line.strip().split(',')]
        site_info[url_site] = site_info.get(url_site, 0) + 1
        cate_key = None
        for cate_list in cate_lists:
            if cate in cate_list:
                cate_key = '/'.join(cate_list)
                break
        cate_key = cate if cate_key is None else cate_key
        cate_info[cate_key] = cate_info.get(cate_key, 0) + 1
    print('cate num: %d, site num: %d' % (len(cate_info), len(site_info)))

    site2idx = {site: idx for (idx, site) in enumerate(site_info.keys(), 0)}
    cate2idx = {}
    cate_idx = 0
    for key in cate_info.keys():
        for k in key.split('/'):
            cate2idx[k] = cate_idx
        cate_idx += 1

    url_f_list = []
    post_user_info = {}
    post_freq_info = {}
    # Item level:ID, category, websites, # of posted
    for line in lines[1:]:
        url_no,url,url_site,cate,post_users,post_freq = [item.strip() for item in line.strip().split(',')]
        post_user_bin, post_freq_bin = get_value_bin(post_users), get_value_bin(post_freq)
        post_user_info[post_user_bin] = post_user_info.get(post_user_bin, 0) + 1
        post_freq_info[post_freq_bin] = post_freq_info.get(post_freq_bin, 0) + 1
        url_f = [int(url_no), cate2idx[cate], site2idx[url_site], post_user_bin, post_freq_bin, url]
        url_f_list.append(url_f)

    url_f_list.sort(key=lambda x: x[0], reverse=False)
    json.dump(cate_info, open(out_dir + os.sep + 'cate_info.json', 'w'))
    json.dump(site_info, open(out_dir + os.sep + 'site_info.json', 'w'))
    json.dump(post_user_info, open(out_dir + os.sep + 'post_user_info.json', 'w'))
    json.dump(post_freq_info, open(out_dir + os.sep + 'post_freq_info.json', 'w'))
    json.dump(cate2idx, open(out_dir + os.sep + 'cate2idx.json', 'w'))
    json.dump(site2idx, open(out_dir + os.sep + 'site2idx.json', 'w'))
    json.dump(url_f_list, open(out_dir + os.sep + 'url_feature_list.json', 'w'), indent=True)
    print('url feature done')


def process_user_feature(train_file, user_ori_feature_file, url_feature_json_file, out_dir):
    # [int(url_no), cate2idx[cate], site2idx[url_site], post_user_bin, post_freq_bin, url]
    url_no2cate_dict, url_no2site_dict = {}, {}
    with open(url_feature_file) as f:
        url_feature_list = json.loads(f.read())
    assert len(url_feature_list) == 4732
    for item in url_feature_list:
        url_no2cate_dict[item[0]] = item[1]
        url_no2site_dict[item[0]] = item[2]

    user_cate_dict, user_site_dict = {}, {}
    with open(train_file) as f:
        lines = f.readlines()
    for line in lines:
        uid, url_no, freq, ts = line.strip().split()
        if int(uid) not in user_cate_dict:
            user_cate_dict[int(uid)] = []
        if int(uid) not in user_site_dict:
            user_site_dict[int(uid)] = []

        user_cate_dict[int(uid)].append(url_no2cate_dict[int(url_no)])
        user_site_dict[int(uid)].append(url_no2site_dict[int(url_no)])

    with open(user_ori_feature_file) as f:
        lines = f.readlines()

    user_feature_list = []
    user_post_urls = {}
    user_post_freq = {}
    user_favorite_cate_no = {}
    user_favorite_site_no = {}
    follow_info = {}
    follow_in_info = {}
    for line in lines[1:]:
        '''uid_no,uid,name,post_urls,post_freq,follow_cnt,follow_in_cnt
        0,767036004,upayr,68,237,274,324'''
        uid_no,uid,name,post_urls,post_freq,follow_cnt,follow_in_cnt = [item.strip() for item in line.split(',')]

        #User level:ID, # of followers, # of following, posts urls, post cnt, favorite category, favorite website
        value_bins = [2, 4, 8, 16, 32, 64, 128, 256, 512, 1024]
        follow_cnt_bin = get_value_bin(value=follow_cnt)
        follow_info[follow_cnt_bin] = follow_info.get(follow_cnt_bin, 0) + 1

        follow_in_cnt_bin = get_value_bin(value=follow_in_cnt)
        follow_in_info[follow_in_cnt_bin] = follow_in_info.get(follow_in_cnt_bin, 0) + 1

        post_cnt_bin = get_value_bin(value=post_freq)
        user_post_freq[post_cnt_bin] = user_post_freq.get(post_cnt_bin, 0) + 1
        
        post_url_bin = get_value_bin(value=post_urls)
        user_post_urls[post_url_bin] = user_post_urls.get(post_url_bin, 0) + 1
        
        c = Counter(user_cate_dict[int(uid_no)])
        favorite_cate_no = c.most_common(1)[0][0]
        user_favorite_cate_no[favorite_cate_no] = user_favorite_cate_no.get(favorite_cate_no, 0) + 1

        c = Counter(user_site_dict[int(uid_no)])
        favorite_site_no = c.most_common(1)[0][0]
        user_favorite_site_no[favorite_site_no] = user_favorite_site_no.get(favorite_site_no, 0) + 1

        out_f = [int(uid_no), follow_cnt_bin, follow_in_cnt_bin, post_url_bin, post_cnt_bin
                ,favorite_cate_no, favorite_site_no, uid, name]
        user_feature_list.append(out_f)
    assert len(user_feature_list) == 11576

    user_feature_list.sort(key=lambda x: x[0], reverse=False)
    json.dump(user_feature_list, open(out_dir + os.sep + 'user_feature_list.json', 'w'), indent=True)

    json.dump(user_post_urls, open(out_dir + os.sep + 'user_post_urls.json', 'w'))
    json.dump(user_post_freq, open(out_dir + os.sep + 'user_post_freq.json', 'w'))
    json.dump(user_favorite_cate_no, open(out_dir + os.sep + 'user_favorite_cate_no.json', 'w'))
    json.dump(user_favorite_site_no, open(out_dir + os.sep + 'user_favorite_site_no.json', 'w'))
    json.dump(follow_info, open(out_dir + os.sep + 'follow_info.json', 'w'))
    json.dump(follow_in_info, open(out_dir + os.sep + 'follow_in_info.json', 'w'))

    print('user feature done')


def build_new_cnn_relation(train_file, user_feature_file, user_follow_file, out_dir):
    user_url_dict = {}
    url_user_dict = {}
    with open(train_file) as f:
        lines = f.readlines()
    for line in lines:
        uid_no, url_no, freq, ts = line.strip().split()
        if int(uid_no) not in user_url_dict:
            user_url_dict[int(uid_no)] = []
        user_url_dict[int(uid_no)].append(int(url_no))

        if int(url_no) not in url_user_dict:
            url_user_dict[int(url_no)] = []
        url_user_dict[int(url_no)].append(int(uid_no))
    print('user url dict: %d, url user dict: %d' % (len(user_url_dict), len(url_user_dict)))

    user2idx_dict = {}
    with open(user_feature_file) as f:
        user_feature_list = json.loads(f.read())
        assert len(user_feature_list) == 11576
    ##User level: ID, # of followers, # of following, posts urls, post cnt, favorite category, favorite website, uid, name
    for item in user_feature_list:
        # item[7] is string
        user2idx_dict[item[7]] = int(item[0])

    uid_mapped_file = '/home/dyou/url_recsys/data_gen/data_181110/uid_mapped.txt'
    with open(uid_mapped_file) as f:
        lines = f.readlines()
    for line in lines:
        #0   767036004   upayr
        info = [item.strip() for item in line.strip().split()]
        assert len(info) == 3
        uid_no, uid = info[0], info[1]
        assert int(uid_no) == user2idx_dict[uid]
    print('check uid mapped success')

    with open(user_follow_file) as f:
        lines = f.readlines()
    ori_follower_dict = {}
    ori_follow_in_dict = {}
    for line in lines[1:]:
        #follower,followee    followee = follow in
        #757778071734804480,815708250217840640
        follower, follow_in = line.strip().split(',')
        if follower not in ori_follower_dict:
            ori_follower_dict[follower] = []
        if follow_in not in ori_follower_dict[follower]:
            ori_follower_dict[follower].append(follow_in)

        if follow_in not in ori_follow_in_dict:
            ori_follow_in_dict[follow_in] = []
        if follower not in ori_follow_in_dict[follow_in]:
            ori_follow_in_dict[follow_in].append(follower)

    miss_follower, miss_follow_in = 0, 0
    uid_no_follower_dict, uid_no_follow_in_dict = {}, {}
    for uid in user2idx_dict:
        if uid not in ori_follower_dict:
            miss_follower += 1
            uid_no_follower_dict[int(user2idx_dict[uid])] = []
        else:
            res_list = []
            for uid2 in ori_follower_dict[uid]:
                if uid2 in user2idx_dict:
                    res_list.append(int(user2idx_dict[uid2]))
            if len(res_list) == 0:
                miss_follower += 1
                uid_no_follower_dict[int(user2idx_dict[uid])] = []
            else:
                uid_no_follower_dict[int(user2idx_dict[uid])] = res_list

        if uid not in ori_follow_in_dict:
            miss_follow_in += 1
            uid_no_follow_in_dict[int(user2idx_dict[uid])] = []
        else:
            res_list = []
            for uid2 in ori_follow_in_dict[uid]:
                if uid2 in user2idx_dict:
                    res_list.append(int(user2idx_dict[uid2]))
            if len(res_list) == 0:
                miss_follow_in += 1
                uid_no_follow_in_dict[int(user2idx_dict[uid])] = []
            else:
                uid_no_follow_in_dict[int(user2idx_dict[uid])] = res_list

    print('follow users: %d, miss follower: %d, follow in users: %d, miss follow in: %d' 
        % (len(uid_no_follower_dict), miss_follower, len(uid_no_follow_in_dict), miss_follow_in))

    url_neighb_urls = {}
    for url_no in url_user_dict:
        direct_user_list = url_user_dict[url_no]
        two_step_url_list = []
        for user_no in direct_user_list:
            two_step_url_list.extend(user_url_dict[user_no])
        two_step_url_set = set(two_step_url_list)
        two_step_url_set.remove(url_no)
        url_neighb_urls[url_no] = list(two_step_url_set)

    print('url_neighb_urls:', len(url_neighb_urls))

    user_neighb_users = {}
    for uid_no in user_url_dict:
        direct_url_list = user_url_dict[uid_no]
        two_step_user_list = []
        for url_no in direct_url_list:
            two_step_user_list.extend(url_user_dict[url_no])
        two_step_user_set = set(two_step_user_list)
        two_step_user_set.remove(uid_no)
        user_neighb_users[uid_no] = list(two_step_user_set)
    print('user_neighb_users:', len(user_neighb_users))
    
    json.dump(user2idx_dict, open(out_dir + os.sep + 'user2idx.json', 'w'))
    json.dump(ori_follower_dict, open(out_dir + os.sep + 'ori_follower_dict.json', 'w'))
    json.dump(ori_follow_in_dict, open(out_dir + os.sep + 'ori_follow_in_dict.json', 'w'))
    json.dump(user_url_dict, open(out_dir + os.sep + 'user_url_dict.json', 'w'))
    json.dump(url_user_dict, open(out_dir + os.sep + 'url_user_dict.json', 'w'))
    json.dump(url_neighb_urls, open(out_dir + os.sep + 'url_neighb_urls.json', 'w'))
    json.dump(user_neighb_users, open(out_dir + os.sep + 'user_neighb_users.json', 'w'))
    json.dump(uid_no_follower_dict, open(out_dir + os.sep + 'uid_no_follower_dict.json', 'w'))
    json.dump(uid_no_follow_in_dict, open(out_dir + os.sep + 'uid_no_follow_in_dict.json', 'w'))
    print('new cnn relation done')


def build_train_sample_random(train_file, target_dir, neg_num=5, total_url_cnt=4732):
    with open(train_file) as f:
        lines = f.readlines()
    # uid, url_id, freq, ts
    uid_dict = {}
    all_samples = []
    for line in lines:
        uid, url_id, freq, ts = line.strip().split()
        if int(uid) not in uid_dict:
            uid_dict[int(uid)] = []
        uid_dict[int(uid)].append(int(url_id))
        all_samples.append([int(uid), int(url_id), 1])
    for uid in uid_dict.keys():
        cur_url = set(uid_dict[uid])
        neg_urls = []
        neg_cnt = 0
        while neg_cnt < len(cur_url) * neg_num:
            url_index = random.randint(0, total_url_cnt-1)
            if url_index in cur_url or url_index in neg_urls:
                continue
            else:
                all_samples.append([int(uid), int(url_index), 0])
                neg_urls.append(int(url_index))
                neg_cnt += 1

    #all_samples.sort(key=lambda x: x[0], reverse=False)
    with open(target_dir + os.sep + 'train_dl_random.txt', 'w') as f:
        f.write(json.dumps(all_samples))
    return all_samples


def build_train_sample_cate_random(train_file, url_feature_file, target_dir, neg_num=5):
    # [int(url_no), cate2idx[cate], site2idx[url_site], post_user_bin, post_freq_bin, url]
    url_cate_dict = {} # key cate, value urls
    url2cate_dict = {}
    url_weight_dict = {}
    with open(url_feature_file) as f:
        url_feature_list = json.loads(f.read())
    assert len(url_feature_list) == 4732

    for item in url_feature_list:
        if item[1] not in url_cate_dict:
            url_cate_dict[item[1]] = [item[0]]
        else:
            url_cate_dict[item[1]].append(item[0])

        url2cate_dict[item[0]] = item[1]
        url_weight_dict[item[0]] = math.pow(2, item[3])

    # read pos train case
    with open(train_file) as f:
        lines = f.readlines()
    uid_dict = {}
    all_samples = []
    for line in lines:
        uid, url_no, freq, ts = line.strip().split()
        if int(uid) not in uid_dict:
            uid_dict[int(uid)] = []
        uid_dict[int(uid)].append(int(url_no))
        #all_samples.append([int(uid), int(url_no), 1])

    for uid in uid_dict.keys():
        #print (uid)
        pos_url_nos = uid_dict[uid]
        for pos_url_no in pos_url_nos:
            url_cate_no = url2cate_dict[pos_url_no]
            neg_url_nos = choose_neg_url_nos(url_cate_dict, url_cate_no, url_weight_dict, pos_url_nos, neg_num=5, total_url_cnt=4732)
            all_samples.append([int(uid), int(pos_url_no), 1])
            for url_no in neg_url_nos:        
                all_samples.append([int(uid), int(url_no), 0])

    #all_samples.sort(key=lambda x: x[0], reverse=False)
    with open(target_dir + os.sep + 'train_dl_cate_random.txt', 'w') as f:
        f.write(json.dumps(all_samples))
    return all_samples


def choose_neg_url_nos(url_cate_dict, url_cate_no, url_weight_dict, pos_url_nos, neg_num=5, total_url_cnt=4732):
    neg_url_nos = []
    url_cate_list = [url_no for url_no in url_cate_dict[url_cate_no] if url_no not in pos_url_nos]
    if len(url_cate_list) <= neg_num:
        print('little cate:', url_cate_no)
        neg_url_nos.extend(url_cate_list)
        while len(neg_url_nos) < neg_num:
            url_no = random.randint(0, total_url_cnt-1)
            if url_no in neg_url_nos or url_no in pos_url_nos:
                continue
            else:
                neg_url_nos.append(url_no)
    else:
        cate_urls = copy.deepcopy(url_cate_list)
        #cate_urls.sort()
        while len(neg_url_nos) < neg_num:
            all_weight = np.sum([url_weight_dict[url_no] for url_no in cate_urls])
            random_weight = random.randint(1, all_weight)
            cumsum_weight = 0
            for index in range(len(cate_urls)):
                cumsum_weight += url_weight_dict[cate_urls[index]]
                if random_weight <= cumsum_weight:
                    break
            url_no = cate_urls[index]

            if url_no in neg_url_nos or url_no in pos_url_nos:
                continue
            else:
                neg_url_nos.append(url_no)
                cate_urls.remove(url_no)
    return neg_url_nos


if __name__ == '__main__':
    train_pos_file = '/home/dyou/url_recsys/data/train_dl.txt'
    #test_file = '/home/dyou/url_recsys/data/train_dl.txt'
    url_ori_feature_file = '/home/dyou/url_recsys/data/url_mapped_feature_sign.csv'
    user_ori_feature_file = '/home/dyou/url_recsys/data/uid_mapped_feature.csv'
    target_dir = '/home/dyou/url_recsys/dl_data/sample'
    #print (get_value_bin(value=1025, value_bins=[2, 4, 8, 16, 32, 64, 128, 256, 512, 1024]))
    process_url_feature(url_ori_feature_file, '/home/dyou/url_recsys/dl_data/feature')
    url_feature_file = '/home/dyou/url_recsys/dl_data/feature/url_feature_list.json'

    process_user_feature(train_pos_file, user_ori_feature_file, url_feature_file, '/home/dyou/url_recsys/dl_data/feature')

    user_feature_file = '/home/dyou/url_recsys/dl_data/feature/user_feature_list.json'
    user_follow_file = '/home/dyou/url_recsys/data_gen/data_181110/unique_directed_edges.csv'
    build_new_cnn_relation(train_pos_file, user_feature_file, user_follow_file, out_dir='/home/dyou/url_recsys/dl_data/feature')


    '''all_samples = build_train_sample_random(train_pos_file, target_dir, neg_num=5, total_url_cnt=4732)
    print('random sample done:', len(all_samples))

    all_samples = build_train_sample_cate_random(train_pos_file, url_feature_file, target_dir, neg_num=5)
    print('cate random sample done:', len(all_samples))'''
