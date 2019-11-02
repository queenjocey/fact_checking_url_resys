import os
import torch
import json
import math
import numpy as np

from datetime import datetime
from multiprocessing import Process
import multiprocessing


def read_dict_json(json_dict_file, name='json_dict'):
    json_dict = json.load(open(json_dict_file, 'r'))
    json_dict = {int(key): [int(item) for item in item_list] for key, item_list in json_dict.items()}
    #print(name, len(json_dict), list(json_dict.keys())[0:3])
    return json_dict


def gen_relation_list(user_num, url_num):
    train_pos_file = '/home/dyou/url_recsys/data/train_dl.txt'
    user_url_matrix_list = [[0.0 for j in range(url_num)] for i in range(user_num)]
    url_user_matrix_list = [[0.0 for j in range(user_num)] for i in range(url_num)]
    user_url_freq_list = [[0.0 for j in range(url_num)] for i in range(user_num)]
    user_post_url_list = [0 for i in range(user_num)]
    url_posted_user_list = [0 for i in range(url_num)]

    with open(train_pos_file) as f:
        lines = f.readlines()
    print('all pos sample:', len(all_pos_sample), len(lines))
    max_uid, max_url = -1, -1
    for line in lines:
        uid, url_id, freq, ts = [int(item) for item in line.strip().split()]
        assert str(uid) + '_' + str(url_id) in all_pos_sample
        max_uid = max(max_uid, uid)
        max_url = max(max_url, url_id)
        user_url_matrix_list[uid][url_id] = 1.0
        url_user_matrix_list[url_id][uid] = 1.0
        user_url_freq_list[uid][url_id] = 1.0 * freq
        user_post_url_list[uid] += 1.0
        url_posted_user_list[url_id] += 1.0
    print('user num:', max_uid+1, 'url num:', max_url+1)

    #json.dump(cate_info, open(out_dir + os.sep + 'cate_info.json', 'w'))
    json.dump(user_url_matrix_list, open(target_dir + os.sep + 'user_url_matrix_list.json', 'w'))
    json.dump(url_user_matrix_list, open(target_dir + os.sep + 'url_user_matrix_list.json', 'w'))
    json.dump(user_url_freq_list, open(target_dir + os.sep + 'user_url_freq_list.json', 'w'))
    json.dump(user_post_url_list, open(target_dir + os.sep + 'user_post_url_list.json', 'w'))
    json.dump(url_posted_user_list, open(target_dir + os.sep + 'url_posted_user_list.json', 'w'))
    print('gen relation list success!')


def gen_user_url_weight(user_url_freq_list, user_max_post_freq_list, url_num=4732):
    # user url weight: (freq / user max freq) * log(user_num / url posted user num)
    print('start to gen user url weight')
    user_num = len(user_url_freq_list)
    user_url_weight = [[0 for j in range(url_num)] for i in range(user_num)]
    weights, count = 0.0, 0
    for uid in range(user_num):
        for url_id in range(url_num):
            if user_url_freq_list[uid][url_id] != 0:
                user_url_weight[uid][url_id] = (user_url_freq_list[uid][url_id] / user_max_post_freq_list[uid]) / math.log(1.0*user_num / url_posted_user_list[url_id])
                weights += user_url_weight[uid][url_id]
                count += 1
    print('user url weight:', count, weights/count, 'user num:', user_num)
    json.dump(user_url_weight, open(target_dir + os.sep + 'user_url_weight.json', 'w'))


# post_list fenmu
def gen_one_weight(i_list, num, out_weight, post_list, matrix_2d_list, D=1.0):
    count, weights = 0, 0.0
    print(datetime.now(), i_list[0], 'start')
    for i in i_list:
        for j in range(num):
            if i == j or post_list[i] == 0 or post_list[j] == 0:
                out_weight[i][j] = 0
            else:
                ij_cnt = np.sum([1 for item1, item2 in zip(matrix_2d_list[i], matrix_2d_list[j]) if item1+item2 == 2])
                pmi_fix = 1.0 * ij_cnt * D / (post_list[i] * post_list[j])
                out_weight[i][j] = pmi_fix
                count += 1
                weights += pmi_fix
                # max(0, pmi - log(s)), pmi = log(ij_cnt * D / i_cnt * j_cnt)
        #if count % 50 == 0:
        #    print(datetime.now(), i_list[0], 50, 'done')
        #count += 1
    print(datetime.now(), i_list[0], 'done', count, weights) 


def gen_weight(num, post_list, matrix_2d_list, out_name):
    print(out_name, num)
    out_weight = [[0 for j in range(num)] for i in range(num)]
    #pool = multiprocessing.Pool(processes=8)
    one_num = int(num / 10)
    i_lists = []
    i_list = []
    for i in range(num):
        i_list.append(i)
        if len(i_list) >= one_num:
            i_lists.append(i_list)
            i_list = []
    if len(i_list) > 0:
        i_lists.append(i_list)

    p_list = []
    for i_list in i_lists:
        p = Process(target=gen_one_weight, args=(i_list, num, out_weight, post_list, matrix_2d_list))
        p.start()
        p_list.append(p)
        #pool.apply_async(gen_one_weight, (i, num, out_weight, post_list, matrix_2d_list))
    #pool.close()
    #pool.join()
    for p in p_list:
        p.join()
    total_weight = np.sum(out_weight)
    if total_weight == 0:
        print('error')
    json.dump(out_weight, open(target_dir + os.sep + out_name, 'w'))
    print(out_name, 'pmi fix done')
    return out_weight

# 
def gen_weight_by_gpu(num, post_list, matrix_2d_list, out_name, device_id=0):
    print(out_name, num)
    #ij_cnt = np.sum([1 for item1, item2 in zip(matrix_2d_list[i], matrix_2d_list[j]) if item1+item2 == 2])
    #            pmi_fix = 1.0 * ij_cnt * D / (post_list[i] * post_list[j])
    post_list = torch.tensor(post_list, dtype=torch.float)
    matrix_2d_list = torch.tensor(matrix_2d_list, dtype=torch.float)
    post_list = post_list.cuda(device_id)
    matrix_2d_list = matrix_2d_list.cuda(device_id)

    ij_cnt = torch.mm(matrix_2d_list, torch.transpose(matrix_2d_list, 0, 1))
    ij_post = post_list.view(-1, 1) * post_list.view(1, -1)
    ij_post[ij_post == 0] = 1
    out_weight = ij_cnt / ij_post

    weight_cnt = torch.sum(out_weight > 0)
    weight_sum = torch.sum(out_weight)
    shape = out_weight.shape

    out_weight = out_weight.cpu().data.numpy().tolist()
    json.dump(out_weight, open(target_dir + os.sep + out_name, 'w'))
    print(out_name, shape, weight_cnt, weight_sum, weight_sum / weight_cnt, 'pmi fix done')
    return out_weight


def check_data1(user_url_matrix_list, url_user_matrix_list):
    ### check data
    ## read user url relation
    user_url_dict_file = '/home/dyou/url_recsys/dl_data/feature/user_url_dict.json'
    user_url_dict = read_dict_json(user_url_dict_file, 'user_url')
    for uid in user_url_dict:
        for url_id in user_url_dict[uid]:
            assert user_url_matrix_list[uid][url_id] == 1
            assert url_user_matrix_list[url_id][uid] == 1
    print('check user url relation success, user url dict:', len(user_url_dict))

    url_user_dict_file = '/home/dyou/url_recsys/dl_data/feature/url_user_dict.json'
    url_user_dict = read_dict_json(url_user_dict_file, 'url_user')
    for url_id in url_user_dict:
        for uid in url_user_dict[url_id]:
            assert user_url_matrix_list[uid][url_id] == 1
            assert url_user_matrix_list[url_id][uid] == 1
    print('check url user relation success, url user dict:', len(url_user_dict))


def check_data2(user_user_weight, url_url_weight):
    ## read url neighb urls
    url_neighb_urls_dict_file = '/home/dyou/url_recsys/dl_data/feature/url_neighb_urls.json'
    url_neighb_urls = read_dict_json(url_neighb_urls_dict_file, 'url_neighb_urls')
    for url_id1 in url_neighb_urls:
        for url_id2 in url_neighb_urls[url_id1]:
            assert url_url_weight[url_id1][url_id2] > 0
    print('check url url weight success')

    ## read user url neighb info
    user_neighb_users_dict_file = '/home/dyou/url_recsys/dl_data/feature/user_neighb_users.json'
    user_neighb_users = read_dict_json(user_neighb_users_dict_file, 'user_neighb')
    for uid1 in user_neighb_users:
        for uid2 in user_neighb_users[uid1]:
            assert user_user_weight[uid1][uid2] > 0
    print('check user user weight success')


if __name__ == '__main__':
    target_dir = '/home/dyou/url_recsys/dl_data/gcn'
    '''train_json_file = '/home/dyou/url_recsys/dl_data/sample/train_dl_random.txt'
    with open(train_json_file) as f:
        all_samples = json.loads(f.read())
    all_pos_sample = [str(item[0]) + '_' + str(item[1]) for item in all_samples if item[2] == 1]

    gen_relation_list(user_num=11576, url_num=4732)'''
    user_url_matrix_list = json.load(open(target_dir + os.sep + 'user_url_matrix_list.json'))
    url_user_matrix_list = json.load(open(target_dir + os.sep + 'url_user_matrix_list.json'))
    user_url_freq_list = json.load(open(target_dir + os.sep + 'user_url_freq_list.json'))
    user_post_url_list = json.load(open(target_dir + os.sep + 'user_post_url_list.json'))
    url_posted_user_list = json.load(open(target_dir + os.sep + 'url_posted_user_list.json'))
    user_max_post_freq_list = [max(user_url_freq_list[uid]) for uid in range(11576)]
    print('user_url_matrix_list:', np.sum(user_url_matrix_list)) 
    print('url_user_matrix_list:', np.sum(url_user_matrix_list)) 
    print('user_url_freq_list:', np.sum(user_url_freq_list)) 
    print('user_post_url_list:', np.sum(user_post_url_list)) 
    print('url_posted_user_list:', np.sum(url_posted_user_list)) 

    check_data1(user_url_matrix_list, url_user_matrix_list)

    #gen_user_url_weight(user_url_freq_list, user_max_post_freq_list)

    #url_url_weight = gen_weight(4732, url_posted_user_list, url_user_matrix_list, 'url_url_weight.json')
    #user_user_weight = gen_weight(11576, user_post_url_list, user_url_matrix_list, 'user_user_weight.json')

    url_url_weight = gen_weight_by_gpu(4732, url_posted_user_list, url_user_matrix_list, 'url_url_weight.json', 1)
    user_user_weight = gen_weight_by_gpu(11576, user_post_url_list, user_url_matrix_list, 'user_user_weight.json', 1)

    user_user_weight = json.load(open(target_dir + os.sep + 'user_user_weight.json'))
    url_url_weight = json.load(open(target_dir + os.sep + 'url_url_weight.json'))
    check_data2(user_user_weight, url_url_weight)
