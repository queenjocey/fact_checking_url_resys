import os
import torch
import math
import random
import numpy as np


def init_embedding(input_embedding):
    """
    Initialize embedding
    """
    bias = np.sqrt(3.0 / input_embedding.weight.size(1))
    torch.nn.init.uniform_(input_embedding.weight, -bias, bias)


def init_normal(input_embedding):
    torch.nn.init.normal_(input_embedding.weight, mean=0, std=0.1)


def init_linear(input_linear):
    """
    Initialize linear transformation
    """
    bias = np.sqrt(6.0 / (input_linear.weight.size(0) + input_linear.weight.size(1)))
    torch.nn.init.uniform_(input_linear.weight, -bias, bias)
    if input_linear.bias is not None:
        input_linear.bias.data.zero_()


def get_metric(res, num=100):
    count = 0
    one_res = []
    hit_top_10 = 0.0
    total_ndcg = 0.0
    total_diff_cnt = 0.0
    total_samples = len(res) / num
    for value in res:
        count += 1
        one_res.append(value)
        if count == num:
            #print('start')
            pos_value = one_res[0]
            one_res.sort(reverse=True)
            pos = num - 1
            for i in range(num):
                # chose last pos
                if one_res[i] < pos_value:
                    pos = i - 1
                    break
            if pos < 10:
                hit_top_10 += 1
            ndcg = math.log(2) / math.log(pos+2) if pos < 10 else 0
            total_ndcg += ndcg
            total_diff_cnt += len(set(one_res))
            one_res = []
            count = 0
    return hit_top_10 / total_samples, total_ndcg / total_samples, total_diff_cnt / total_samples


def get_metric2(res, num=100):
    count = 0
    one_res = []
    hit_top_list = [0.0, 0.0, 0.0, 0.0]
    total_ndcg = [0.0, 0.0, 0.0, 0.0]
    total_diff_cnt = 0.0
    total_samples = len(res) / num
    for value in res:
        count += 1
        one_res.append(value)
        if count == num:
            #print('start')
            pos_value = one_res[0]
            one_res.sort(reverse=True)
            pos = num - 1
            for i in range(num):
                # chose last pos
                if one_res[i] < pos_value:
                    pos = i - 1
                    break

            index = 0
            for top in [1, 3, 5, 10]:
                if pos < top:
                    hit_top_list[index] += 1
                ndcg = math.log(2) / math.log(pos+2) if pos < top else 0
                total_ndcg[index] += ndcg
                index += 1
            total_diff_cnt += len(set(one_res))
            one_res = []
            count = 0
    for i in range(len(total_ndcg)):
        hit_top_list[i] = hit_top_list[i] / total_samples
        total_ndcg[i] = total_ndcg[i] / total_samples
    return hit_top_list, total_ndcg, total_diff_cnt / total_samples





def get_inverse_idx(orgin_idx_tensor, data_type='url', dim_type='char'):
    new_lists = list(orgin_idx_tensor.data.numpy())
    # 3d
    if data_type == 'url' and dim_type == 'char':
        inverse_idx = [ [list2[::-1] for list2 in list1]
            for list1 in new_lists]
    # 2d
    elif data_type == 'url' and dim_type == 'word':
        inverse_idx = [ list1[::-1] for list1 in new_lists]
    # 4d
    elif data_type == 'tweets' and dim_type == 'char':
        inverse_idx = [ [ [list3[::-1] for list3 in list2] for list2 in list1]
            for list1 in new_lists]
    # 3d
    else:
        inverse_idx = [ [list2[::-1] for list2 in list1]
            for list1 in new_lists]
    return torch.tensor(inverse_idx, dtype=torch.long)


def generate_batch_index(batch_num, total_num=0, index_list=None):
    index_lists = []
    batch_cnt = 0
    one_index_list = []
    if index_list is None:
        index_list = range(total_num)
    for i in index_list:
        one_index_list.append(i)
        batch_cnt += 1
        if batch_cnt == batch_num:
            index_lists.append(one_index_list)
            batch_cnt = 0
            one_index_list = []
    if len(one_index_list) > 0:
        index_lists.append(one_index_list)
    return index_lists


def build_train_sample_random(train_file, neg_num=5, total_url_cnt=4732):
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
    #with open(target_dir + os.sep + 'train_dl_random.txt', 'w') as f:
    #    f.write(json.dumps(all_samples))
    print('random %s samples: %s' % (neg_num, len(all_samples)))
    return all_samples
