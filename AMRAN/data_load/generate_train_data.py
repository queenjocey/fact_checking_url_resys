import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import math

import os
import sys
import random
import json
import numpy as np
from datetime import datetime
from multiprocessing import Process
import multiprocessing


def build_train_data(neg_num=5, total_url_cnt=4732):
    '''with open('/home/dyou/url_recsys/data/train_dl.txt') as f:
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
        neg_cnt = 0
        while neg_cnt < len(cur_url) * neg_num:
            url_index = random.randint(0, total_url_cnt-1)
            if url_index in cur_url:
                continue
            else:
                all_samples.append([int(uid), int(url_index), 0])
                neg_cnt += 1

    with open('/home/dyou/url_recsys/dl_data/train_dl.txt', 'w') as f:
        f.write(json.dumps(all_samples))'''
    sample_file = '/home/dyou/url_recsys/dl_data/sample/train_dl_random.txt'
    #'/home/dyou/url_recsys/dl_data/train_dl.txt'
    with open(sample_file) as f:
        all_samples = json.loads(f.read())
    return all_samples


def build_test_data():
    with open('/home/dyou/url_recsys/data/test.txt') as f:
        lines = f.readlines()
    # uid, url_id, freq, ts
    all_test_samples = []
    uid_dict = {}
    for line in lines:
        uid, url_id, freq, ts = line.strip().split()
        if int(uid) not in uid_dict:
            uid_dict[int(uid)] = []
        all_samples.append([int(uid), int(url_id), 0 if int(freq) == 0 else 1])

    return all_samples


def get_word_char_ids(context, word_limit, char_limit):
    #{'word': [], 'char': [[], []]}
    word_ids = [] # shape 1 word_limit
    char_ids = [] # shape 2 (word_limit, char_limit)
    cur_word_ids = context['word']
    cur_word_cnt = len(context['word'])
    word_cnt = 0
    while word_cnt < word_limit:
        if word_cnt < cur_word_cnt:
            word_ids.append(int(cur_word_ids[word_cnt]))
            one_word_char_ids = [int(_id) for _id in context['char'][word_cnt][:char_limit]]
            one_word_char_len = len(one_word_char_ids)
            while one_word_char_len < char_limit:
                one_word_char_ids.append(0)
                one_word_char_len += 1
            char_ids.append(one_word_char_ids)
        else: 
            word_ids.append(0)
            one_word_char_ids = [0 for index in range(char_limit)]
            char_ids.append(one_word_char_ids)
        word_cnt += 1

    return word_ids, char_ids


def get_tweets_word_char_ids(user_id, url_id, tweet_limit=50, word_limit=30, char_limit=16):
    tfidf_dir = '/home/dyou/url_recsys/dl_data/tfidf200'
    tweet_dir = '/home/dyou/url_recsys/dl_data/encode_tweets'
    tfidf_file = tfidf_dir + os.sep + '{}_tfidf_top_200.txt'.format(url_id)
    with open(tfidf_file, 'r') as f:
        url_dict = json.loads(f.read())
    # read one user top 200 simility tweet ids
    tweet_ids = url_dict[str(user_id)] # [0, 323, 44, 556, ...]
    # read one user all tweets
    tweet_file = tweet_dir + os.sep + 'ids_{}.json'.format(user_id)
    with open(tweet_file, 'r') as f:
        # {'data_type': data_type, 'id': _id, 'context': [{'word': [], 'char': [[], []]}]}
        tweet_dict = json.loads(f.read())
        contexts = tweet_dict['content']
    # construct word char ids
    tweets_word_idx = [] # shape 2 (tweet_cnt, word_limit)
    tweets_char_idx = [] # shape 3 (tweet_cnt, word_limit, char_limit)
    tweet_cnt = 0
    cur_tweets_cnt = len(tweet_ids) 
    while tweet_cnt < tweet_limit:
        if tweet_cnt < cur_tweets_cnt:
            tweet_index = tweet_ids[tweet_cnt]
            context = contexts[tweet_index]
            word_ids, char_ids = get_word_char_ids(context, word_limit, char_limit)
        else:
            word_ids = [0 for i in range(word_limit)]
            char_ids = [[0 for j in range(char_limit)] for i in range(word_limit)]
        tweets_word_idx.append(word_ids)
        tweets_char_idx.append(char_ids)
        tweet_cnt += 1

    return tweets_word_idx, tweets_char_idx
            

def get_url_word_char_ids(url_id, word_limit=200, char_limit=16):
    url_dir = '/home/dyou/url_recsys/dl_data/encode_url'
    url_file = url_dir + os.sep + 'ids_{}.json'.format(url_id)
    with open(url_file, 'r') as f:
         # {'data_type': data_type, 'id': _id, 'content': [{'word': [], 'char': [[], []]}]}
        url_content = json.loads(f.read())
        word_idx, char_idx = get_word_char_ids(url_content['content'][0], word_limit, char_limit)
    return word_idx, char_idx

def get_url_word_char_ids_parag(url_id, parag=20, word_limit=50, char_limit=16):
    url_dir = '/home/dyou/url_recsys/dl_data/encode_url'
    url_file = url_dir + os.sep + 'ids_{}.json'.format(url_id)
    with open(url_file, 'r') as f:
         # {'data_type': data_type, 'id': _id, 'content': [{'word': [], 'char': [[], []]}]}
        url_content = json.loads(f.read())
        word_idx, char_idx = get_word_char_ids(url_content['content'][0], parag*word_limit, char_limit)

    url_word_idxs, url_char_idxs = [], []
    for i in range(parag):
        url_word_idxs.append(word_idx[i*word_limit: (i+1)*word_limit])
        url_char_idxs.append(char_idx[i*word_limit: (i+1)*word_limit])
    return url_word_idxs, url_char_idxs


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


def get_batch_case(all_samples, index_list, no, file_path):
    print (datetime.now(), 'batch size:', len(index_list), 'batch start:', no, index_list[0])
    user_idxs = []
    url_idxs = []
    url_word_idxs = []
    url_char_idxs = []
    tweets_word_idxs = []
    tweets_char_idxs = []
    labels = []
    for i in index_list:
        uid, url_id, label = all_samples[i]
        user_idx = int(uid)
        url_idx = int(url_id)
        url_word_idx, url_char_idx = get_url_word_char_ids_parag(url_id, parag=20, word_limit=50, char_limit=16) 
        #url_word_idx2, url_char_idx2 = get_url_word_char_ids(url_id)
        tweets_word_idx, tweets_char_idx = get_tweets_word_char_ids(uid, url_id)
        y = int(label)
        user_idxs.append(user_idx)
        url_idxs.append(url_idx)
        url_word_idxs.append(url_word_idx)
        url_char_idxs.append(url_char_idx)
        tweets_word_idxs.append(tweets_word_idx)
        tweets_char_idxs.append(tweets_char_idx)
        labels.append(label)

    #print(np.array(user_idxs).shape)
    #print(np.array(url_idxs).shape)
    #print(np.array(url_word_idxs).shape)
    #print(np.array(url_char_idxs).shape)
    #print(np.array(tweets_word_idxs).shape)
    #print(np.array(tweets_char_idxs).shape)

    user_idxs = torch.tensor(user_idxs, dtype=torch.long)
    url_idxs = torch.tensor(url_idxs, dtype=torch.long)
    url_word_idxs = torch.tensor(url_word_idxs, dtype=torch.long)
    #url_word_inverse_idxs = get_inverse_idx(url_word_idxs, 'url', 'word')

    url_char_idxs = torch.tensor(url_char_idxs, dtype=torch.long)
    #url_char_inverse_idxs = get_inverse_idx(url_char_idxs, 'url', 'char')

    tweets_word_idxs = torch.tensor(tweets_word_idxs, dtype=torch.long)
    #tweets_word_inverse_idxs = get_inverse_idx(tweets_word_idxs, 'tweets', 'word')

    tweets_char_idxs = torch.tensor(tweets_char_idxs, dtype=torch.long)
    #tweets_char_inverse_idxs = get_inverse_idx(tweets_char_idxs, 'tweets', 'char')
    #print(labels, len(labels))
    labels = torch.tensor(labels, dtype=torch.long)

    #print('user_idxs:', sys.getsizeof(user_idxs))
    #print('url_idxs:', sys.getsizeof(url_idxs))
    #print('url_word_idxs:', sys.getsizeof(url_word_idxs))
    #print('url_char_idxs:', sys.getsizeof(url_char_idxs))
    #print('tweets_word_idxs:', sys.getsizeof(tweets_word_idxs))
    #print('tweets_char_idxs:', sys.getsizeof(tweets_char_idxs))

    torch.save(user_idxs, file_path + os.sep + str(no) + '_user_idxs')
    torch.save(url_idxs, file_path + os.sep + str(no) + '_url_idxs')
    torch.save(url_word_idxs, file_path + os.sep + str(no) + '_url_word_idxs')
    #torch.save(url_word_inverse_idxs, file_path + os.sep + str(no) + '_url_word_inverse_idxs')
    torch.save(url_char_idxs, file_path + os.sep + str(no) + '_url_char_idxs')
    #torch.save(url_char_inverse_idxs, file_path + os.sep + str(no) + '_url_char_inverse_idxs')
    torch.save(tweets_word_idxs, file_path + os.sep + str(no) + '_tweets_word_idxs')
    #torch.save(tweets_word_inverse_idxs, file_path + os.sep + str(no) + '_tweets_word_inverse_idxs')
    torch.save(tweets_char_idxs, file_path + os.sep + str(no) + '_tweets_char_idxs')
    #torch.save(tweets_char_inverse_idxs, file_path + os.sep + str(no) + '_tweets_char_inverse_idxs')
    torch.save(labels, file_path + os.sep + str(no) + '_labels')
    #print(file_path + os.sep + str(no) + '_labels')
    print (datetime.now(), 'batch size:', len(index_list), 'batch end:', no)
    #return user_idxs, url_idxs, url_word_idxs, url_char_idxs, tweets_word_idxs, tweets_char_idxs, labels


def get_batch_case_one_by_one(all_samples, index_list, no, file_path):
    print (datetime.now(), 'batch size:', len(index_list), 'batch start:', no, index_list[0])
    #user_idxs = []
    #url_idxs = []
    #url_word_idxs = []
    #url_char_idxs = []
    #tweets_word_idxs = []
    #tweets_char_idxs = []
    #labels = []
    for i in index_list:
        uid, url_id, label = all_samples[i]
        user_idx = int(uid)
        url_idx = int(url_id)
        #url_word_idx, url_char_idx = get_url_word_char_ids_parag(url_id, parag=20, word_limit=50, char_limit=16) 
        #url_word_idx2, url_char_idx2 = get_url_word_char_ids(url_id)
        tweets_word_idx, tweets_char_idx = get_tweets_word_char_ids(uid, url_id)
        y = int(label)
        #user_idxs.append(user_idx)
        #url_idxs.append(url_idx)
        #url_word_idxs.append(url_word_idx)
        #url_char_idxs.append(url_char_idx)
        tweets_word_idx = torch.tensor(tweets_word_idx, dtype=torch.long)
        tweets_char_idx = torch.tensor(tweets_char_idx, dtype=torch.long)
        torch.save(tweets_word_idx, file_path + os.sep + 'tweets_word' + os.sep + str(uid) + '_' + str(url_id) + '_tweets_word_idx')
        torch.save(tweets_char_idx, file_path + os.sep + 'tweets_char' + os.sep + str(uid) + '_' + str(url_id) + '_tweets_char_idx')
        
    print (datetime.now(), 'batch size:', len(index_list), 'batch end:', no)
    #return user_idxs, url_idxs, url_word_idxs, url_char_idxs, tweets_word_idxs, tweets_char_idxs, labels

def get_batch_indexs_list(indexs, batch_size):
    indexs_list = []
    batch_cnt = 0
    batch_list = [] 
    for index in indexs:
        batch_cnt += 1
        batch_list.append(index)
        if batch_cnt == batch_size:
            indexs_list.append(batch_list)
            batch_cnt = 0
            batch_list = []
    if len(batch_list) != 0:
        indexs_list.append(batch_list)
    return indexs_list



if __name__ == '__main__':
    target_dir = '/home/dyou/url_recsys/dl_data/batch_tweets'
    os.system('rm -rf %s; mkdir -p %s; mkdir -p %s' 
        % (target_dir + os.sep + 'train', target_dir + os.sep + 'train' + os.sep + 'tweets_word',
            target_dir + os.sep + 'train' + os.sep + 'tweets_char'))
    #os.system('rm -rf %s; mkdir -p %s' % (target_dir + os.sep + 'test', target_dir + os.sep + 'test'))
    all_samples = build_train_data()
    #print(len(all_samples), '---------------------------------')
    sample_index = list(range(len(all_samples)))
    random.shuffle(sample_index)
    batch_size = 128
    indexs_list = get_batch_indexs_list(sample_index, batch_size)
    #with open('/home/dyou/url_recsys/dl_data/batch_parag_sample_index.json', 'w') as f:
    #    f.write(json.dumps(indexs_list) + '\n')
    print(datetime.now(), 'get train batch size: %s, total batch: %d' % (batch_size, len(indexs_list)))
    p_list = []
    pool = multiprocessing.Pool(processes=8)
    for i in range(len(indexs_list)):
        index_list = indexs_list[i]
        #get_batch_case(all_samples, index_list, i, target_dir+os.sep+'train')
        pool.apply_async(get_batch_case_one_by_one, (all_samples, index_list, i, target_dir+os.sep+'train'))
        #p = Process(target=get_batch_case, args=(all_samples, index_list, i, target_dir+os.sep+'train'))
    pool.close()
    pool.join()
        
    '''
    all_test_samples = build_test_data()
    sample_index = list(range(len(all_test_samples)))
    batch_size = 100
    indexs_list = get_batch_indexs_list(sample_index, batch_size)
    print(datetime.now(), 'get test batch size: %s, total batch: %d' % (batch_size, len(indexs_list)))
    pool = multiprocessing.Pool(processes=8)
    for i in range(len(indexs_list)):
        index_list = indexs_list[i]
        pool.apply_async(get_batch_case, (all_test_samples, index_list, i, target_dir+os.sep+'test'))
    pool.close()
    pool.join()
    '''