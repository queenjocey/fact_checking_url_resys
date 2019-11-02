'''
20190401  no tfidf chose
'''

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

import os
import sys
import random
import math
import json

import numpy as np
from datetime import datetime
from multiprocessing import Process
import multiprocessing


def get_word_char_ids(context, word_limit, char_limit):
    # {'word': [], 'char': [[], []]}
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


def get_tweets_word_char_ids(user_id, tweet_limit=50, word_limit=30, char_limit=16):
    # read one user all tweets
    tweet_dir = '/home/dyou/url_recsys/dl_data/encode_tweets'
    tweet_file = tweet_dir + os.sep + 'ids_{}.json'.format(user_id)
    with open(tweet_file, 'r') as f:
        # {'data_type': data_type, 'id': _id, 'context': [{'word': [], 'char': [[], []]}]}
        tweet_dict = json.loads(f.read())
        contexts = tweet_dict['content']
    # construct word char ids
    tweets_word_idx = [] # shape 2 (tweet_cnt, word_limit)
    tweets_char_idx = [] # shape 3 (tweet_cnt, word_limit, char_limit)
    tweet_cnt = 0
    cur_tweets_cnt = len(contexts) 
    while tweet_cnt < tweet_limit:
        if tweet_cnt < cur_tweets_cnt:
            context = contexts[tweet_cnt]
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
         # {'data_type': data_type, 'id': _id, 'context': [{'word': [], 'char': [[], []]}]}
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


if __name__ == '__main__':
    target_dir = '/home/dyou/url_recsys/dl_data/batch_fixed'
    tweet_dir = '/home/dyou/url_recsys/dl_data/encode_tweets'
    all_tweets_word_idx = []
    all_tweets_char_idx = []
    tweet_limit = 50
    word_limit = 30
    char_limit = 16
    for user_id in range(11576):
        tweets_word_idx, tweets_char_idx = get_tweets_word_char_ids(user_id, tweet_limit=50, word_limit=30, char_limit=16)
        all_tweets_word_idx.append(tweets_word_idx)
        all_tweets_char_idx.append(tweets_char_idx)

    all_tweets_word_idx = torch.tensor(all_tweets_word_idx, dtype=torch.long)
    all_tweets_char_idx = torch.tensor(all_tweets_char_idx, dtype=torch.long)

    torch.save(all_tweets_word_idx, target_dir + os.sep + 'all_tweets_word_idx')
    torch.save(all_tweets_char_idx, target_dir + os.sep + 'all_char_word_idx')
    print('tweets done')

    url_dir = '/home/dyou/url_recsys/dl_data/encode_url'
    all_url_word_idx = []
    all_url_char_idx = []
    for url_id in range(4732):
        url_word_idx, url_char_idx = get_url_word_char_ids_parag(url_id, parag=20, word_limit=50, char_limit=16)
        all_url_word_idx.append(url_word_idx)
        all_url_char_idx.append(url_char_idx)

    all_url_word_idx = torch.tensor(all_url_word_idx, dtype=torch.long)
    all_url_char_idx = torch.tensor(all_url_char_idx, dtype=torch.long)
    torch.save(all_url_word_idx, target_dir + os.sep + 'all_url_word_idx')
    torch.save(all_url_char_idx, target_dir + os.sep + 'all_url_char_idx')
    print('url done')
