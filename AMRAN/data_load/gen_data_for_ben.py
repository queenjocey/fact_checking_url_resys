import os
import json
import random

train_json_file = '/home/dyou/url_recsys/dl_data/sample/train_dl_random.txt'
valid_file = '/home/dyou/url_recsys/data/valid.txt'
test_file = '/home/dyou/url_recsys/data/test.txt'

user_feature_file = '/home/dyou/url_recsys/dl_data/feature/user_feature_list.json'
user_follow_file = '/home/dyou/url_recsys/data_gen/data_181110/unique_directed_edges.csv'
uid_mapped_file = '/home/dyou/url_recsys/data_gen/data_181110/uid_mapped.txt'
url_mapped_file = '/home/dyou/url_recsys/data_gen/data_181110/url_mapped.txt'

out_dir = '/home/dyou/data_for_ben'


idx2user_dict = {}
#f = open(out_dir + os.sep + 'user_feature.csv')
#f.write(','.join(['uid', 'user_name', 'followers_cnt', 'following', 'post_urls', 'post_cnt', 'favorite_category', 'favorite_website']))
with open(user_feature_file) as f:
    user_feature_list = json.loads(f.read())
    assert len(user_feature_list) == 11576
    ##User level: ID, # of followers, # of following, posts urls, post cnt, favorite category, favorite website, uid, name
    for item in user_feature_list:
        # item[7] is string
        idx2user_dict[int(item[0])] = item[7]
    json.dump(idx2user_dict, open(out_dir + os.sep + 'idx2user_dict.json', 'w'))

'''idx2url_dict = {}
#f = open(out_dir + os.sep + 'user_feature.csv')
#f.write(','.join(['uid', 'user_name', 'followers_cnt', 'following', 'post_urls', 'post_cnt', 'favorite_category', 'favorite_website']))
with open(user_feature_file) as f:
    url_feature_list = json.loads(f.read())
    assert len(url_feature_list) == 4732
    #url_f = [int(url_no), cate2idx[cate], site2idx[url_site], post_user_bin, post_freq_bin, url]
    for item in user_feature_list:
        # item[5] is string
        idx2user_dict[int(item[0])] = item[5]
    json.dump(idx2url_dict, open(out_dir + os.sep + 'idx2url_dict.json', 'w'))'''
idx2url_dict = {}
with open(url_mapped_file) as f:
    lines = f.readlines()
for line in lines:
    url_id, url = line.strip().split('\t')
    idx2url_dict[int(url_id)] = url
assert len(idx2url_dict) == 4732
json.dump(idx2url_dict, open(out_dir + os.sep + 'idx2url_dict.json', 'w'))


# test
with open(test_file) as f:
    lines = f.readlines()
# uid, url_id, freq, ts
all_test_samples = []
f = open(out_dir + os.sep + 'test.csv', 'w')
f.write(','.join(['user', 'url', 'label']) + '\n')
for line in lines:
    uid, url_id, freq, ts = line.strip().split()
    f.write(','.join([idx2user_dict[int(uid)], idx2url_dict[int(url_id)], freq]) + '\n')
    all_test_samples.append([int(uid), int(url_id), 0 if int(freq) == 0 else 1])
print('test case:', len(all_test_samples))
f.close()

# valid
with open(valid_file) as f:
    lines = f.readlines()
# uid, url_id, freq, ts
all_valid_samples = []
f = open(out_dir + os.sep + 'valid.csv', 'w')
f.write(','.join(['user', 'url', 'label']) + '\n')
for line in lines:
    uid, url_id, freq, ts = line.strip().split()
    f.write(','.join([idx2user_dict[int(uid)], idx2url_dict[int(url_id)], freq]) + '\n')
    all_valid_samples.append([int(uid), int(url_id), 0 if int(freq) == 0 else 1])
print('valid case:', len(all_valid_samples))
f.close()


# train
with open(train_json_file) as f:
    all_samples = json.loads(f.read())
print('train case:', len(all_samples), all_samples[0])
train_dict = {}
for item in all_samples:
    if item[0] not in train_dict:
        train_dict[item[0]] = [[], []]
    if item[2] == 1:
        #print(train_dict[item[0]][0], item[1])
        train_dict[item[0]][0].append(item[1])
    else:
        train_dict[item[0]][1].append(item[1])

keys = list(train_dict.keys())
keys.sort()

## check test
for item in all_test_samples:
    if item[2] == 1:
        if item[1] in train_dict[item[0]][0]:
            print('test in train error')
print('check test in train done')

## check valid
for item in all_valid_samples:
    if item[2] == 1:
        if item[1] not in train_dict[item[0]][0]:
            print('valid not in train_dl error')
        else:
            train_dict[item[0]][0].remove(item[1])
print('check valid in train done')


# out train
f = open(out_dir + os.sep + 'train.csv', 'w')
f.write(','.join(['user', 'url', 'label']) + '\n')
count = 0
for uid in keys:
    pos = train_dict[uid][0]
    neg = train_dict[uid][1]
    assert len(neg) == len(set(neg))
    #random.shuffle(neg)
    for url_id in pos:
        f.write(','.join([idx2user_dict[int(uid)], idx2url_dict[int(url_id)], '1']) + '\n')
        count += 1
    for url_id in neg[:-5]:
        f.write(','.join([idx2user_dict[int(uid)], idx2url_dict[int(url_id)], '0']) + '\n')
        count += 1
print('train done:', count)
