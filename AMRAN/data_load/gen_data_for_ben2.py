import os
import json
import random

train_json_file = '/home/dyou/url_recsys/dl_data/sample/train_dl_random.txt'
with open(train_json_file) as f:
    all_samples = json.loads(f.read())
print('all samples:', len(all_samples))
all_samples = ['_'.join([str(item) for item in sample]) for sample in all_samples]

ben_train_file = '/home/dyou/gau_data/train.csv'
ben_url_map = '/home/dyou/gau_data/idx2url_dict.json'
ben_user_map = '/home/dyou/gau_data/idx2user_dict.json'

with open(ben_url_map) as f:
    json_dict = json.loads(f.read())
#print(json_dict)
url2idx_dict = {v: int(k) for k, v in json_dict.items()}

with open(ben_user_map) as f:
    json_dict = json.loads(f.read())
user2idx_dict = {v: int(k) for k, v in json_dict.items()}

# out train
with open(ben_train_file) as f:
    lines = f.readlines()

samples = []
for line in lines[1:]:
    user, url, freq = [item.strip() for item in line.split(',')]
    sample = [user2idx_dict[user], url2idx_dict[url], 1 if int(freq) > 0 else 0]
    samples.append(sample)

json.dump(samples, open('/home/dyou/url_recsys/dl_data/sample/train_dl_random_ben.txt', 'w'))

for sample in samples:
    sample_str = '_'.join([str(item) for item in sample])
    assert sample_str in all_samples
print('check success:', len(samples))
