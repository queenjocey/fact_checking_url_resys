import os
import json
import random

train_json_file = '/home/dyou/url_recsys/dl_data/sample/train_dl_random.txt'
#valid_file = '/home/dyou/url_recsys/data/valid.txt'
test_file = '/home/dyou/url_recsys/data/test.txt'

user_feature_file = '/home/dyou/url_recsys/dl_data/feature/user_feature_list.json'
user_follow_file = '/home/dyou/url_recsys/data_gen/data_181110/unique_directed_edges.csv'
uid_mapped_file = '/home/dyou/url_recsys/data_gen/data_181110/uid_mapped.txt'
url_mapped_file = '/home/dyou/url_recsys/data_gen/data_181110/url_mapped.txt'

out_dir = '/home/dyou/data_for_ngcf'

# test
test_pos = {}
test_negs = {}
with open(test_file) as f:
    lines = f.readlines()
# uid, url_id, freq, ts
for line in lines:
    uid, url_id, freq, ts = line.strip().split()
    if int(freq) > 0:
        if uid not in test_pos:
            test_pos[uid] = []
        test_pos[uid].append(url_id)
    else:
        if uid not in test_negs:
            test_negs[uid] = []
        test_negs[uid].append(url_id)


uids = list(test_pos.keys())
#uids.sort()
uids = sorted(uids, key=lambda x: int(x))
f1 = open(out_dir + os.sep + 'test.txt', 'w')
f2 = open(out_dir + os.sep + 'test_negs.txt', 'w')
for uid in uids:
    f1.write(uid + ' ' + ' '.join(test_pos[uid]) + '\n')
    f2.write(uid + ' ' + ' '.join(test_negs[uid]) + '\n')
    #print(len(test_negs[uid]))
    assert len(test_negs[uid]) == 99
    assert len(test_pos[uid]) == 1
print('test:', len(test_pos), len(test_negs))
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
# out train
f = open(out_dir + os.sep + 'train.txt', 'w')
count = 0
for uid in keys:
    pos = train_dict[uid][0]
    neg = train_dict[uid][1]
    f.write(str(uid) + ' ' + ' '.join([str(item) for item in pos]) + '\n')
    count += len(pos)

print('train user:', len(keys), 'train items:', count)


'''
test: 11576 11576
train case: 311118 [0, 2226, 1]
train user: 11576 train items: 51853
'''