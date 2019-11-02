import json


train_json_file = '/home/dyou/url_recsys/dl_data/sample/train_dl_random.txt'


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

all_4_samples = []
all_3_samples = []
all_2_samples = []
all_1_samples = []
total_pos = 0
for uid in keys:
    pos_samples = train_dict[uid][0]
    total_pos += len(pos_samples)
    for url_id in pos_samples:
        all_4_samples.append([int(uid), int(url_id), 1])
        all_3_samples.append([int(uid), int(url_id), 1])
        all_2_samples.append([int(uid), int(url_id), 1])
        all_1_samples.append([int(uid), int(url_id), 1])

    for neg_url_id in train_dict[uid][1][0:4*5]:
        all_4_samples.append([int(uid), int(neg_url_id), 0])

    for neg_url_id in train_dict[uid][1][0:3*5]:
        all_3_samples.append([int(uid), int(neg_url_id), 0])

    for neg_url_id in train_dict[uid][1][0:2*5]:
        all_2_samples.append([int(uid), int(neg_url_id), 0])

    for neg_url_id in train_dict[uid][1][0:1*5]:
        all_1_samples.append([int(uid), int(neg_url_id), 0])

print('total pos sample:', total_pos)
print('all 4:', len(all_4_samples))
print('all 3:', len(all_3_samples))
print('all 2:', len(all_2_samples))
print('all 1:', len(all_1_samples))

json.dump(all_4_samples, open('/home/dyou/url_recsys/dl_data/sample/train_dl_random4.txt', 'w'))
json.dump(all_3_samples, open('/home/dyou/url_recsys/dl_data/sample/train_dl_random3.txt', 'w'))
json.dump(all_2_samples, open('/home/dyou/url_recsys/dl_data/sample/train_dl_random2.txt', 'w'))
json.dump(all_1_samples, open('/home/dyou/url_recsys/dl_data/sample/train_dl_random1.txt', 'w'))