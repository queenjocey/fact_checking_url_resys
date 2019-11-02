import os
import json

target_dir = '/home/dyou/url_recsys/dl_data'
with open(target_dir + os.sep + 'word2idx.json') as f:
    word2idx_dict = json.loads(f.read())
with open(target_dir + os.sep + 'char2idx.json') as f:
    char2idx_dict = json.loads(f.read())
print (len(word2idx_dict), len(char2idx_dict))

idx2word = {idx: word for (word, idx) in word2idx_dict.items()}
idx2char = {idx: word for (word, idx) in char2idx_dict.items()}

#with open('/home/dyou/url_recsys/dl_data/train_dl.txt') as f:
#    all_samples = json.loads(f.read())


_id = 2226

with open(target_dir + os.sep + 'encode_url' + os.sep + 'ids_%d.json' % (_id)) as f:
    res = json.loads(f.read())

out_res = {}
out_res['data_type'] = res['data_type']
out_res['id'] = res['id']
out_res['content'] = []
for wc_dict in res['content']:
    word_id_list = wc_dict['word']
    word_list = [idx2word[w_id] for w_id in word_id_list]
    char_id_lists = wc_dict['char']
    char_list = [[idx2char[c_id] for c_id in c_id_list] for c_id_list in char_id_lists]
    out_res['content'].append({'word': word_list, 'char': char_list})

with open('reverse_ids_%d.json' % (_id), 'w') as f:
    f.write(json.dumps(out_res))

_id = 0

with open(target_dir + os.sep + 'encode_tweets' + os.sep + 'ids_%d.json' % (_id)) as f:
    res = json.loads(f.read())

out_res = {}
out_res['data_type'] = res['data_type']
out_res['id'] = res['id']
out_res['content'] = []
for wc_dict in res['content']:
    word_id_list = wc_dict['word']
    word_list = [idx2word[w_id] for w_id in word_id_list]
    char_id_lists = wc_dict['char']
    char_list = [[idx2char[c_id] for c_id in c_id_list] for c_id_list in char_id_lists]
    out_res['content'].append({'word': word_list, 'char': char_list})

with open('reverse_tweet_ids_%d.json' % (_id), 'w') as f:
    f.write(json.dumps(out_res))