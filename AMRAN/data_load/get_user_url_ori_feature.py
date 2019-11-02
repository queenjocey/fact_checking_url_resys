import os
import json


uid_mapped_file = '/home/dyou/url_recsys/data_gen/data_181110/uid_mapped.txt'
url_mapped_file = '/home/dyou/url_recsys/data_gen/data_181110/url_mapped.txt'
user_url_file = '/home/dyou/url_recsys/data_gen/data_181110/user_url_info.txt'
user_follow_file = '/home/dyou/url_recsys/data_gen/data_181110/unique_directed_edges.csv'


## interaction
with open(user_url_file) as f:
    lines = f.readlines()

user_post_freq = {}
url_post_freq = {}
for line in lines[1:]:
    # uid_no    uid name    url_no  url tweeet_id   datetime    ts  freq
    uid_no, uid, name, url_no, url, tweeet_id, datetime, ts, freq = line.strip().split('\t')
    if uid_no not in user_post_freq:
        user_post_freq[uid_no] = [0, 0]
    user_post_freq[uid_no][0] += 1
    user_post_freq[uid_no][1] += int(freq)

    if url_no not in url_post_freq:
        url_post_freq[url_no] = [0, 0]
    url_post_freq[url_no][0] += 1
    url_post_freq[url_no][1] += int(freq)

print('user: %d, url: %d' % (len(user_post_freq), len(url_post_freq)))


# user follow
with open(user_follow_file) as f:
    lines = f.readlines()

follower_dict = {}
follow_in_dict = {}
for line in lines[1:]:
    #follower,followee    followee = follow in
    #757778071734804480,815708250217840640
    follower, follow_in = line.strip().split(',')
    follower_dict[follower] = follower_dict.get(follower, 0) + 1
    follow_in_dict[follow_in] = follow_in_dict.get(follow_in, 0) + 1

print('follower: %d, follow in: %s' % (len(follower_dict), len(follow_in_dict)))


# output user feature
with open(uid_mapped_file) as f:
    lines = f.readlines()

f_out = open('/home/dyou/url_recsys/data/uid_mapped_feature.csv', 'w')
f_out.write(','.join(['uid_no', 'uid', 'name', 'post_urls', 'post_freq', 'follow_cnt', 'follow_in_cnt']) + '\n')
miss_follower, miss_follow_in = 0, 0
for line in lines:
    #0   767036004   upayr
    info = line.strip().split('\t')
    uid_no, uid = info[0], info[1]
    info.append(user_post_freq[uid_no][0])
    info.append(user_post_freq[uid_no][1])

    if uid not in follower_dict:
        miss_follower += 1
        follower_dict[uid] = 0
    info.append(follower_dict[uid])

    if uid not in follow_in_dict:
        miss_follow_in += 1
        follow_in_dict[uid] = 0
    info.append(follow_in_dict[uid])

    f_out.write(','.join([str(item).strip() for item in info]) + '\n')

f_out.close()
print('process user feature: %d, miss follower: %d, miss follow in: %d' % (len(lines), miss_follower, miss_follow_in))


# output url feature
with open(url_mapped_file) as f:
    lines = f.readlines()

url_site_set = set()
cate_dict = {
    'politifact': 'politics',
    'opensecrets': 'politics',
    'factcheck': 'politics'
}
f_out = open('/home/dyou/url_recsys/data/url_mapped_feature.csv', 'w')
f_out.write(','.join(['url_no', 'url', 'url_site', 'cate', 'post_users', 'post_freq']) + '\n')
for line in lines:
    info = line.strip().split()
    url_no, url = info[0], info[1]
    url_site = url.split('.')[1]
    for key in cate_dict:
        if key in url:
            url_site = key
            break
    url_site_set.add(url_site)
    cate = cate_dict.get(url_site, 'UNK')
    info.append(url_site)
    info.append(cate)

    info.append(url_post_freq[url_no][0])
    info.append(url_post_freq[url_no][1])

    f_out.write(','.join([str(item).strip() for item in info]) + '\n')
f_out.close()
print('process url feature: %d, diff site: %s' % (len(lines), url_site_set))
