import sys
import numpy as np

out_str = []
hr_list = []
ndcg_list = []
cost_list = []
for line in sys.stdin:
    if 'lr' in line:
        out_str.append(line.strip())
    elif 'train loss' in line:
        # 2019-05-19 03:36:39.845724 epoch 9 end, train loss: 262.5918031781912, test case: 1157600, test hr: 0.6477194194885971, ncdg： 0.4004153915666544, avg_diff_cnt: 99.99930891499655
        hr = float(line.split('test hr:')[1].split(',')[0])
        hr_list.append(hr)
        ndcg = float(line.replace('：', ':').split('ncdg:')[1].split(',')[0])
        ndcg_list.append(ndcg)

    elif 'end train epoch' in line: 
        #2019-05-19 03:36:38.710244, end train epoch 8, total cost 466.928422
        #print(line.split())
        cost_list.append(float(line.split()[8]))

if len(hr_list) > 0:
    print(out_str[0], 'epoch:', len(hr_list), np.mean(cost_list), 'max:', np.max(hr_list), np.max(ndcg_list)
    , 'mean:', np.mean(hr_list), np.mean(ndcg_list)
    )
else:
    print(out_str[0])
#print (out_str, cost_list, hr_list, ndcg_list)