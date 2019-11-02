## environment
python >= 3.6
pytorch >= 1.0.0


## data
There is only a test data. We need to pre generate the train and test data because of the resource. Some process code is in the data_loader folder.


## how to train
nohup python -u main6_pre.py gpu 0 lr 0.001 no 1 l2 0 > 1.log 2>&1 &
nohup python -u main6_features.py gpu 0 lr 0.001 neg_num 5 neighb_num 10 > 2.log 2>&1 &
nohup python -u main6_gcn.py gpu 0 lr 0.001 no 1 l2 0 top_ratio 0.25 v_num 3 layer 2 neg_num 5 > 3.log 2>&1 &
nohup python -u main6_main.py gpu 0 lr 0.001 dim 64 l2 0 top_ratio 0.25 v_num 8 layer 2 neg_num 5 neighb_num 10 > 4.log 2>&1 &
