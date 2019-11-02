# 20190520 final
buzz 
ps -ef | grep python | grep dyou | awk '{print $2}' | xargs kill -9
nohup python -u main6_main.py gpu 0 lr 0.0003 dim 64 l2 0 top_ratio 0.25 v_num 8 layer 2 neg_num 5 neighb_num 10 1_1 > log6/main6_1_1.log 2>&1 &
nohup python -u main6_main.py gpu 1 lr 0.0003 dim 64 l2 0 top_ratio 0.25 v_num 8 layer 2 neg_num 4 neighb_num 10 2_1 > log6/main6_2_1.log 2>&1 &
nohup python -u main6_main.py gpu 0 lr 0.0003 dim 64 l2 0 top_ratio 0.25 v_num 8 layer 2 neg_num 3 neighb_num 10 3_1 > log6/main6_3_1.log 2>&1 &
nohup python -u main6_main.py gpu 1 lr 0.0003 dim 64 l2 0 top_ratio 0.25 v_num 8 layer 2 neg_num 2 neighb_num 10 4_1 > log6/main6_4_1.log 2>&1 &
nohup python -u main6_main.py gpu 0 lr 0.0003 dim 64 l2 0 top_ratio 0.25 v_num 8 layer 2 neg_num 1 neighb_num 10 5_1 > log6/main6_5_1.log 2>&1 &

nohup python -u main6_main.py gpu 1 lr 0.0003 dim 8 l2 0 top_ratio 0.25 v_num 8 layer 2 neg_num 5 neighb_num 10 16_1 > log6/main6_16_1.log 2>&1 &
nohup python -u main6_main.py gpu 0 lr 0.0003 dim 8 l2 0 top_ratio 0.25 v_num 8 layer 2 neg_num 4 neighb_num 10 17_1 > log6/main6_17_1.log 2>&1 &
nohup python -u main6_main.py gpu 1 lr 0.0003 dim 8 l2 0 top_ratio 0.25 v_num 8 layer 2 neg_num 3 neighb_num 10 18_1 > log6/main6_18_1.log 2>&1 &
nohup python -u main6_main.py gpu 0 lr 0.0003 dim 8 l2 0 top_ratio 0.25 v_num 8 layer 2 neg_num 2 neighb_num 10 19_1 > log6/main6_19_1.log 2>&1 &
nohup python -u main6_main.py gpu 1 lr 0.0003 dim 8 l2 0 top_ratio 0.25 v_num 8 layer 2 neg_num 1 neighb_num 10 20_1 > log6/main6_20_1.log 2>&1 &

cat log6/main6_1_1.log | python -u get_log.py > read_f.log
cat log6/main6_2_1.log | python -u get_log.py >> read_f.log
cat log6/main6_3_1.log | python -u get_log.py >> read_f.log
cat log6/main6_4_1.log | python -u get_log.py >> read_f.log
cat log6/main6_5_1.log | python -u get_log.py >> read_f.log
cat log6/main6_16_1.log | python -u get_log.py >> read_f.log
cat log6/main6_17_1.log | python -u get_log.py >> read_f.log
cat log6/main6_18_1.log | python -u get_log.py >> read_f.log
cat log6/main6_19_1.log | python -u get_log.py >> read_f.log
cat log6/main6_20_1.log | python -u get_log.py >> read_f.log
cat read_f.log



buzz2
ps -ef | grep python | grep dyou | awk '{print $2}' | xargs kill -9
nohup python -u main6_main.py gpu 0 lr 0.0003 dim 32 l2 0 top_ratio 0.25 v_num 8 layer 2 neg_num 5 neighb_num 10 6_1 > log6/main6_6_1.log 2>&1 &
nohup python -u main6_main.py gpu 1 lr 0.0003 dim 32 l2 0 top_ratio 0.25 v_num 8 layer 2 neg_num 4 neighb_num 10 7_1 > log6/main6_7_1.log 2>&1 &
nohup python -u main6_main.py gpu 0 lr 0.0003 dim 32 l2 0 top_ratio 0.25 v_num 8 layer 2 neg_num 3 neighb_num 10 8_1 > log6/main6_8_1.log 2>&1 &
nohup python -u main6_main.py gpu 1 lr 0.0003 dim 32 l2 0 top_ratio 0.25 v_num 8 layer 2 neg_num 2 neighb_num 10 9_1 > log6/main6_9_1.log 2>&1 &
nohup python -u main6_main.py gpu 0 lr 0.0003 dim 32 l2 0 top_ratio 0.25 v_num 8 layer 2 neg_num 1 neighb_num 10 10_1 > log6/main6_10_1.log 2>&1 &

cat log6/main6_6_1.log | python -u get_log.py > read_f.log
cat log6/main6_7_1.log | python -u get_log.py >> read_f.log
cat log6/main6_8_1.log | python -u get_log.py >> read_f.log
cat log6/main6_9_1.log | python -u get_log.py >> read_f.log
cat log6/main6_10_1.log | python -u get_log.py >> read_f.log
cat read_f.log


buzz3
ps -ef | grep python | grep dyou | awk '{print $2}' | xargs kill -9
nohup python -u main6_main.py gpu 0 lr 0.0003 dim 16 l2 0 top_ratio 0.25 v_num 8 layer 2 neg_num 5 neighb_num 10 11_1 > log6/main6_11_1.log 2>&1 &
nohup python -u main6_main.py gpu 1 lr 0.0003 dim 16 l2 0 top_ratio 0.25 v_num 8 layer 2 neg_num 4 neighb_num 10 12_1 > log6/main6_12_1.log 2>&1 &
nohup python -u main6_main.py gpu 0 lr 0.0003 dim 16 l2 0 top_ratio 0.25 v_num 8 layer 2 neg_num 3 neighb_num 10 13_1 > log6/main6_13_1.log 2>&1 &
nohup python -u main6_main.py gpu 1 lr 0.0003 dim 16 l2 0 top_ratio 0.25 v_num 8 layer 2 neg_num 2 neighb_num 10 14_1 > log6/main6_14_1.log 2>&1 &
nohup python -u main6_main.py gpu 0 lr 0.0003 dim 16 l2 0 top_ratio 0.25 v_num 8 layer 2 neg_num 1 neighb_num 10 15_1 > log6/main6_15_1.log 2>&1 &

cat log6/main6_11_1.log | python -u get_log.py > read_f.log
cat log6/main6_12_1.log | python -u get_log.py >> read_f.log
cat log6/main6_13_1.log | python -u get_log.py >> read_f.log
cat log6/main6_14_1.log | python -u get_log.py >> read_f.log
cat log6/main6_15_1.log | python -u get_log.py >> read_f.log
cat read_f.log


buzz
nohup python -u main6_pre.py gpu 0 lr 0.001 no 1 l2 0 > log6/main6_pre1_1.log 2>&1 &
nohup python -u main6_pre.py gpu 1 lr 0.001 no 2 l2 0 > log6/main6_pre1_2.log 2>&1 &

nohup python -u main6_pre.py gpu 0 lr 0.001 no 1 l2 0 f > log6/main6_pre2_1.log 2>&1 &
nohup python -u main6_pre.py gpu 1 lr 0.001 no 2 l2 0 f > log6/main6_pre2_2.log 2>&1 &

cat log6/main6_pre1_1.log | python -u get_log.py > read_f.log
cat log6/main6_pre1_2.log | python -u get_log.py >> read_f.log
cat log6/main6_pre2_1.log | python -u get_log.py >> read_f.log
cat log6/main6_pre2_2.log | python -u get_log.py >> read_f.log
cat log6/main6_f1_1.log | python -u get_log.py >> read_f.log
cat log6/main6_f1_2.log | python -u get_log.py >> read_f.log
cat log6/main6_f2_1.log | python -u get_log.py >> read_f.log
cat log6/main6_f2_2.log | python -u get_log.py >> read_f.log
cat read_f.log


nohup python -u main6_features.py gpu 0 lr 0.001 neg_num 0 neighb_num 10 f1_1 > log6/main6_f1_1.log 2>&1 &
nohup python -u main6_features.py gpu 1 lr 0.001 neg_num 0 neighb_num 10 f1_2 > log6/main6_f1_2.log 2>&1 &

nohup python -u main6_features.py gpu 0 lr 0.001 neg_num 0 neighb_num 10 f2_1 > log6/main6_f2_1.log 2>&1 &
nohup python -u main6_features.py gpu 1 lr 0.001 neg_num 0 neighb_num 10 f2_2 > log6/main6_f2_2.log 2>&1 &

nohup python -u main6_features.py gpu 0 lr 0.001 neg_num 0 neighb_num 10 f2_1 > log6/main6_f3_1.log 2>&1 &
nohup python -u main6_features.py gpu 1 lr 0.001 neg_num 0 neighb_num 10 f2_2 > log6/main6_f3_2.log 2>&1 &
