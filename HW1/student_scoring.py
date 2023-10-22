# load test
import os
import pandas as pd

with open('answer.txt', 'r', encoding="utf-8") as f:
    all = f.readlines()
ans = [[one.split(', ')[i] for i in range(2)] for one in all]
ans_dict = {}
for item in ans:
    ans_dict[int(item[0])] = item[1][:-1]

remove_list = ['0019.wav', '0020.mp3', '0055.wav', '0060.mp3', '0085.wav', '0107.wav', '0118.wav', '0121.wav', \
    '0134.mp3', '0180.wav', '0210.wav', '0257.mp3', '0275.mp3', '0283.wav', '0284.wav', '0302.wav', '0332.wav', '0347.wav', \
        '0371.wav', '0408.wav', '0417.wav', '0479.wav', '0482.wav', '0526.wav', '0564.wav', '0588.mp3', '0632.wav', '0700.wav', \
            '0701.wav', '0770.wav', '0822.wav', '0827.wav', '0832.wav', '0902.wav', '0924.wav', '0929.wav', '0980.wav']

remove_list = [int(x[:4]) for x in remove_list]
n_remove = len(remove_list)

def Top1_acc(pred, ans_dict):
    correct = 0
    for pre in pred:
        if pre[0] not in remove_list:
            if ans_dict[pre[0]] == pre[1][:]:
                correct += 1
    return correct / (970-n_remove)

def Top3_acc(pred, ans_dict):
    correct = 0
    for pre in pred:
        if pre[0] not in remove_list:
            ans = ans_dict[pre[0]]
            if  ans == pre[1][:] or ans == pre[2][:] or ans == pre[3][:]:
                correct += 1
    return correct / (970-n_remove)  
  
data_path = '1696395933_761___COMME5070-HW1_prediction_submissions'
file_list = [os.path.join(data_path, i) for i in os.listdir(data_path)]
data = [['ID', 'Top1 accuracy', 'Top3 accuracy', 'Rank score']]
for file in file_list:
    pred = pd.read_csv(file, header=None)
    pred = pred.to_numpy().tolist()
    acc = Top1_acc(pred, ans_dict)
    top3_acc = Top3_acc(pred, ans_dict)
    rank_score = acc + 0.5 * top3_acc
    data.append([os.path.basename(file).split('#')[0], f'{acc:.5f}', f'{top3_acc:.5f}', f'{rank_score:.5f}'])

csv_file_path = 'score_result_20231004.csv'

df = pd.DataFrame(data)
df.to_csv(csv_file_path, index=False, header=False)