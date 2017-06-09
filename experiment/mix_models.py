# -*- coding: utf-8 -*-
import pandas as pd
from sklearn.model_selection import cross_val_score
from sklearn.ensemble import RandomForestClassifier

# загрузка данных
predicts = pd.read_csv('./../result/res_lr_0506.csv', header=0, sep=',')

for f in ['res_nb_0606.csv', 'res_rf_0506.csv', 'res_xgb_0606.csv']:
    temp = pd.read_csv('./../result/' + f, header=0, sep=',')
    predicts[f] = temp['GROUP_ID']

print predicts.head()

# голосование максимумом

def max_c(x):
    return x.value_counts().idxmax()

predicts['result'] = predicts.apply(max_c, axis=1)

print predicts.head()
predicts[['id', 'result']].to_csv('./../result/res_all_0606.csv', sep=',', header=('id', 'GROUP_ID'), index=False)