# -*- coding: utf-8 -*-

from hyperopt import fmin, tpe, hp, Trials, STATUS_OK
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier

train_data = pd.read_csv('./../data/lem_train_data.csv', sep=';', header=0)
test_data = pd.read_csv('./../data/lem_test_data.csv', sep=';', header=0)

def hyperopt_rf_opt(params):
    print "Training with params : "
    print params

    # create features and del parameter
    cv = TfidfVectorizer(max_features=params['features'])
    cv_train_data = cv.fit_transform(train_data['NAME'])
    del params['features']

    # learn classifier
    clf = XGBClassifier(**params)
    scores = cross_val_score(clf, cv_train_data, train_data['GROUP_ID'], scoring='accuracy', cv=3).mean()
    print "Scores : "
    print scores.mean()
    return scores.mean()

def f(params):
    acc = hyperopt_rf_opt(params)
    return {'loss': 1-acc, 'status': STATUS_OK}

random_state = 27
n_jobs = 8
max_evals = 100
penalty = ['l1', 'l2']

space4xgb = {
    'features': hp.choice('features', range(4000, 15000)),
    'n_estimators': hp.choice('n_estimators', range(50,450)),
    'max_depth': hp.choice('max_depth', range(1,7)),
    'learning_rate': hp.uniform('learning_rate', 0, 1),
    'min_child_weight': hp.choice('min_child_weight', range(10,20)),
    'reg_alpha': hp.uniform('reg_alpha', 1, 3),
    'subsample': hp.uniform('subsample', 0, 1),
    'colsample_bytree': hp.uniform('colsample_bytree', 0.2, 1),
    'objective': 'multi:softmax',
    'seed': random_state,
    'nthread': n_jobs

}


trials = Trials()
best = fmin(f, space4xgb, algo=tpe.suggest, max_evals=max_evals, trials=trials)
print 'best:'
print best


cv = TfidfVectorizer(max_features=best['features'])
cv_train_data = cv.fit_transform(train_data['NAME'])
cv_test_data = cv.transform(test_data['NAME'])

best = XGBClassifier(reg_alpha=best['reg_alpha'], colsample_bytree=best['colsample_bytree'],
                          learning_rate=best['learning_rate'], min_child_weight=best['min_child_weight'],
                          n_estimators=best['n_estimators'], subsample=best['subsample'], max_depth=best['max_depth'],
                          seed=random_state, objective='multi:softmax', nthread=n_jobs)

print 'Score of best: '
print cross_val_score(best, cv_train_data, train_data['GROUP_ID'], cv=3, scoring='accuracy').mean()


best.fit(cv_train_data, train_data['GROUP_ID'])

predict_clf = best.predict(cv_test_data)
test_data['GROUP_ID'] = predict_clf
test_data[['id', 'GROUP_ID']].to_csv('./../result/res_xgb_0506.csv', sep=',', header=True, index=False)