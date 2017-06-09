# -*- coding: utf-8 -*-

from hyperopt import fmin, tpe, hp, Trials, STATUS_OK
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.model_selection import cross_val_score
from sklearn.ensemble import RandomForestClassifier

train_data = pd.read_csv('./../../data/lem_train_data.csv', sep=';', header=0)
test_data = pd.read_csv('./../../data/lem_test_data.csv', sep=';', header=0)

def hyperopt_rf_opt(params):
    print "Training with params : "
    print params

    # create features and del parameter
    cv = CountVectorizer(max_features=params['features'])
    cv_train_data = cv.fit_transform(train_data['NAME'])
    del params['features']

    # learn classifier
    rf = RandomForestClassifier(**params)
    scores = cross_val_score(rf, cv_train_data, train_data['GROUP_ID'], cv=3, scoring='accuracy')
    print "Scores : "
    print scores.mean()
    return scores.mean()

def f(params):
    acc = hyperopt_rf_opt(params)
    return {'loss': 1-acc, 'status': STATUS_OK}

random_state = 11
n_jobs = 8
max_evals = 50
criterion = ['gini', 'entropy']

space4rf = {
    'features': hp.choice('features', range(4000, 20000)),
    'n_estimators': hp.choice('n_estimators', range(100,600)),
    'criterion': hp.choice('criterion', ["gini", "entropy"]),
    'random_state': random_state,
    'n_jobs': n_jobs
}


trials = Trials()
best = fmin(f, space4rf, algo=tpe.suggest, max_evals=max_evals, trials=trials)
print 'best:'
print best

cv = CountVectorizer(max_features=best['features'])
cv_train_data = cv.fit_transform(train_data['NAME'])
cv_test_data = cv.transform(test_data['NAME'])

best = RandomForestClassifier(n_estimators=best['n_estimators'], criterion=criterion[best['criterion']],
                            random_state=random_state, n_jobs=n_jobs)

print 'Score of best: '
print cross_val_score(best, cv_train_data, train_data['GROUP_ID'], cv=3, scoring='accuracy').mean()

best.fit(cv_train_data, train_data['GROUP_ID'])

predict_train = best.predict(cv_train_data)
predict_test = best.predict(cv_test_data)

test_data['GROUP_ID'] = predict_test
test_data[['id', 'GROUP_ID']].to_csv('./../../result/res_rf_test_0906.csv', sep=',', header=True, index=False)

train_data['PREDICT_GROUP_ID'] = predict_train
train_data[['id', 'GROUP_ID', 'PREDICT_GROUP_ID']].to_csv('./../../result/res_rf_train_0906.csv', sep=',', header=True, index=False)