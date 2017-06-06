# -*- coding: utf-8 -*-

from hyperopt import fmin, tpe, hp, Trials, STATUS_OK
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import cross_val_score
from sklearn.ensemble import ExtraTreesClassifier

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
    clf = ExtraTreesClassifier(**params)
    scores = cross_val_score(clf, cv_train_data, train_data['GROUP_ID'], scoring='accuracy', cv=3).mean()
    print "Scores : "
    print scores.mean()
    return scores.mean()

def f(params):
    acc = hyperopt_rf_opt(params)
    return {'loss': 1-acc, 'status': STATUS_OK}

random_state = 11
n_jobs = 4
criterion = ['gini', 'entropy']
max_evals = 20

space4et = {
    'features': hp.choice('features', range(4000, 20000)),
    'n_estimators': hp.choice('n_estimators', range(100,600)),
    'criterion': hp.choice('criterion', ["gini", "entropy"]),
    'random_state': random_state,
    'n_jobs': n_jobs

}


trials = Trials()
best = fmin(f, space4et, algo=tpe.suggest, max_evals=max_evals, trials=trials)
print 'best:'
print best

cv = TfidfVectorizer(max_features=best['features'])
cv_train_data = cv.fit_transform(train_data['NAME'])
cv_test_data = cv.transform(test_data['NAME'])

best = ExtraTreesClassifier(n_estimators=best['n_estimators'], criterion=criterion[best['criterion']],
                            random_state=random_state, n_jobs=n_jobs)

print 'Score of best: '
print cross_val_score(best, cv_train_data, train_data['GROUP_ID'], cv=3, scoring='accuracy').mean()

clf = ExtraTreesClassifier(n_estimators=best['n_estimators'], criterion=criterion[best['criterion']],
                            random_state=random_state, n_jobs=n_jobs)
clf.fit(cv_train_data, train_data['GROUP_ID'])

predict_clf = clf.predict(cv_test_data)
test_data['GROUP_ID'] = predict_clf
test_data[['id', 'GROUP_ID']].to_csv('./../result/res_et_0606.csv', sep=',', header=True, index=False)