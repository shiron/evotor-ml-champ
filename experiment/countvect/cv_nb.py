# -*- coding: utf-8 -*-

from hyperopt import fmin, tpe, hp, Trials, STATUS_OK
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import cross_val_score
from sklearn.naive_bayes import MultinomialNB

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
    clf = MultinomialNB(**params)
    scores = cross_val_score(clf, cv_train_data, train_data['GROUP_ID'], cv=3, scoring='accuracy')
    print "Scores : "
    print scores.mean()
    return scores.mean()

def f(params):
    acc = hyperopt_rf_opt(params)
    return {'loss': 1-acc, 'status': STATUS_OK}

random_state = 1
n_jobs = 8
max_evals = 50

space4nb = {
    'features': hp.choice('features', range(2000, 20000)),
    'alpha': hp.uniform('alpha', 0, 1)
}


trials = Trials()
best = fmin(f, space4nb, algo=tpe.suggest, max_evals=max_evals, trials=trials)
print 'best:'
print best


cv = CountVectorizer(max_features=best['features'])
cv_train_data = cv.fit_transform(train_data['NAME'])
cv_test_data = cv.transform(test_data['NAME'])

best = MultinomialNB(alpha=best['alpha'])

print 'Score of best: '
print cross_val_score(best, cv_train_data, train_data['GROUP_ID'], cv=3, scoring='accuracy').mean()


best.fit(cv_train_data, train_data['GROUP_ID'])

predict_train = best.predict(cv_train_data)
predict_test = best.predict(cv_test_data)

test_data['GROUP_ID'] = predict_test
test_data[['id', 'GROUP_ID']].to_csv('./../../result/res_nb_test_0906.csv', sep=',', header=True, index=False)

train_data['PREDICT_GROUP_ID'] = predict_train
train_data[['id', 'GROUP_ID', 'PREDICT_GROUP_ID']].to_csv('./../../result/res_nb_train_0906.csv', sep=',', header=True, index=False)