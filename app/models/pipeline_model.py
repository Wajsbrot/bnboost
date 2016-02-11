#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Created on Sat Jan 30 15:47:09 2016

@author: Nicolas Thiebaut
@email: nkthiebaut@gmail.com
@company: Quantmetry
"""

import cPickle
import numpy as np
import pandas as pd

from sklearn.preprocessing import LabelEncoder
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.cross_validation import KFold, cross_val_score
from sklearn.grid_search import GridSearchCV
from sklearn.metrics import make_scorer
from xgboost.sklearn import XGBClassifier
import xgboost as xgb


from categorical_features_transformer import RareItemsGrouper
from ndcg_score import *



df_train = pd.read_csv('../data/train.csv', index_col='id')
# cPickle.load(open('train.pkl', 'rb'))
le = LabelEncoder()
y = df_train['country_destination'].values
y = le.fit_transform(y)

ids = df_train.index  # ['id']
df_train.drop(['country_destination'], axis=1, inplace=True)
X = df_train.values
features = df_train.columns
del df_train

#dtrain = xgb.DMatrix(data, label=label, missing = -999.0)

rf = RandomForestClassifier(n_jobs=4)
xgb = XGBClassifier(objective='multi:softprob')
#XGBClassifier(max_depth=6, learning_rate=0.3, n_estimators=25,
#                    objective='multi:softprob', subsample=0.5, colsample_bytree=0.5, seed=0)

"""

grouper = RareItemsGrouper()
pipe = Pipeline([
    ('grouper', grouper),
    ('rf', rf)
])

GS_space = dict(cft__max_categorical=[5, 10],
                rf__n_estimators=[100, 200, 300],
                rf__max_depth=[5, 10, 15])
"""

param_grid_rf = {"max_depth": [15],
                 "bootstrap": [True],
                 "n_estimators": [300],
                 "criterion": ["gini"]}

param_grid_xgb = {'max_depth': [3, 6],
                  'learning_rate': [0.1, 0.3],
                  'n_estimators': [25]}

ndcg_scorer = make_scorer(ndcg_score, needs_proba=True, k=5)

kf = KFold(len(X), n_folds=2)
# score = cross_val_score(rf, X, y, cv=kf)#, scoring=ndcg_scorer)

estimator = GridSearchCV(xgb, param_grid_xgb, verbose=2, n_jobs=4, cv=kf)
estimator.fit(X, y)
print estimator.grid_scores_

df_test = pd.read_csv('../data/test.csv', index_col='id')
# cPickle.load(open('test.pkl', 'rb'))
ids_test = df_test.index  # ['id']
X_test = df_test.values  # .drop(['id'], axis=1).values

y_pred = estimator.predict_proba(X_test)
# y_pred = le.inverse_transform(y_pred)


# Taking the 5 classes with highest probabilities
ids = []  # list of ids
cts = []  # list of countries
for i in range(len(ids_test)):
    idx = ids_test[i]
    ids += [idx] * 5
    cts += le.inverse_transform(np.argsort(y_pred[i])[::-1])[:5].tolist()


sub = pd.DataFrame(np.column_stack((ids, cts)), columns=['id', 'country'])
sub.to_csv('../out/submission.csv', index=False)

# Feature importances
fi = pd.DataFrame()
fi['feature'] = features
fi['importance'] = estimator.best_estimator_.feature_importances_
fi.to_csv('../out/feature_importances.csv', index=False)
