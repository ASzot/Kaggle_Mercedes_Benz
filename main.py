import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split

train = pd.read_csv('data/train.csv')
test = pd.read_csv('data/test.csv')

for c in train.columns:
    if train[c].dtype == 'object':
        lbl = LabelEncoder()
        lbl.fit(list(train[c].values) + list(test[c].values))
        train[c] = lbl.transform(list(train[c].values))
        test[c] = lbl.transform(list(test[c].values))

from sklearn.decomposition import PCA, FastICA

usable_columns = list(set(train.columns) - set(['ID', 'y']))

y_train = train['y']
id_test = test['ID'].astype(np.int32)

train = train[usable_columns]
test = test[usable_columns]

for column in usable_columns:
    cardinality = len(np.unique(train[column]))
    if cardinality == 1:
        train.drop(column, axis=1)
        test.drop(column, axis=1)

n_comp = 200

pca = PCA(n_components=n_comp)
pca2_results_train = pca.fit_transform(train)
pca2_results_test = pca.fit_transform(test)

#print('Before ' + str(train.shape))
#print('After ' + str(pca2_results_train.shape))

#train = pca2_results_train
#test = pca2_results_test

#ica = FastICA(n_components=n_comp)
#ica2_results_train = ica.fit_transform(train)
#ica2_results_test = ica.fit_transform(test)
#
#for i in range(1, n_comp+1):
#    train['pca_' + str(i)] = pca2_results_train[:, i-1]
#    test['pca_' + str(i)] = pca2_results_test[:, i-1]
#
#    train['ica_' + str(i)] = ica2_results_train[:, i-1]
#    test['ica_' + str(i)] = ica2_results_test[:, i-1]

y_mean = np.mean(y_train)

import xgboost as xgb

from sklearn.metrics import r2_score

x_train, x_valid, y_train, y_valid = train_test_split(train, y_train,
        test_size=0.2)

d_train = xgb.DMatrix(x_train, label=y_train)
d_valid = xgb.DMatrix(x_valid, label=y_valid)

d_test = xgb.DMatrix(test)

params = {}
params['objective'] = 'reg:linear'
params['eta'] = 0.02
params['max_depth'] = 4

def xgb_r2_score(preds, dtrain):
    labels = dtrain.get_label()
    return 'r2', r2_score(labels, preds)

watchlist = [(d_train, 'train'), (d_valid, 'valid')]

clf = xgb.train(params, d_train, 1000, watchlist, early_stopping_rounds=50,
        feval=xgb_r2_score, maximize=True, verbose_eval=10)

print('Dimensionality is now ' + str(train.shape))

print(r2_score(d_valid.get_label(), clf.predict(d_valid)))

y_pred = clf.predict(d_test)
output = pd.DataFrame({'id': id_test, 'y': y_pred})
output.to_csv('data/xgboost-depth{}-pca-ica.csv'.format(params['max_depth']),
        index=False)
