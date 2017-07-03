import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor, ExtraTreesRegressor
from sklearn.linear_model import ElasticNet, Lasso
from sklearn.feature_selection import SelectFromModel
from sklearn.svm import SVR
from sklearn.model_selection import KFold, cross_val_score
from sklearn.pipeline import make_pipeline
from sklearn.decomposition import PCA
from sklearn.preprocessing import RobustScaler
import xgboost as xgb
from sklearn.feature_extraction import DictVectorizer

BASE_DIR = 'data/'

train = pd.read_csv(BASE_DIR + 'train.csv')
test = pd.read_csv(BASE_DIR + 'test.csv')

y_train = train['y'].values
y_mean = np.mean(y_train)

id_test = test['ID']

prev_len = len(train.columns)

for k in train.keys():
    if len(train[k].unique()) == 1:
        train.pop(k)
        test.pop(k)

print('%i columns were dropped' % (prev_len - len(train.columns)))

num_train = len(train)
df_all = pd.concat([train, test])
df_all.drop(['ID', 'y'], axis=1, inplace=True)

df_all = pd.get_dummies(df_all, drop_first=True)

train = df_all[:num_train]
test = df_all[num_train:]

def test_model(model, x_train, y_train, x_test, id_test):
    y_pred = model.fit(x_train, y_train).predict(x_test)
    output = pd.DataFrame({'id': id_test, 'y': y_pred})
    output.to_csv('data/preds/preds0.csv', index=False)
    print('Saved outputs to file')


en = make_pipeline(RobustScaler(), SelectFromModel(Lasso(alpha=0.03)),
        ElasticNet(alpha=0.001, l1_ratio=0.1))

rf = RandomForestRegressor(n_estimators=250, n_jobs=4, min_samples_split=25,
        min_samples_leaf=25, max_depth=3)

et = ExtraTreesRegressor(n_estimators=100, n_jobs=4, min_samples_split=25,
        min_samples_leaf=35, max_features=150)

xgbm = xgb.sklearn.XGBRegressor(max_depth=4, learning_rate=0.005, subsample=0.9,
        base_score=y_mean, objective='reg:linear', n_estimators=1000)

stack_avg = StackingCVRegressorAveraged((en, rf, et),
        ElasticNet(l1_ratio=0.1, alpha=1.4))

stack_with_feats = StackingCVRegressorRetrained((en, rf, et), xgbm,
        use_features_in_secondary=True)

stack_retrain = StackingCVRegressorRetrained((en, rf, et),
        ElasticNet(l1_ratio=0.1, alpha=1.4))

averaged = AveragingRegressor((en, rf, et, xgbm))

#results = cross_val_score(en, train.values, y_train, cv=5, scoring='r2')
#print("ElasticNet score: %.4f (%.4f)" % (results.mean(), results.std()))
#
#results = cross_val_score(rf, train.values, y_train, cv=5, scoring='r2')
#print("RandomForest score: %.4f (%.4f)" % (results.mean(), results.std()))
#
#results = cross_val_score(et, train.values, y_train, cv=5, scoring='r2')
#print("ExtraTrees score: %.4f (%.4f)" % (results.mean(), results.std()))
#
#results = cross_val_score(xgbm, train.values, y_train, cv=5, scoring='r2')
#print("XGBoost score: %.4f (%.4f)" % (results.mean(), results.std()))
#
#results = cross_val_score(averaged, train.values, y_train, cv=5, scoring='r2')
#print("Averaged base models score: %.4f (%.4f)" % (results.mean(), results.std()))
#
#results = cross_val_score(stack_with_feats, train.values, y_train, cv=5, scoring='r2')
#print("Stacking (with primary feats) score: %.4f (%.4f)" % (results.mean(), results.std()))

results = cross_val_score(averaged, train.values, y_train, cv=5, scoring='r2')
print("Stacking (retrained) score: %.4f (%.4f)" % (results.mean(), results.std()))


test_model(stack_retrain, train.values, y_train, test, id_test)

#results = cross_val_score(stack_avg, train.values, y_train, cv=5, scoring='r2')
#print("Stacking (averaged) score: %.4f (%.4f)" % (results.mean(), results.std()))
