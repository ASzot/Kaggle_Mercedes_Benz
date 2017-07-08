import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor, ExtraTreesRegressor
from sklearn.linear_model import ElasticNet, Lasso
from sklearn.feature_selection import SelectFromModel
from sklearn.svm import SVR
from sklearn.model_selection import cross_val_score
from sklearn.pipeline import make_pipeline
from sklearn.decomposition import PCA
from sklearn.preprocessing import RobustScaler
import xgboost as xgb
from ensemble.regressor_averaged import RegressorAveraged
from ensemble.stacked_regressor_averaged import StackedRegressorAveraged
from ensemble.stacked_regressor_retrained import StackedRegressorRetrained
from model.nn import BasicNeuralNetwork
from preprocessing.preprocessor import Preprocessor
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
from sklearn.decomposition import PCA, FastICA

BASE_DIR = 'data/'

def test_model(model, x_train, y_train, x_test, id_test):
    #x_train, x_val, y_train, y_val = train_test_split(x_train, y_train,
    #        test_size=0.2)

    model = model.fit(x_train, y_train)

    #pred_y_val = model.predict(x_val)
    #print('Validation score: %.7f' % r2_score(pred_y_val, y_val))

    y_pred = model.predict(x_test)

    output = pd.DataFrame({'id': id_test, 'y': y_pred})
    output.to_csv('data/preds/preds2.csv', index=False)
    print('Saved outputs to file')


#########################
# Preprocess data
#########################

train = pd.read_csv(BASE_DIR + 'train.csv')
test = pd.read_csv(BASE_DIR + 'test.csv')

preprocessor = Preprocessor(magicFeature=True)
train_p, test_p = preprocessor.transform(train, test)

col = list(test_p.columns)

gb1 = GradientBoostingRegressor(n_estimators=1000, max_features=0.95,
        learning_rate=0.005, max_depth=4)
gb1_train, gb1_test =

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

n_comp = 12

#pca = PCA(n_components=n_comp)
#pca_train = pca.fit_transform(train)
#pca_test = pca.transform(test)
#
#pca_train_df = pd.DataFrame()
#pca_test_df = pd.DataFrame()
#
#for i in range(n_comp):
#    pca_train_df['pca_' + str(i)] = pca_train[:,i-1]
#    pca_test_df['pca_' + str(i)] = pca_test[:,i-1]

train = train.values
test = test.values

#train = pca_train_df.values
#test = pca_test_df.values

print('data has a shape of ' + str(train.shape))

#########################
# Create models
#########################

en = make_pipeline(RobustScaler(), SelectFromModel(Lasso(alpha=0.03)),
        ElasticNet(alpha=0.001, l1_ratio=0.1))

rf = RandomForestRegressor(n_estimators=250, n_jobs=4, min_samples_split=25,
        min_samples_leaf=25, max_depth=3)

et = ExtraTreesRegressor(n_estimators=100, n_jobs=4, min_samples_split=25,
        min_samples_leaf=35, max_features=150)

xgbm = xgb.sklearn.XGBRegressor(max_depth=4, learning_rate=0.005, subsample=0.9,
        base_score=y_mean, objective='reg:linear', n_estimators=1000)

nn = BasicNeuralNetwork(train.shape[1])

#########################
# Create Ensembles
#########################

stack_avg = StackedRegressorAveraged((en, rf, et),
        ElasticNet(l1_ratio=0.1, alpha=1.4))

stack_with_feats = StackedRegressorRetrained((nn, en, rf, et), xgbm,
        use_features_in_secondary=True)

stack_retrain = StackedRegressorRetrained((nn, en, rf, et),
        ElasticNet(l1_ratio=0.1, alpha=1.4))

averaged = RegressorAveraged((nn, rf, en, xgbm,))

#########################
# Evaluate Models
#########################

use_model = stack_with_feats

print('Evaluating validation score')
results = cross_val_score(use_model, train, y_train, cv = 4,
        scoring='r2', n_jobs=1, verbose=1)
print("Validation score: %.4f (%.4f)" % (results.mean(), results.std()))

test_model(use_model, train, y_train, test, id_test)

#results = cross_val_score(stack_avg, train.values, y_train, cv=5, scoring='r2')
#print("Stacking (averaged) score: %.4f (%.4f)" % (results.mean(), results.std()))
