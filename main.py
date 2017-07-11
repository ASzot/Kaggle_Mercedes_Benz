import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor, ExtraTreesRegressor
from sklearn.ensemble import GradientBoostingRegressor
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


#########################
# Preprocess data
#########################

train = pd.read_csv(BASE_DIR + 'train.csv')
test = pd.read_csv(BASE_DIR + 'test.csv')

preprocessor = Preprocessor(magicFeature=True)
train_p, test_p = preprocessor.transform(train, test)


#########################
# Create models
#########################

gb = GradientBoostingRegressor(n_estimators=1000, max_features=0.95,
        learning_rate=0.005, max_depth=4)
las = Lasso(alpha=5)
lgb = {
        'objective': 'regression',
        'metric': 'rmse',
        'boosting': 'gbdt',
        'learning_rate': 0.0045 , #small learn rate, large number of iterations
        'verbose': 0,
        'num_iterations': 500,
        'bagging_fraction': 0.95,
        'bagging_freq': 1,
        'bagging_seed': 42,
        'feature_fraction': 0.95,
        'feature_fraction_seed': 42,
        'max_bin': 100,
        'max_depth': 3,
        'num_rounds': 800
        }

regressors = [gb, las, lgb]

meta_regressor = {
    'eta': 0.005,
    'max_depth': 2,
    'objective': 'reg:linear',
    'eval_metric': 'rmse',
    'base_score': StackedRegressorAveraged.FILL_AVG, # base prediction = mean(target)
    'silent': 1
}

col = list(test_p.columns)
stacked_regressor = StackedRegressorAveraged(regressors, meta_regressor, col)

xgb = {
    'n_trees': 520,
    'eta': 0.0045,
    'max_depth': 4,
    'subsample': 0.93,
    'objective': 'reg:linear',
    'eval_metric': 'rmse',
    'base_score': StackedRegressorAveraged.FILL_AVG, # base prediction = mean(target)
    'silent': True,
    'seed': 42,
}


avg_regressor = RegressorAveraged([stacked_regressor, xgb], col, pred_weights = [0.25, 0.75])

avg_regressor = avg_regressor.fit(train_p, train_p['y'])

avg_regressor.predict(test_p)
