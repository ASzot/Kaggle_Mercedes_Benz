import xgboost as xgb
import lightgbm as lgb
import numpy as np
from sklearn.base import clone
from sklearn.base import BaseEstimator, TransformerMixin, RegressorMixin

class EnsembleRegressor(BaseEstimator, RegressorMixin, TransformerMixin):
    FILL_AVG = "$FILLAVG$"

    def fit_model(self, model, x_train, y_train, col):
        fn_transform_x_test = lambda x: x
        print('Fitting model')
        clf = clone(model)

        if isinstance(clf, dict):
            params = clf
            # This is a parameter set for some model.
            if params['modelName'] == 'lgb':
                print('Model is lgb')
                params.pop('modelName', None)
                num_boost_rounds = params.pop('boostRounds', None)
                if num_boost_rounds is None:
                    raise ValueError('Number of boost rounds must be set in parameters')

                train_lgb = lgb.Dataset(x_train[col], y_train)
                model = lgb.train(params, train_lgb, num_boost_round=num_boost_rounds)
            elif params['modelName'] == 'xgb':
                print('Model is xgb')
                params.pop('modelName', None)
                num_boost_rounsd = params.pop('boostRounds', None)

                for param_key in params:
                    if params[param_key] == EnsembleRegressor.FILL_AVG:
                        params[param_key] = np.mean(y_train)

                dtrain = xgb.DMatrix(x_train[col], y_train)
                model = xgb.train(params, dtrain, num_boost_round = 900)
                fn_transform_x_test = lambda x: xgb.DMatrix(x)
            else:
                raise ValueError('Invalid model name in custom parameters')
        else:
            print('Basic sklearn model')
            clf.fit(x_train[col], y_train)
            model = clf

        return model, fn_transform_x_test


    def fit_and_predict_model(self, clf, train, y_train, x_test, col):
        model, fn_transform_x_test = self.fit_model(clf, train, y_train, col)

        print('Making predictions using trained model')
        use_x_test = fn_transform_x_test(x_test[col])

        pred = model.predict(use_x_test)

        return pred


