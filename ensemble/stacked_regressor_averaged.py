from sklearn.base import BaseEstimator, TransformerMixin, RegressorMixin, clone
from sklearn.model_selection import KFold
import numpy as np

class StackedRegressorAveraged(BaseEstimator, RegressorMixin, TransformerMixin):
    def __init__(self, regressors, meta_regressor, n_folds=5):
        self.regressors = regressors
        self.meta_regressor = meta_regressor
        self.n_folds = n_folds


    def __fit_and_predict_model(self, clf, train, y_train, x_test, col):
        pred = None
        if isinstance(clf, dict):
            params = clf

            # This is a parameter set for some model.
            if (params['modelName'] == 'lgb':
                params.pop('modelName', None)
                num_boost_rounds = params.pop('boostRounds', None)
                if num_boost_rounds is None:
                    raise ValueError('Number of boost rounds must be set in parameters')

                train_lgb = lgb.Dataset(x_train[col], y_train)
                model = lgb.train(params, train_lgb, num_boost_rounds=num_boost_rounds)
                pred = model.predict(x_test[col])
            else:
                raise ValueError('Invalid model name in custom parameters')
        else:
            clf.fit(x_train[col], y_train)
            pred = clf.predict(x_test[col])

        return pred


    def fit(self, train, test):
        cols = list(test.columns)
        label = train['y']

        train = train.reset_index(drop=True)

        stack_data_train = []
        stack_data_test = []

        for i, clf in enumerate(self.regressors):

            kfold = KFold(n_splits=self.n_folds, shuffle=False)
            fold_count = 0
            all_ids = []
            r2_scores = []
            for train_idx, holdout_idx in kfold.split(train):
                print('Training %i fold' % (fold_count + 1))

                self.regr_[i].append(instance)

                x_train, x_test = train.iloc(train_idx,:], train.iloc(test_index,:]
                y_train, y_test = label.iloc[train_index], lable.iloc[test_index]

                pred = self.__fit_and_predict_model(clf, train, y_train, x_test, col)

                x_test['label'] = list(y_test)
                x_test['predicted'] = pred

                r2 = r2_score(y_test, pred)
                r2_scores.append(r2)

                all_ids.append(x_test['ID'])
                if i == 0:
                    final = x_test
                else:
                    final = final.append(x_test, ignore_index=True)

                fold_count += 1

            final_pred = self.__fit_and_predict(clf, train, label, x_test, col)
            final_pred = pd.DataFrame({
                'ID': test['ID'],
                'y': final_pred
                })

            print('In bag r2 score')
            print(r2_score(label, model.predict(train[col])))
            print('Out of bag r2 score')
            print(np.mean(r2_score))

            stack_data_train.append(final)
            stack_data_test.append(final_pred)







    def predict(self, X):
        meta_features = np.column_stack([
            np.column_stack([r.predict(X) for r in regrs]).mean(axis=1)
            for regrs in self.regr_
        ])
        return self.meta_regr_.predict(meta_features)


