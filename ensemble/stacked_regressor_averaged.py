from sklearn.base import BaseEstimator, TransformerMixin, RegressorMixin, clone
from sklearn.model_selection import KFold
import numpy as np
import pandas as pd
from sklearn.metrics import r2_score
from ensemble.ensemble_regressor import EnsembleRegressor

class StackedRegressorAveraged(EnsembleRegressor):

    def __init__(self, regressors, meta_regressor, base_cols, n_folds=5):
        self.regressors = regressors
        self.meta_regressor = meta_regressor
        self.n_folds = n_folds
        # lambda meta regressor uses to transform the test data.
        self.meta_test_transform = None
        self.base_cols = base_cols


    def fit(self, train, label):
        print('Fitting stacked regressor averaged')
        cols = self.base_cols

        train = train.reset_index(drop=True)

        all_stack_data_train = []
        all_stack_data_test = []
        self.all_transform_x = []

        for i, clf in enumerate(self.regressors):
            print('Processing regressor %i' % i)

            kfold = KFold(n_splits=self.n_folds, shuffle=False)
            fold_count = 0
            all_ids = []
            r2_scores = []
            for train_idx, test_idx in kfold.split(train):
                print('Training %i fold' % (fold_count + 1))

                x_train, x_test = train.iloc[train_idx,:], train.iloc[test_idx,:]
                y_train, y_test = label.iloc[train_idx], label.iloc[test_idx]

                pred = self.fit_and_predict_model(clf, train, y_train,
                        x_test, cols)

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

            final_pred = self.fit_and_predict_model(clf, train, label,
                    x_test, cols)

            self.regressors[i], add_transform_x = self.fit_model(clf, train,
                    label, cols)
            self.all_transform_x.append(add_transform_x)

            print('In bag r2 score')
            use_pred_train_x = train[cols]
            print(r2_score(label, self.regressors[i].predict(use_pred_train_x)))
            print('Out of bag r2 score')
            print(np.mean(r2_score))

            all_stack_data_train.append(final)
            all_stack_data_test.append(final_pred)

        print('Getting stacked data')
        stack_train = all_stack_data_train[0][['label']]
        stack_cols = []
        for i, stack_data_train in enumerate(all_stack_data_train):
            pred_iden = 'pred%i' % i
            stack_cols.append(pred_iden)
            stack_train[pred_iden] = stack_data_train['predicted']

        print('Processing meta regressor')
        self.meta_regressor, self.meta_test_transform = self.fit_model(self.meta_regressor, stack_train,
                stack_train['label'], stack_cols)


    def predict(self, X):
        cols = list(X.columns)
        all_stack_data_test = []
        predictors = zip(self.regressors, self.all_transform_x)
        i = 0
        for clf, transform_x in predictors:
            print('Outputting prediction for regressor %i' % i)
            use_test_x = transform_x(X[cols])
            pred = clf.predict(use_test_x)
            final_pred = pd.DataFrame({
                'ID': X['ID'],
                'y': pred
                })
            all_stack_data_test.append(final_pred)
            i += 1

        stack_test = all_stack_data_test[0]['ID']
        for i, stack_data_test in enumerate(all_stack_data_test):
            pred_iden = 'pred%i' % i
            stack_test[pred_iden] = stack_data_test['y']

        del stack_test['ID']

        stack_cols = list(stack_test.columns)

        print('Outputting prediction for meta regressor')
        use_x = self.meta_test_transform(X[stack_cols])
        return self.meta_regressor.predict(use_x)


