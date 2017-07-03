from sklearn.base import BaseEstimator, TransformerMixin, RegressorMixin, clone

class StackedRegressorAveraged(BaseEstimator, RegressorMixin, TransformerMixin):
    def __init__(self, regressors, meta_regressor, n_folds=5):
        self.regressors = regressors
        self.meta_regressor = meta_regressor
        self.n_folds = n_folds

    def fit(self, X, y):
        self.regr_ = [list() for x in self.regressors]
        self.meta_regr_ = clone(self.meta_regressor)

        kfold = KFold(n_splits=self.n_folds, shuffle=True)

        out_of_fold_predictions = np.zeros((X.shape[0], len(self.regressors)))

        for i, clf in enumerate(self.regressors):
            for train_idx, holdout_idx in kfold.split(X, y):
                instance = clone(clf)
                self.regr_[i].append(instance)

                instance.fit(X[train_idx], y[train_idx])
                y_pred = instance.predict(X[holdout_idx])
                out_of_fold_predictions[holdout_idx, i] = y_pred

        self.meta_regr_.fit(out_of_fold_predictions, y)

        return self

    def predict(self, X):
        meta_features = np.column_stack([
            np.column_stack([r.predict(X) for r in regrs]).mean(axis=1)
            for regrs in self.regr_
        ])
        return self.meta_regr_.predict(meta_features)


