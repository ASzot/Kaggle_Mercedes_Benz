from sklearn.base import BaseEstimator, TransformerMixin, RegressorMixin, clone
import numpy as np

class RegressorAveraged(BaseEstimator, RegressorMixin, TransformerMixin):
    def __init__(self, regressors):
        self.regressors = regressors

    def __clone_regressors(self):
        for regressor in self.regressors:
            try:
                ret_obj = clone(regressor)
            except:
                ret_obj = regressor

            yield ret_obj


    def fit(self, X, y):
        self.regr_ = list(self.__clone_regressors())

        # Train base models
        for regr in self.regr_:
            regr.fit(X, y)

        return self

    def predict(self, X):
        predictions = np.column_stack([
            regr.predict(X) for regr in self.regr_
        ])
        return np.mean(predictions, axis=1)



