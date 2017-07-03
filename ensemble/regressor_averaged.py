from sklearn.base import BaseEstimator, TransformerMixin, RegressorMixin, clone

class RegressorAveraged(BaseEstimator, RegressorMixin, TransformerMixin):
    def __init__(self, regressors):
        self.regressors = regressors

    def fit(self, X, y):
        self.regr_ = [clone(x) for x in self.regressors]

        # Train base models
        for regr in self.regr_:
            regr.fit(X, y)

        return self

    def predict(self, X):
        predictions = np.column_stack([
            regr.predict(X) for regr in self.regr_
        ])
        return np.mean(predictions, axis=1)



