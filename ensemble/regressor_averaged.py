import numpy as np
from functools import reduce
from ensemble.ensemble_regressor import EnsembleRegressor

class RegressorAveraged(EnsembleRegressor):
    def __init__(self, regressors, cols, pred_weights = None):
        self.regressors = regressors
        self.pred_weights = pred_weights
        self.use_cols = cols
        self.all_fn_x = []


    def fit(self, X, y):
        print('Fitting regressor averaged')
        self.all_fn_x = []

        def train_model_mapper(clf):
            print('Fitting regressor')
            model, fn_x = self.fit_model(clf, X, y, self.use_cols)
            self.all_fn_x.append(fn_x)
            return model


        # Train base models
        self.regressors = map(train_model_mapper, self.regressors)
        return self


    def predict(self, X):
        if self.pred_weights is None:
            print('Computing predictions to be averaged')
            # Transform the input by the model's input transformer and make a
            # prediction.
            preds = list(map(lambda x: x[0].predict(x[1](X)), zip(self.regressors,
                self.all_fn_x)))

            print('Weighting predictions')
            weighted_pred = reduce(lambda x, y: x + (y[0] * y[1]), zip(self.pred_weights, preds))

            return weighted_pred
        else:
            print('Computing predictions to be averaged')
            predictions = np.column_stack([
                regr.predict(fn_x(X)) for regr, fn_x in zip(self.regressors, self.all_fn_x)
            ])

            print('Averaging predictions')

            return np.mean(predictions, axis=1)
