from keras.models import Sequential, load_model
from keras.layers import Dense, Dropout, BatchNormalization, Activation
from keras.wrappers.scikit_learn import KerasRegressor
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras.layers.recurrent import LSTM

from keras import backend as K
from keras.constraints import maxnorm
from keras.optimizers import SGD, Adam

from sklearn.model_selection import cross_val_score, KFold, train_test_split, GridSearchCV
from sklearn.feature_selection import SelectFromModel
from sklearn.metrics import r2_score, mean_squared_error

from sklearn.feature_selection import f_regression, mutual_info_regression, VarianceThreshold

import os

def r2_keras(y_true, y_pred):
    ss_res = K.sum(K.square(y_true - y_pred))
    ss_tot = K.sum(K.square(y_true - K.mean(y_true)))
    return (1 - ss_res / (ss_tot + K.epsilon()))


class BasicNeuralNetwork(object):
    def __init__(self):
        self.input_dims = train.shape[1] - 1


    def fit(self, x_train, y_train):
        x_train, x_val, y_train, y_val = train_test_split(x_train, y_train,
            test_size=0.2)

        def build_model():
            model = Sequential()

            model.add(Dense(input_dims, input_dim=input_dims, activation='relu',
                kernel_constraint=maxnorm(3)))
            model.add(BatchNormalization())
            model.add(Dropout(0.2))
            model.add(Activation('linear'))

            # Hidden layer 1
            model.add(Dense(input_dims, activation='relu', kernel_constraint=maxnorm(3)))
            model.add(BatchNormalization())
            model.add(Dropout(0.3))
            model.add(Activation('linear'))

            # Hidden layer 2
            model.add(Dense(input_dims, activation='relu', kernel_constraint=maxnorm(3)))
            model.add(BatchNormalization())
            model.add(Dropout(0.3))
            model.add(Activation('linear'))

            # Hidden layer 3
            model.add(Dense(input_dims, activation='relu', kernel_constraint=maxnorm(3)))
            model.add(BatchNormalization())
            model.add(Dropout(0.3))
            model.add(Activation('linear'))

            # Hidden layer 4
            model.add(Dense(input_dims, activation='relu', kernel_constraint=maxnorm(3)))
            model.add(BatchNormalization())
            model.add(Dropout(0.3))
            model.add(Activation('linear'))

            model.add(Dense(1))

            model.compile(loss='mean_squared_error',
                    optimizer='rmsprop',
                    metrics=[r2_keras, 'mse'])

            print(model.summary())
            return model

        self.estimator = KerasRegressor(
                build_fn = build_model,
                nb_epoch = 300,
                batch_size = 30,
                verbose=1
                )

        history = self.estimator.fit(x_train, y_train, epoochs=500, validation_data=(x_val,
            y_val), verbose=2, callbacks=callbacks, shuffle=True)

        return self


    def predict(self, X):
        res = self.estimator.predict(X).ravel()
        return res

