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
    MODEL_PATH = 'data/model/nn.h5'

    def __init__(self, input_dims):
        self.input_dims = input_dims

    def __create_estimator(self, x_train, y_train):
        x_train, x_val, y_train, y_val = train_test_split(x_train, y_train,
            test_size=0.2)

        def build_model():
            model = Sequential()

            units_count = self.input_dims

            model.add(Dense(units_count, input_dim=self.input_dims, activation='relu',
                kernel_constraint=maxnorm(3)))
            model.add(BatchNormalization())
            model.add(Dropout(0.2))
            model.add(Activation('linear'))

            # Hidden layer 1
            model.add(Dense(units_count, activation='relu', kernel_constraint=maxnorm(3)))
            model.add(BatchNormalization())
            model.add(Dropout(0.3))
            model.add(Activation('linear'))

            # Hidden layer 2
            model.add(Dense(units_count, activation='relu', kernel_constraint=maxnorm(3)))
            model.add(BatchNormalization())
            model.add(Dropout(0.3))
            model.add(Activation('linear'))

            # Hidden layer 3
            model.add(Dense(units_count, activation='relu', kernel_constraint=maxnorm(3)))
            model.add(BatchNormalization())
            model.add(Dropout(0.3))
            model.add(Activation('linear'))

            # Hidden layer 4
            model.add(Dense(units_count, activation='relu', kernel_constraint=maxnorm(3)))
            model.add(BatchNormalization())
            model.add(Dropout(0.3))
            model.add(Activation('linear'))

            model.add(Dense(1))

            model.compile(loss='mean_squared_error',
                    optimizer='rmsprop',
                    metrics=[r2_keras, 'mse'])

            print('Neural network model created')

            return model

        self.estimator = KerasRegressor(
                build_fn = build_model,
                nb_epoch = 300,
                batch_size = 30,
                verbose=0
                )

        callbacks = [
                EarlyStopping(
                    monitor='val_r2_keras',
                    patience=20,
                    mode='max',
                    verbose=False),
                ModelCheckpoint(
                    BasicNeuralNetwork.MODEL_PATH,
                    monitor='val_r2_keras',
                    save_best_only=True,
                    mode='max',
                    verbose=False)
                ]

        print('Preparing to fit neural network')
        history = self.estimator.fit(x_train, y_train, epochs=500, validation_data=(x_val,
            y_val), verbose = False, callbacks=callbacks, shuffle=True)
        print('Model fit')


    def __load_estimator(self):
        self.estimator = load_model(BasicNeuralNetwork.MODEL_PATH,
                custom_objects={'r2_keras': r2_keras})

    def fit_try_load(self, x_train, y_train):
        try:
            print('Attempting to load model')
            self.__load_estimator()
            print('Model loaded')
        except Exception as e:
            print('Load failed. Creating model')
            self.__create_estimator(x_train, y_train)
        return self


    def fit(self, x_train, y_train):
        self.__create_estimator(x_train, y_train)
        return self


    def predict(self, X):
        res = self.estimator.predict(X).ravel()
        print()
        return res

