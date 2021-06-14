from optimization import KNNOptimizer, MLPOptimizer, LightGBMOptimizer, SVMOptimizer

from sklearn.model_selection import train_test_split

from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC

from lightgbm import LGBMModel

import numpy as np


class Stack(object):

    def __init__(self, random_state=None, test_size=0.2, verbose=None,
                 optimization_n_call=50,
                 optimization_n_folds=2,
                 optimization_early_stopping_rounds=1,
                 optimization_shuffle=True):

        self.opt = LightGBMOptimizer(
            n_folds=optimization_n_folds,
            n_calls=optimization_n_call,
            early_stopping_rounds=optimization_early_stopping_rounds,
            shuffle=optimization_shuffle,
            n_jobs=-1)

        self.lgb_opt = LightGBMOptimizer(
            n_folds=optimization_n_folds,
            n_calls=optimization_n_call,
            early_stopping_rounds=optimization_early_stopping_rounds,
            shuffle=optimization_shuffle,
            n_jobs=-1)

        self.mlp_opt = MLPOptimizer(
            n_folds=optimization_n_folds,
            n_calls=optimization_n_call,
            shuffle=optimization_shuffle,
            n_jobs=-1)

        self.knn_opt = KNNOptimizer(
            n_folds=optimization_n_folds,
            n_calls=optimization_n_call,
            shuffle=optimization_shuffle,
            n_jobs=-1)

        self.svm_opt = SVMOptimizer(
            n_folds=optimization_n_folds,
            n_calls=optimization_n_call,
            early_stopping_rounds=optimization_early_stopping_rounds,
            shuffle=optimization_shuffle,
            n_jobs=-1)

        self.model = None

        self.lgb_model = None
        self.mlp_model = None
        self.knn_model = None
        self.svm_model = None

        self.random_state = random_state
        self.test_size = test_size
        self.verbose = verbose

    def stack_predict(self, x):

        lgb_y_hat = self.lgb_model.predict(x, num_iteration=self.lgb_model.best_iteration_)
        print(lgb_y_hat.shape)

        mlp_y_hat = self.mlp_model.predict_proba(x)[:, -1]
        print(mlp_y_hat.shape)

        knn_y_hat = self.knn_model.predict_proba(x)[:, -1]
        print(knn_y_hat.shape)

        svm_y_hat = self.svm_model.predict_proba(x)[:, -1]
        print(svm_y_hat.shape)

        return np.array([lgb_y_hat, mlp_y_hat, knn_y_hat, svm_y_hat]).T

    def fit(self, x, y, early_stopping_rounds=None):

        self.fit_lightgbm(x, y, early_stopping_rounds)
        self.fit_knn(x, y)
        self.fit_mlp(x, y, early_stopping_rounds)
        self.fit_svm(x, y)

        x_stack = self.stack_predict(x)

        print('fit stack')

        optimized_params = self.opt.optimize(x, y)
        optimized_params['objective'] = 'binary'

        self.model = LGBMModel(**optimized_params)
        self.model.fit(x_stack, y)

        if early_stopping_rounds is not None and early_stopping_rounds > 0:

            x_train, x_valid, y_train, y_valid = train_test_split(
                x, y, stratify=y, shuffle=True, test_size=self.test_size, random_state=self.random_state)

            self.lgb_model.fit(x_train, y_train, eval_set=[(x_valid, y_valid)], verbose=self.verbose)

        else:
            self.lgb_model.fit(x, y)

    def fit_lightgbm(self, x, y, early_stopping_rounds):

        print('fit lightgbm')

        optimized_params = self.lgb_opt.optimize(x, y)
        optimized_params['objective'] = 'binary'

        optimized_params['random_state'] = self.random_state
        optimized_params['n_jobs'] = -1

        self.lgb_model = LGBMModel(**optimized_params)
        self.lgb_model.fit(x, y)

        if early_stopping_rounds is not None and early_stopping_rounds > 0:

            x_train, x_valid, y_train, y_valid = train_test_split(
                x, y, stratify=y, shuffle=True, test_size=self.test_size, random_state=self.random_state)

            self.lgb_model.fit(x_train, y_train, eval_set=[(x_valid, y_valid)], verbose=self.verbose)

        else:
            self.lgb_model.fit(x, y)

    def fit_svm(self, x, y):

        print('fit svm')

        optimized_params = self.svm_opt.optimize(x, y)

        optimized_params['random_state'] = self.random_state

        self.svm_model = SVC(**optimized_params, probability=True)

        self.svm_model.fit(x, y)

    def fit_mlp(self, x, y, early_stopping_rounds):

        print('fit mlp')

        optimized_params = self.mlp_opt.optimize(x, y)

        optimized_params['random_state'] = self.random_state

        esr = early_stopping_rounds is not None and early_stopping_rounds > 0

        self.mlp_model = MLPClassifier(**optimized_params,
                                       early_stopping=esr,
                                       validation_fraction=self.test_size)

        self.mlp_model.fit(x, y)

    def fit_knn(self, x, y):

        print('fit knn')

        optimized_params = self.knn_opt.optimize(x, y)
        optimized_params['n_jobs'] = -1

        self.knn_model = KNeighborsClassifier(**optimized_params)

        self.knn_model.fit(x, y)

    def predict(self, x):

        x_stack = self.stack_predict(x)

        return self.model.predict(x_stack, num_iteration=self.model.best_iteration_)
