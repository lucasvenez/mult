# Copyright 2020 The MuLT Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

from skopt import gp_minimize
from skopt.space import Real, Integer, Categorical
from skopt.utils import use_named_args
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import log_loss

from sklearn.svm import SVC
import numpy as np

import warnings
warnings.filterwarnings("ignore")


def sigmoid(x, alpha=1.):
    return 1. / (1. + np.exp(-alpha * x))


class SVMOptimizer(object):

    def __init__(self,
                 n_folds=3, n_calls=50, shuffle=True, early_stopping_rounds=None,
                 fixed_parameters={}, random_state=None, verbose=-1, n_jobs=-1):
        self.n_calls = n_calls
        self.n_folds = n_folds
        self.random_state = random_state
        self.shuffle = shuffle
        self.verbose = verbose
        self.n_jobs = n_jobs
        self.optimization_details = {}
        self.early_stopping_rounds = early_stopping_rounds
        self.fixed_parameters = fixed_parameters

        self.iterations = []

    def execute_optimization(self, objective, space):
        params = gp_minimize(objective, space, n_calls=self.n_calls, random_state=self.random_state,
                             verbose=(self.verbose >= 0), n_jobs=self.n_jobs).x

        return {space[i].name: params[i] for i in range(len(space))}

    def optimize(self, x, y):

        space = [
            Real(1e-6, 1e+6, prior='log-uniform', name='C'),
            Real(1e-6, 1e+1, prior='log-uniform', name='gamma'),
            Integer(1, 8, name='degree'),
            Categorical(['linear', 'poly', 'rbf'], name='kernel')]

        @use_named_args(space)
        def objective(C, gamma, degree, kernel):
            try:
                scores = []

                params = {
                    'C': C,
                    'gamma': gamma,
                    'degree': degree,
                    'kernel': kernel,
                    'random_state': self.random_state}

                if isinstance(self.fixed_parameters, dict):
                    params.update(self.fixed_parameters)

                skf = StratifiedKFold(
                    self.n_folds, shuffle=self.shuffle, random_state=self.random_state)

                for train_index, valid_index in skf.split(x, y):

                    try:
                        x_train, y_train = x[train_index, :], y[train_index, 0]

                    except IndexError:
                        x_train, y_train = x[train_index, :], y[train_index].reshape(-1,)

                    try:
                        x_valid, y_valid = x[valid_index, :], y[valid_index, 0]

                    except IndexError:
                        x_valid, y_valid = x[valid_index, :], y[valid_index].reshape(-1,)

                    svm = SVC(**params)

                    svm.fit(x_train, y_train)

                    y_hat = sigmoid(svm.predict(x_valid))

                    scores.append(log_loss(y_valid, y_hat))

                result = np.mean(scores)

                self.iterations.append((params, result))

                return result

            except ValueError:

                return np.inf

        return self.execute_optimization(objective, space)
