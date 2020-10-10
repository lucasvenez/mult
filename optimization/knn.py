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
from skopt.space import Integer, Categorical
from skopt.utils import use_named_args

from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import log_loss

from sklearn.neighbors import KNeighborsClassifier

import numpy as np

import warnings
warnings.filterwarnings("ignore")


class KNNOptimizer(object):

    def __init__(self,
                 n_folds=3, n_calls=50, shuffle=True,
                 fixed_parameters={}, random_state=None, verbose=-1, n_jobs=-1):

        self.n_calls = n_calls
        self.n_folds = n_folds
        self.random_state = random_state
        self.shuffle = shuffle
        self.verbose = verbose
        self.n_jobs = n_jobs
        self.optimization_details = {}
        self.fixed_parameters = fixed_parameters

        self.iterations = []

    def execute_optimization(self, objective, space):

        params = gp_minimize(objective, space, n_calls=self.n_calls, random_state=self.random_state,
                             verbose=(self.verbose >= 0), n_jobs=-1).x

        return {space[i].name: params[i] for i in range(len(space))}

    def optimize(self, x, y):

        self.iterations = []

        space = [
            Integer(1, 30, name='n_neighbors'),
            Categorical(['uniform', 'distance'], name='weights'),
            Integer(1, 30, name='leaf_size'),
            Integer(1, 5, name='p')
        ]

        @use_named_args(space)
        def objective(
            n_neighbors,
            weights,
            leaf_size,
            p
        ):
            try:
                scores = []

                params = {
                    'n_neighbors': int(n_neighbors),
                    'weights': weights,
                    'leaf_size': int(leaf_size),
                    'p': int(p),

                    'n_jobs': self.n_jobs}

                if isinstance(self.fixed_parameters, dict):
                    params.update(self.fixed_parameters)

                skf = StratifiedKFold(self.n_folds,
                                      shuffle=self.shuffle,
                                      random_state=self.random_state)

                for train_index, valid_index in skf.split(x, y):

                    x_train, y_train = x[train_index, :], y[train_index]
                    x_valid, y_valid = x[valid_index, :], y[valid_index]

                    knn = KNeighborsClassifier(**params)

                    knn.fit(x_train, y_train)

                    y_valid_hat = knn.predict(x_valid)

                    loss_valid = log_loss(y_valid, y_valid_hat)

                    scores.append(loss_valid)

                result = np.mean(scores)

                self.iterations.append((params, result))

                return result

            except ValueError:

                return np.inf

        return self.execute_optimization(objective, space)
