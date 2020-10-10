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

import warnings
warnings.filterwarnings("ignore")


class BayesianOptimizer(object):

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

    def execute_optimization(self, objective, space):
        params = gp_minimize(objective, space, n_calls=self.n_calls, random_state=self.random_state,
                             verbose=(self.verbose >= 0), n_jobs=self.n_jobs).x

        return {space[i].name: params[i] for i in range(len(space))}

    def optimize(self, x, y):
        pass
