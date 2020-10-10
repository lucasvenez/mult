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
from skopt.space import Real, Integer
from skopt.utils import use_named_args

from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import log_loss

from lightgbm import LGBMModel

import pandas as pd
import numpy as np

import warnings
warnings.filterwarnings("ignore")


class LightGBMOptimizer(object):

    def __init__(self,
                 n_folds=3, n_calls=50, shuffle=True, early_stopping_rounds=None,
                 fixed_parameters=None, random_state=None, verbose=-1, n_jobs=-1, use_gpu=False):

        self.n_calls = n_calls
        self.n_folds = n_folds
        self.random_state = random_state
        self.shuffle = shuffle
        self.verbose = verbose
        self.n_jobs = n_jobs
        self.optimization_details = {}
        self.early_stopping_rounds = early_stopping_rounds
        self.fixed_parameters = fixed_parameters if fixed_parameters is not None else dict()
        self.use_gpu = use_gpu

        self.iterations = []

    def execute_optimization(self, objective, space):
        params = gp_minimize(objective, space, n_calls=self.n_calls, random_state=self.random_state,
                             verbose=(self.verbose >= 0), n_jobs=-1).x

        return {space[i].name: params[i] for i in range(len(space))}

    def optimize(self, x, y):

        assert isinstance(x, pd.DataFrame) or isinstance(x, np.ndarray), \
            'x should be a pd.DataFrame or np.ndarray'

        assert isinstance(y, pd.DataFrame) or isinstance(y, pd.Series) or isinstance(y, np.ndarray), \
            'y should be a pd.DataFrame or pd.Series or np.ndarray'

        if isinstance(x, pd.DataFrame):
            x = x.values

        if isinstance(y, pd.DataFrame) or isinstance(y, pd.Series):
            y = y.values

        self.iterations = []

        space = [
            Integer(2, 4, name='num_leaves'),
            Real(1e-1, 1.000, name='scale_pos_weight'),
            Real(1e-1, 1.000, name='colsample_bytree'),
            Integer(2, 50, name='min_child_samples'),
            Integer(2000, 8000, name='bin_construct_sample_cnt'),
            Integer(2, 512, name='max_bin'),
            Real(1e-3, 1, name='min_sum_hessian_in_leaf'),
            Real(1e-3, 1, name='bagging_fraction'),
            Real(0.01, 1, name='feature_fraction'),
            Real(0.01, 1, name='feature_fraction_bynode'),
        ]

        @use_named_args(space)
        def objective(
            num_leaves,
            scale_pos_weight,
            colsample_bytree,
            min_child_samples,
            bin_construct_sample_cnt,
            max_bin,
            min_sum_hessian_in_leaf,
            bagging_fraction,
            feature_fraction,
            feature_fraction_bynode,
        ):
            try:
                scores = []

                params = {
                    'num_leaves': int(num_leaves),
                    'scale_pos_weight': scale_pos_weight,
                    'colsample_bytree': colsample_bytree,
                    'min_child_samples': int(min_child_samples),
                    'bin_construct_sample_cnt': int(bin_construct_sample_cnt),
                    'max_bin': int(max_bin),
                    'min_sum_hessian_in_leaf': min_sum_hessian_in_leaf,
                    'bagging_fraction': bagging_fraction,
                    'feature_fraction': feature_fraction,
                    'feature_fraction_bynode': feature_fraction_bynode,

                    'n_jobs': self.n_jobs,
                    'silent': self.verbose < 1,
                    'random_state': self.random_state}

                if isinstance(self.fixed_parameters, dict):
                    params.update(self.fixed_parameters)

                if self.use_gpu:
                    params.update({'device': 'gpu', 'gpu_platform_id': 1, 'gpu_device_id': 0})

                params.update({'metric': 'binary_logloss'})

                skf = StratifiedKFold(
                    self.n_folds, shuffle=self.shuffle, random_state=self.random_state)

                for train_index, valid_index in skf.split(x, y):

                    x_train, y_train = x[train_index, :], y[train_index]
                    x_valid, y_valid = x[valid_index, :], y[valid_index]

                    gbm = LGBMModel(**params)

                    gbm.fit(x_train, y_train,
                            eval_set=[(x_valid, y_valid)],
                            early_stopping_rounds=self.early_stopping_rounds,
                            verbose=int(self.verbose > 0))

                    y_valid_hat = gbm.predict(x_valid)

                    loss_valid = log_loss(y_valid, y_valid_hat)

                    scores.append(loss_valid)

                result = np.mean(scores)

                self.iterations.append((params, result))

                return result

            except Exception as e:
                import os, sys
                print(str(e))
                exc_type, exc_obj, exc_tb = sys.exc_info()
                fname = os.path.split(exc_tb.tb_frame.f_code.co_filename)[1]
                print(exc_type, fname, exc_tb.tb_lineno)
                raise e

        return self.execute_optimization(objective, space)
