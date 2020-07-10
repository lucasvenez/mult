from optimization import BayesianOptimizer

from skopt.space import Real, Integer
from skopt.utils import use_named_args
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score

from lightgbm import LGBMModel, Dataset

import numpy as np

import warnings
warnings.filterwarnings("ignore")


class LightGBMOptimizer(BayesianOptimizer):

    def __init__(self, n_folds=3, n_calls=50, shuffle=True, early_stopping_rounds=None,
                 random_state=None, verbose=-1, n_jobs=-1, use_gpu=False):
        super().__init__(n_folds, n_calls, shuffle, early_stopping_rounds, random_state, verbose, n_jobs)
        self.use_gpu = use_gpu

    def optimize(self, x, y):

        space = [
            Real(1e-6, 1e-1, 'log-uniform', name='learning_rate'),
            Integer(4, 512, name='num_leaves'),
            Integer(4, 20, name='max_depth'),
            Real(0.01, 1.00, name='scale_pos_weight'),
            Real(0.01, 1.00, name='min_child_weight'),
            Real(0.10, 1.00, name='colsample_bytree'),
            Real(0.001, 100, 'log-uniform', name='min_split_gain'),
            Integer(1, 1000, name='min_child_samples'),
            Real(0.01, 0.99, name='subsample'),
            Integer(200000, 800000, name='bin_construct_sample_cnt')]

        @use_named_args(space)
        def objective(
                      learning_rate,
                      num_leaves,
                      max_depth,
                      scale_pos_weight,
                      min_child_weight,
                      colsample_bytree,
                      min_split_gain,
                      min_child_samples,
                      subsample,
                      bin_construct_sample_cnt):
            try:
                scores = []

                params = {
                    'learning_rate': learning_rate,
                    'num_leaves': int(num_leaves),
                    'max_depth': int(max_depth),
                    'scale_pos_weight': scale_pos_weight,
                    'min_child_weight': min_child_weight,

                    'colsample_bytree': colsample_bytree,
                    'min_split_gain': min_split_gain,
                    'min_child_samples': int(min_child_samples),
                    'subsample': subsample,
                    'bin_construct_sample_cnt': int(bin_construct_sample_cnt),

                    'n_jobs': self.n_jobs,
                    'silent': self.verbose < 1,
                    'random_state': self.random_state}

                params.update(super().fixed_parameters)

                if self.use_gpu:
                    params.update({'device': 'gpu', 'gpu_platform_id': 1, 'gpu_device_id': 0})

                skf = StratifiedKFold(
                    self.n_folds, shuffle=self.shuffle, random_state=self.random_state)

                for train_index, valid_index in skf.split(x, y):

                    x_train, y_train = x[train_index, :], y[train_index]
                    x_valid, y_valid = x[valid_index, :], y[valid_index]

                    gbm = LGBMModel(**params)

                    gbm.fit(x_train, y_valid,
                            eval_set=Dataset(x_valid, y_valid),
                            early_stopping_rounds=self.early_stopping_rounds,
                            verbose=int(self.verbose > 0))

                    y_hat = gbm.predict(x_valid)

                    scores.append(roc_auc_score(y_valid, y_hat))

                return -np.mean([s for s in scores if s is not None])

            except ValueError:
                return 0.0

        return super().execute_optimization(objective, space)
