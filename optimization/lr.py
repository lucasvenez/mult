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
from skopt.space import Real, Categorical
from skopt.utils import use_named_args
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import log_loss

from sklearn.linear_model import LogisticRegression

import numpy as np

import warnings
warnings.filterwarnings("ignore")


class LogisticRegressionOptimizer(object):

    def __init__(self,
                 n_folds=3, n_calls=50, shuffle=True,
                 fixed_parameters=None, random_state=None, verbose=-1, n_jobs=-1):
        self.n_calls = n_calls
        self.n_folds = n_folds
        self.random_state = random_state
        self.shuffle = shuffle
        self.verbose = verbose
        self.n_jobs = n_jobs
        self.optimization_details = {}
        self.fixed_parameters = fixed_parameters if fixed_parameters is None else dict()

        self.iterations = []

    def optimize(self, x, y):
        """
        Description of each optimized hyperparameter. Checkout original description at
        https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LogisticRegression.html.
        Accessed on 28/06/2020 at 15:07:01.

        penalty{‘l1’, ‘l2’, ‘elasticnet’, ‘none’}, default=’l2’
            Used to specify the norm used in the penalization. The ‘newton-cg’, ‘sag’ and ‘lbfgs’
             support only l2 penalties. ‘elasticnet’ is only supported by the ‘saga’ solver. If ‘none’
             (not supported by the liblinear solver), no regularization is applied.

        dual: bool, default=False
            Dual or primal formulation. Dual formulation is only implemented for l2 penalty with liblinear solver.
            Prefer dual=False when n_samples > n_features.

        tol: float, default=1e-4
            Tolerance for stopping criteria.

        C: float, default=1.0
            Inverse of regularization strength; must be a positive float. Like in support vector machines,
            smaller values specify stronger regularization.

        fit_intercept: bool, default=True
            Specifies if a constant (a.k.a. bias or intercept) should be added to the decision function.

        intercept_scaling: float, default=1
            Useful only when the solver ‘liblinear’ is used and self.fit_intercept is set to True. In this case,
            x becomes [x, self.intercept_scaling], i.e. a “synthetic” feature with constant value equal to
            intercept_scaling is appended to the instance vector. The intercept becomes
            intercept_scaling * synthetic_feature_weight.

            Note! the synthetic feature weight is subject to l1/l2 regularization as all other features.
            To lessen the effect of regularization on synthetic feature weight (and therefore on the intercept)
            intercept_scaling has to be increased.

        solver{‘newton-cg’, ‘lbfgs’, ‘liblinear’, ‘sag’, ‘saga’}, default=’lbfgs’
            Algorithm to use in the optimization problem.

            For small datasets, ‘liblinear’ is a good choice, whereas ‘sag’ and ‘saga’ are faster for large ones.

            For multiclass problems, only ‘newton-cg’, ‘sag’, ‘saga’ and ‘lbfgs’ handle multinomial loss; ‘liblinear’
            is limited to one-versus-rest schemes.

            ‘newton-cg’, ‘lbfgs’, ‘sag’ and ‘saga’ handle L2 or no penalty

            ‘liblinear’ and ‘saga’ also handle L1 penalty

            ‘saga’ also supports ‘elasticnet’ penalty

            ‘liblinear’ does not support setting penalty='none'

            Note that ‘sag’ and ‘saga’ fast convergence is only guaranteed on features with
            approximately the same scale. You can preprocess the data with a scaler from sklearn.preprocessing.

            New in version 0.17: Stochastic Average Gradient descent solver.

            New in version 0.19: SAGA solve

        :param x:
        :param y:
        :param num_boost_round:
        :param early_stopping_rounds:
        :return:
        """

        space = [
            Categorical(['l1', 'l2', 'elasticnet', 'none'], name='penalty'),
            Categorical([False, True], name='dual'),
            Real(1e-5, 1.00, prior='log-uniform', name='tol'),
            Real(1e-5, 1.00, prior='log-uniform', name='C'),
            Categorical([False, True], name='fit_intercept'),
            Real(1e-5, 1.00, prior='log-uniform', name='intercept_scaling'),
            Categorical(['newton-cg', 'lbfgs', 'liblinear', 'sag', 'saga'], name='solver')]

        @use_named_args(space)
        def objective(
                penalty,
                dual,
                tol,
                C,
                fit_intercept,
                intercept_scaling,
                solver):

            try:

                scores, params = [], {
                    'penalty': penalty,
                    'dual': dual,
                    'tol': tol,
                    'C': C,
                    'fit_intercept': fit_intercept,
                    'intercept_scaling': intercept_scaling,
                    'solver': solver,

                    'random_state': self.random_state,
                    'n_jobs': self.n_jobs,
                    'verbose': self.verbose}

                if isinstance(self.fixed_parameters, dict):
                    params.update(self.fixed_parameters)

                skf = StratifiedKFold(
                    self.n_folds, shuffle=self.shuffle, random_state=self.random_state)

                for train_index, valid_index in skf.split(x, y):

                    x_train, y_train = x[train_index, :], y[train_index]
                    x_valid, y_valid = x[valid_index, :], y[valid_index]

                    lr = LogisticRegression(**params)

                    model = lr.fit(x_train, y_train)

                    y_hat = model.predict_proba(x_valid)

                    scores.append(log_loss(y_valid, y_hat))

                result = np.mean(scores)

                self.iterations.append((params, result))

                return result

            except ValueError:

                return np.inf

        return self.execute_optimization(objective, space)

    def execute_optimization(self, objective, space):
        params = gp_minimize(objective, space, n_calls=self.n_calls, random_state=self.random_state,
                             verbose=(self.verbose >= 0), n_jobs=self.n_jobs).x

        return {space[i].name: params[i] for i in range(len(space))}


