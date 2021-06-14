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
from skopt.space import Integer, Categorical, Real
from skopt.utils import use_named_args

from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import log_loss

from sklearn.ensemble import RandomForestClassifier

import numpy as np

import warnings
warnings.filterwarnings("ignore")


class RFOptimizer(object):

    def __init__(self,
                 n_folds=10, n_calls=10, shuffle=True,
                 fixed_parameters={}, random_state=None,
                 verbose=-1, n_jobs=-1):

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
        params = gp_minimize(objective, space, n_calls=self.n_calls,
                             random_state=self.random_state,
                             verbose=(self.verbose >= 0), n_jobs=-1).x

        return {space[i].name: params[i] for i in range(len(space))}

    def optimize(self, x, y):
        """"
        n_estimators int, default=100, The number of trees in the forest. Changed in version 0.22: The default value of n_estimators changed from 10 to 100 in 0.22.

        criterion{“gini”, “entropy”}, default=”gini”, The function to measure the quality of a split. Supported criteria are “gini” for the Gini impurity and “entropy” for the information gain. Note: this parameter is tree-specific.

        max_depth int, default=None, The maximum depth of the tree. If None, then nodes are expanded until all leaves are pure or until all leaves contain less than min_samples_split samples.

        min_samples_split int or float, default=2, The minimum number of samples required to split an internal node: If int, then consider min_samples_split as the minimum number. If float, then min_samples_split is a fraction and ceil(min_samples_split * n_samples) are the minimum number of samples for each split. Changed in version 0.18: Added float values for fractions.

        min_samples_leaf int or float, default=1, The minimum number of samples required to be at a leaf node. A split point at any depth will only be considered if it leaves at least min_samples_leaf training samples in each of the left and right branches. This may have the effect of smoothing the model, especially in regression. If int, then consider min_samples_leaf as the minimum number. If float, then min_samples_leaf is a fraction and ceil(min_samples_leaf * n_samples) are the minimum number of samples for each node. Changed in version 0.18: Added float values for fractions.

        min_weight_fraction_leaf float, default=0.0 The minimum weighted fraction of the sum total of weights (of all the input samples) required to be at a leaf node. Samples have equal weight when sample_weight is not provided.

        max_leaf_nodes int, default=None, Grow trees with max_leaf_nodes in best-first fashion. Best nodes are defined as relative reduction in impurity. If None then unlimited number of leaf nodes.

        min_impurity_decrease float, default=0.0, A node will be split if this split induces a decrease of the impurity greater than or equal to this value. The weighted impurity decrease equation is the following:

        min_impurity_split float, default=None, Threshold for early stopping in tree growth. A node will split if its impurity is above the threshold, otherwise it is a leaf.

        ccp_alpha non-negative float, default=0.0, Complexity parameter used for Minimal Cost-Complexity Pruning. The subtree with the largest cost complexity that is smaller than ccp_alpha will be chosen. By default, no pruning is performed. See Minimal Cost-Complexity Pruning for details.

        New in version 0.22.

        max_samples int or float, default=None, If bootstrap is True, the number of samples to draw from X to train each base estimator.

        n_jobs int, default=None
        The number of jobs to run in parallel. fit, predict, decision_path and apply are all parallelized over the trees. None means 1 unless in a joblib.parallel_backend context. -1 means using all processors. See Glossary for more details.

        random_state int or RandomState, default=None
        Controls both the randomness of the bootstrapping of the samples used when building trees (if bootstrap=True) and the sampling of the features to consider when looking for the best split at each node (if max_features < n_features). See Glossary for details.


        """
        self.iterations = []

        space = [
            Integer(10, 5000, name='n_estimators'),
            Categorical(['gini', 'entropy'], name='criterion'),
            Integer(1, 100, name='max_depth'),
            Integer(1, 100, name='min_samples_split'),
            Integer(1, 100, name='min_samples_leaf'),
            Real(1e-8, 1, name='min_weight_fraction_leaf'),
            Integer(1, 100, name='max_leaf_nodes'),
            Real(1e-8, 1, name='min_impurity_decrease'),
            Real(1e-8, 1, name='min_impurity_split')
        ]

        @use_named_args(space)
        def objective(
            n_estimators,
            criterion,
            max_depth,
            min_samples_split,
            min_samples_leaf,
            min_weight_fraction_leaf,
            max_leaf_nodes,
            min_impurity_decrease,
            min_impurity_split,
        ):
            try:
                scores = []

                params = {
                    'n_estimators': int(round(n_estimators, 0)),
                    'criterion': criterion,
                    'max_depth': max_depth,
                    'min_samples_split': min_samples_split,
                    'min_samples_leaf': min_samples_leaf,
                    'min_weight_fraction_leaf': min_weight_fraction_leaf,
                    'max_leaf_nodes': int(round(max_leaf_nodes, 0)),
                    'min_impurity_decrease': min_impurity_decrease,
                    'min_impurity_split': min_impurity_split,

                    'random_state': self.random_state}

                if isinstance(self.fixed_parameters, dict):
                    params.update(self.fixed_parameters)

                skf = StratifiedKFold(self.n_folds,
                                      shuffle=self.shuffle,
                                      random_state=self.random_state)

                for train_index, valid_index in skf.split(x, y):

                    x_train, y_train = x[train_index, :], y[train_index]
                    x_valid, y_valid = x[valid_index, :], y[valid_index]

                    rf = RandomForestClassifier(**params, n_jobs=-1)

                    rf.fit(x_train, y_train)

                    y_valid_hat = rf.predict(x_valid)

                    loss_valid = log_loss(y_valid, y_valid_hat)

                    scores.append(loss_valid)

                result = np.mean(scores)

                self.iterations.append((params, result))

                return result

            except ValueError:

                return np.inf

        return self.execute_optimization(objective, space)
