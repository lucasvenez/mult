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

from optimization import SVMOptimizer, LightGBMOptimizer, LogisticRegressionOptimizer, MLPOptimizer

import unittest
from sklearn.datasets import load_wine


class OptimizerTestCase(unittest.TestCase):
    """
    Unit test for all models to have their hyper parameters optimized.
    """
    def setUp(self):
        """
        Loading wine benchmark data set and using only classes 0 and 1.
        :return:
        """
        self.X, self.y = load_wine(True)
        self.X, self.y = self.X[(self.y == 0) | (self.y == 1), :], self.y[(self.y == 0) | (self.y == 1)]

    def test_svm_optimizer(self):

        opt = SVMOptimizer(random_state=10)

        params = opt.optimize(self.X, self.y)

        expected = {
            'C': 9.99679631077478,
            'gamma': 1.5003828761999214e-06,
            'degree': 6,
            'kernel': 'linear'}

        self.check_result(expected, params)

    def test_lightgbm_optimizer(self):

        opt = LightGBMOptimizer(early_stopping_rounds=100, random_state=10)

        params = opt.optimize(self.X, self.y)

        expected = {'learning_rate': 3.1174949556452575e-05,
                    'num_leaves': 255,
                    'max_depth': 11,
                    'scale_pos_weight': 0.8335922453814891,
                    'min_child_weight': 0.587488519568359,
                    'colsample_bytree': 0.12265455557597107,
                    'min_split_gain': 3.5159286177583637,
                    'min_child_samples': 266,
                    'subsample': 0.2683307891083734,
                    'bin_construct_sample_cnt': 290227}

        self.check_result(expected, params)

    def test_lr_optimizer(self):

        opt = LogisticRegressionOptimizer(random_state=10)

        params = opt.optimize(self.X, self.y)

        print(params)

        expected = {'penalty': 'l2',
                    'dual': False,
                    'tol': 0.0016408721292541359,
                    'C': 0.14439654265063323,
                    'fit_intercept': True,
                    'intercept_scaling': 1.3361605407109191e-05,
                    'solver': 'sag'}

        self.check_result(expected, params)

    def test_mlp_optimizer(self):

        opt = MLPOptimizer(random_state=10)

        params = opt.optimize(self.X, self.y)

        print(params)

        expected = {'hidden_layer_sizes': 67,
                    'n_hidden_layers': 2,
                    'activation': 'logistic',
                    'solver': 'adam',
                    'alpha': 0.005622962806072959,
                    'learning_rate': 'constant',
                    'learning_rate_init': 0.0068680277204210645,
                    'power_t': 0.00011541559825941273,
                    'tol': 0.0006177440031163255,
                    'momentum': 0.00039321543339091477,
                    'beta_1': 0.050580591307804734,
                    'beta_2': 0.16945004562775126,
                    'epsilon': 1.0385550140003627e-06}

        self.check_result(expected, params)

    def check_result(self, expected, actual):
        for k, v in expected.items():
            self.assertEqual(v, actual[k])
