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

import unittest

from pipeline import SMLA
from optimization import *
from lightgbm import LGBMModel
from sklearn.datasets import load_wine
from sklearn.model_selection import train_test_split


class PipelineTest(unittest.TestCase):

    def setUp(self):
        self.X, self.y = load_wine(True)
        self.X, self.y = self.X[(self.y == 0) | (self.y == 1), :], self.y[(self.y == 0) | (self.y == 1)]

    def test_smla(self):

        smla = SMLA(LGBMModel, LightGBMOptimizer)

        x_train, y_train, x_valid, y_valid = train_test_split(self.X, self.y, stratify=self.y)

        smla.fit(x_train, y_train, x_valid, y_valid, early_stopping_rounds=100)

        smla.predict(x_valid)
