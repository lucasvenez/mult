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

from model import *

import unittest


class ModelTest(unittest.TestCase):
    """

    """
    def test_dummy(self):

        df = pd.DataFrame({
            'A': [1, 2, 3, 4, 5, 6, 7, 8],
            'B': [0, 0, 0, 0, 0, 0, 0, 0],
            'C': ['C3', 'C2', 'C3', 'C3', 'C2', 'C2', 'C1', 'C1'],
            'D': [1, 1, 1, 1, 1, 1, 1, 1]
        })

        df2 = pd.DataFrame({
            'A': [8, 8, 7],
            'B': [3, 2, 1],
            'C': ['C4', 'C4', 'C3'],
            'D': [1, 1, 1]
        })

        pp = PreProcessing()

        r1 = pp.fit_transform(df)

        r2 = pp.transform(df2)

        self.assertListEqual(list(r1.columns), list(r2.columns))
