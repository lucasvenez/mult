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

from sklearn.model_selection import KFold
import unittest
import pandas as pd
from model import ExtremeGradientBoosting


class TestTrain(unittest.TestCase):

    def test_xgb(self):

        dependent_variable = 'Y'

        #
        #
        #

        mic = pd.read_table('../output/correlation_onevar_valid.tsv', sep='\t')

        genetic_data = pd.read_table('../input/iss_status_without_log_valid.tsv', sep='\t')

        independent_variables = mic[mic['MIC'] > 0.003]['IND_VAR'].as_matrix().tolist()

        genetic_data['PUBLIC_ID'] = genetic_data['PUBLIC_ID'].map(lambda x: x.replace('_', ''))

        genetic_data.set_index('PUBLIC_ID', inplace=True)

        cli_data = pd.read_table('../input/clin_status.tsv', sep='\t').set_index('ID')[['therapyA']]

        cli_data.index.rename('PUBLIC_ID', inplace=True)

        dummies_therapy = pd.get_dummies(cli_data['therapyA'])

        cli_data = pd.concat([cli_data, dummies_therapy], axis=1)

        independent_variables += list(dummies_therapy.columns)

        all = genetic_data.join(cli_data)

        # ========================================================

        x = all[independent_variables].as_matrix()
        y = all[dependent_variable].as_matrix().reshape((-1, 1))

        # ========================================================

        kfold = KFold(n_splits=4)

        xgb = ExtremeGradientBoosting(eval_metric=['mlogloss', 'merror'])
        y -= 1
        for train, test in kfold.split(x, y, y):
            xgb.optimize(x[train, :], y[train, :],
                         test_x=x[test, :], test_y=y[test, :],
                         valid_x=x[test, :], valid_y=y[test, :])
