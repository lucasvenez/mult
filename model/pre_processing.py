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

import pandas as pd


class PreProcessing(object):
    """

    """

    def __init__(self):

        self.dropped_columns = []

        self.dummy_columns = {}

    def fit(self, data_set):

        assert isinstance(data_set, pd.DataFrame)

        for col in data_set.columns:

            if len(data_set[col].unique()) < 2:
                self.dropped_columns.append(col)

            elif data_set[col].dtype == 'object':
                self.dummy_columns[col] = data_set[col].unique().tolist()

    def transform(self, data_set):

        assert isinstance(data_set, pd.DataFrame), 'data_set parameter should be of pd.DataFrame type'

        for col in self.dropped_columns:
            if col in data_set.columns:
                del data_set[col]

        for col in self.dummy_columns:

            if col in data_set.columns:

                dummy = data_set[col].map(lambda x: x if x in self.dummy_columns[col] else None)

                dummy = pd.get_dummies(dummy)

                dummy.columns = [col + '_' + str(c) for c in dummy.columns]

                for v in self.dummy_columns[col]:
                    if v is not None:
                        if col + '_' + v not in dummy.columns:
                            dummy[col + '_' + v] = 0

                dummy = dummy[[col + '_' + v for v in self.dummy_columns[col] if v is not None]]

                del data_set[col]

                data_set = data_set.join(dummy, how='inner')

        return data_set

    def fit_transform(self, data_set):

        assert isinstance(data_set, pd.DataFrame)

        self.fit(data_set)

        return self.transform(data_set)
