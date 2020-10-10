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

from correlation import compute_ks

import pandas as pd


class FeatureSelector(object):
    """

    """

    def __init__(self, pvalue_threshould=0.05, pcc_threshould=0.95):

        self.pvalue_threshould = pvalue_threshould

        self.pcc_threshould = pcc_threshould

        self.selected_features = None

    def fit(self, x, y):

        assert isinstance(x, pd.DataFrame)

        assert isinstance(y, pd.Series) or (isinstance(y, pd.DataFrame) and len(y.colums) == 1)

        if isinstance(y, pd.DataFrame):
            y = y[y.columns[0]]

        x = x.join(y, how='inner')

        kss = compute_ks(x, [y.name])

        kss = kss.loc[kss['ks'] < self.pvalue_threshould, :]

        ind_vars = kss.sort_values(by='ks')['IND_VAR'].values

        pccs = pd.DataFrame(x[ind_vars].corr())

        all_features = set(pccs.index)

        deleted_features = set()

        for i, n in enumerate(pccs.columns):

            tmp = pccs.loc[list(pccs.index[(i + 1):]), n].reset_index()

            tmp = pd.DataFrame({'V2': tmp.iloc[:, 0], 'PCC': tmp[n].abs()})

            subset = set(tmp.loc[tmp['PCC'] >= self.pcc_threshould, 'V2'])

            deleted_features = deleted_features.union(subset)

        self.selected_features = all_features.difference(deleted_features)

    def transform(self, x):
        return x[self.selected_features]

    def fit_transform(self, x, y):

        self.fit(x, y)

        return self.transform(x)
