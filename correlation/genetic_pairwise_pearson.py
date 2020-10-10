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

from correlation import PearsonOneVsAll
import pandas as pd
import os

import datetime


def pairwise_pearson():

    print('Loading data.')

    df_gen = pd.read_csv('input/iss_status_with_genetic.tsv', sep='\t')

    df_gen = df_gen.rename(columns={'PUBLIC_ID': 'ID'}).set_index('ID')

    df_gen = df_gen[[col for col in df_gen.columns if 'FPKM' in col and 'LOG' not in col]]

    df_gen.columns = [col.replace('_FPKM', '').replace('-', '').replace('_', '') for col in df_gen.columns]

    df_gen.index = df_gen.index.str.replace('MMRF_', '')

    df_gen_nona = df_gen.dropna(how='all')

    del df_gen

    pearson = PearsonOneVsAll()

    print('Calculating correlation.')

    result = []

    start = datetime.datetime.now()

    for index, (name, values) in enumerate(df_gen_nona.iloc[:, :-1].iteritems()):

        subset = df_gen_nona.iloc[:,(index + 1):]

        names = subset.columns

        print('Computing pearson for {} genetic expression'.format(name))

        scores = pearson.compute_score(values.as_matrix(), subset.as_matrix())

        row = pd.DataFrame({'var2': names, 'pearson': scores})

        row['var1'] = name

        result.append(row.set_index('var1'))

        if (index + 1) % 1000 == 0:
            for r in result:
                if not os.path.isfile('genetic_correlation.tsv'):
                    r.to_csv('genetic_correlation.tsv', sep='\t', mode='w', header=True)
                else:
                    r.to_csv('genetic_correlation.tsv', sep='\t', mode='a', header=False)

            result = []

    for r in result:
        if not os.path.isfile('genetic_correlation.tsv'):
            r.to_csv('genetic_correlation.tsv', sep='\t', mode='w', header=True)
        else:
            r.to_csv('genetic_correlation.tsv', sep='\t', mode='a', header=False)

    print('Finished after {} of processing'.format(str(datetime.datetime.now() - start)))
