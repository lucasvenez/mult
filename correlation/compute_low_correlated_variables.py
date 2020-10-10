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


def select_variables_based_on_pearson(file_path='data/output/genetic_correlation.tsv'):

    variables, to_be_deleted = set(), set()

    with open(file_path) as file:

        count_var, previous_var = 0, None

        for line in file:

            try:

                var1, pearson, var2 = line.split('\t')

                var2 = var2.replace('\n', '')

                if pearson != '':

                    pearson = abs(float(pearson))

                    if pearson >= .75:
                        to_be_deleted = to_be_deleted.union({var2})

                    if previous_var != var1:

                        variables = variables.union({var1})

                        count_var += 1

                        print('Computing var {} of {}'.format(count_var, 27167))

                    previous_var = var1

            except Exception as e:
                pass

    import pandas as pd

    return pd.DataFrame({'variable': list(variables.difference(to_be_deleted))})
