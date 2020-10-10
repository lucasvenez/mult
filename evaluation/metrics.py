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
import numpy as np


def classification_metrics(tn, fp, fn, tp):
    
    sensitivity = (tp / float(tp + fn)) if tp + fn > 0 else 1

    precision = (tp / float(tp + fp)) if tp + fp > 0 else 1

    specificity = (tn / float(tn + fp)) if tn + fp > 0 else 1

    accuracy = (tp + tn) / (tp + tn + fp + fn)
    
    return {'accuracy': accuracy, 'precision': precision, 'sensitivity': sensitivity, 'specificity': specificity}


def ks_score(y_true, y_hat):
    
    try:
        
        deciles = [-1] + list(np.quantile(y_hat, [.1, .2, .3, .4, .5, .6, .7, .8, .9, 1]))

        result = []

        for i in y_hat:
            for index, (a, b) in enumerate(zip(deciles[:-1], deciles[1:])):
                if a < i <= b:
                    result.append(index + 1)
                    continue

        result = pd.DataFrame({'decile': result})

        result['positive'] = y_true
        result['negative'] = 1 - y_true

        negative_sum = result['negative'].sum()
        positive_sum = result['positive'].sum()

        result = result.groupby(by=['decile']).sum()

        result['positive_percentage'] = result['positive'] / positive_sum
        result['negative_percentage'] = result['negative'] / negative_sum

        result['positive_percentage_cumsum'] = result['positive_percentage'].cumsum()
        result['negative_percentage_cumsum'] = result['negative_percentage'].cumsum()

        result['ks'] = np.abs(result['positive_percentage_cumsum'] - result['negative_percentage_cumsum'])

        return result['ks'].max()
    
    except ValueError:
        return 0.0