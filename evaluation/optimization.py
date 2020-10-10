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

from sklearn.metrics import confusion_matrix
import numpy as np


def optimize_threshold(y_true, y_, by=0.01):

    t, max_metric = None, -np.inf

    for i in np.arange(min(y_), max(y_), by):

        y_hat = np.copy(y_)

        filter__ = y_hat >= i

        y_hat[filter__], y_hat[~filter__] = 1, 0

        tn, fp, fn, tp = confusion_matrix(y_true, y_hat).ravel()

        sensitivity = (tp / float(tp + fn)) if tp + fn > 0 else 1

        specificity = (tn / float(tn + fp)) if tn + fp > 0 else 1

        ks = abs(sensitivity + specificity - 1.)
        
        metric = ks

        if metric > max_metric and metric is not np.inf:

            max_metric = metric

            t = i

    return t