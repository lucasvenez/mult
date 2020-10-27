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

from optimization import LightGBMOptimizer, SVMOptimizer, LogisticRegressionOptimizer
from optimization import KNNOptimizer, MLPOptimizer, RFOptimizer

from pipeline import SelectMarker

from lightgbm import LGBMModel, Dataset

from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC

from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split

import numpy as np
import pandas as pd


class SMLA(SelectMarker):
    
    def __init__(self,
                 predictor,
                 optimizer_default_params=None,
                 model_default_params=None,
                 verbose=-1,
                 random_state=None,
                 use_gpu=False,
                 test_size=.2,
                 n_gene_limit=None):

        assert isinstance(optimizer_default_params, dict) or optimizer_default_params is None
        assert isinstance(model_default_params, dict) or model_default_params is None

        #
        self.predictor = predictor
        self.model_default_params = model_default_params

        #
        self.optimized_params = None
        self.optimizer_default_params = optimizer_default_params

        #
        self.model = None
        self.fitted_shape = None

        #
        self.random_state = random_state
        self.verbose = verbose
        self.use_gpu = use_gpu

        #
        self.test_size = test_size

        #
        self.scaler = MinMaxScaler()

        #
        self.n_gene_limit = n_gene_limit
        self.selected_clinical = None
        self.selected_genes = None

        #
        if self.predictor == 'lightgbm':
            self.optimizer = LightGBMOptimizer(**self.optimizer_default_params)

        elif self.predictor == 'svm':
            self.optimizer = SVMOptimizer(**self.optimizer_default_params)

        elif self.predictor == 'knn':
            self.optimizer = KNNOptimizer(**self.optimizer_default_params)

        elif self.predictor == 'lr':
            self.optimizer = LogisticRegressionOptimizer(**self.optimizer_default_params)

        elif self.predictor == 'mlp':
            self.optimizer = MLPOptimizer(**self.optimizer_default_params)

        elif self.predictor == 'rf':
            self.optimizer = RFOptimizer(**self.optimizer_default_params)

        else:
            raise ValueError('predictor should be one of the following: lightgbm, svm, knn, lr, or mlp')

    def fit(self, clinical_markers, genes, treatments, clinical_outcome,
            clinical_marker_selection_threshold,
            genes_marker_selection_threshold,
            early_stopping_rounds=None):

        ######

        self.selected_clinical = self.select_markers(
            clinical_markers, clinical_outcome, threshold=clinical_marker_selection_threshold)

        self.selected_genes = self.select_markers(
            genes, clinical_outcome, threshold=genes_marker_selection_threshold)

        clinical_markers = clinical_markers.loc[:, self.selected_clinical[0]].join(treatments)
        genes = genes.loc[:, self.selected_genes[0]]

        x, y = clinical_markers.join(genes, how='inner').fillna(0).values, clinical_outcome.values

        x = self.scaler.fit_transform(x)

        ######

        self.fitted_shape = x.shape

        self.optimized_params = self.optimizer.optimize(x, y)

        self.optimized_params['random_state'] = self.random_state

        self.optimized_params['n_jobs'] = -1

        if self.model_default_params is not None:
            self.optimized_params.update(self.model_default_params)

        if self.predictor == 'lightgbm':
            self.fit_lightgbm(x, y, early_stopping_rounds)

        elif self.predictor == 'svm':
            self.fit_svm(x, y)

        elif self.predictor == 'knn':
            self.fit_knn(x, y)

        elif self.predictor == 'lr':
            self.fit_lr(x, y)

        elif self.predictor == 'mlp':
            self.fit_mlp(x, y, early_stopping_rounds)

        elif self.predictor == 'rf':
            self.fit_rf(x, y)

    def fit_rf(self, x, y):

        self.model = RandomForestClassifier(**self.optimized_params)

        self.model.fit(x, y)

    def fit_lightgbm(self, x, y, early_stopping_rounds):

        self.model = LGBMModel(**self.optimized_params)

        if early_stopping_rounds is not None:

            x_valid, y_valid = train_test_split(x, stratify=y, shuffle=True,
                                                test_size=self.test_size, random_state=self.random_state)

            self.model.fit(x, y,
                           eval_set=Dataset(x_valid, y_valid),
                           early_stopping_rounds=early_stopping_rounds,
                           verbose=self.verbose)

        else:
            self.model.fit(x, y)

    def fit_svm(self, x, y):

        del self.optimized_params['n_jobs']

        self.model = SVC(**self.optimized_params, probability=True)

        self.model.fit(x, y)

    def fit_lr(self, x, y):

        self.model = LogisticRegression(**self.optimized_params)

        self.model.fit(x, y)

    def fit_mlp(self, x, y, early_stopping_rounds):

        esr = early_stopping_rounds is not None and early_stopping_rounds > 0

        del self.optimized_params['n_jobs']

        self.model = MLPClassifier(**self.optimized_params,
                                   early_stopping=esr,
                                   validation_fraction=self.test_size)

        self.model.fit(x, y)

    def fit_knn(self, x, y):

        del self.optimized_params['random_state']

        self.model = KNeighborsClassifier(**self.optimized_params)

        self.model.fit(x, y)

    def predict(self, genes, clinical_markers=None, treatments=None):

        assert isinstance(genes, pd.DataFrame), 'genes should a pd.DataFrame'

        if clinical_markers is not None:
            X = clinical_markers.loc[:, self.selected_clinical[0]]

        if treatments is not None:
            X = X.join(treatments) if X is not None else treatments

        genes = genes.loc[:, self.selected_genes[0]]

        X = X.join(genes, how='inner').fillna(0).values if X is not None else genes.fillna(0)

        X = np.maximum(0, np.minimum(1, self.scaler.transform(X)))

        assert X.shape[1] == self.fitted_shape[1], \
            'new data should have same number of features used to fit model'

        if self.predictor == 'lightgbm':
            result = self.model.predict(X)
        else:
            result = self.model.predict_proba(X)

        if len(result.shape) > 1:
            result = result[:, -1]

        return result
