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

from model import OptimizedKMeans
from sklearn.metrics import auc
from sklearn.decomposition import PCA
from collections import Counter

import pandas as pd
import numpy as np


class GeneticProfiling(object):
    """

    """

    def __init__(self, early_stopping_rounds=10, random_state=None, verbose=0, optimization_output_file_path=None):
        
        self.model = None
        
        self.early_stopping_rounds = early_stopping_rounds
        
        self.verbose = verbose
        
        self.optimization_output_file_path = optimization_output_file_path
        
        self.random_state = random_state

    def fit(self, x):
        
        self.model = OptimizedKMeans(early_stopping_rounds=self.early_stopping_rounds,
                                     verbose=self.verbose, min_n_observations=1,
                                     optimization_output_path=self.optimization_output_file_path,
                                     random_state=self.random_state)
        self.model.fit(x)
    
    def predict(self, x):
        return self.model.predict(x)        
    
    def transform(self, x):
        profiling = self.model.transform(x)
        return profiling / np.sum(profiling, axis=1).reshape([-1, 1])
    
    def fit_transform(self, x):
        self.fit(x)
        return self.transform(x)
    
    def fit_predict(self, x):
        self.fit(x)
        return self.predict(x)


class GeneticClustering(object):
    """

    """
    
    def __init__(self, early_stopping_rounds=10, random_state=None, verbose=0, optimization_output_file_path=None):
        
        self.model = None
        
        self.early_stopping_rounds = early_stopping_rounds
        
        self.verbose = verbose
        
        self.optimization_output_file_path = optimization_output_file_path
        
        self.random_state = random_state
        
        self.clusters = None
        
        self.cluster_mapping = None

        self.pca = None

    def fit(self, x):
        """

        :param x:
        :return:
        """
        
        self.model = OptimizedKMeans(early_stopping_rounds=self.early_stopping_rounds,
                                     verbose=self.verbose,
                                     min_n_observations=1,
                                     max_n_clusters=x.shape[0] - 1,
                                     optimization_output_path=self.optimization_output_file_path,
                                     random_state=self.random_state)

        self.pca = PCA(n_components=max(2, int(.1 * x.shape[1])), random_state=self.random_state)

        x = self.pca.fit_transform(x.T)

        self.cluster_mapping = self.model.fit_predict(x)
        
        self.clusters = sorted(np.unique(self.cluster_mapping))
        
        for k, v in Counter(self.cluster_mapping).items():
            if v < 2:
                self.clusters.remove(k)
        
        self.distances = np.min(self.model.transform(x), axis=1)
        
        if self.verbose is not None and self.verbose > 0:
            print('gene expression clustering model contains {} clusters'.format(len(self.clusters)))
        
    def transform(self, x, method='mean'):
        """

        :param x:
        :param method:
        :return:
        """
        
        if isinstance(x, pd.DataFrame):
            x = x.values         
        
        result = []
        
        for c in self.clusters:
            
            selected_genes = [i for i, j in enumerate(self.cluster_mapping == c) if j]
            
            if self.verbose is not None and self.verbose > 1:
                print('{} selected genes for cluster {}'.format(len(selected_genes), c))
            
            x_current = x[:, selected_genes]

            if method == 'mean':
                result.append(np.mean(x_current, axis=1))

            else:

                x_current_distance = self.distances[selected_genes]

                # sorting values by their distances from centroid
                x_current = x_current[:, np.argsort(x_current_distance)]

                current_result = []

                # computing auc of a sorted list of values
                # Is it make sense? I do not think so. Area will not change.
                for __x__ in x_current:
                    current_result.append(auc(np.cumsum(np.sort(x_current_distance)) / np.sum(x_current_distance),
                                              np.cumsum(__x__) / np.sum(__x__)))

                result.append(current_result)
        
        return np.array(result).T
