from model import OptimizedKMeans
from sklearn.metrics import auc
from collections import Counter

import pandas as pd
import numpy as np


class GeneticProfiling(object):
    '''
    
    '''
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
    
    def __init__(self, early_stopping_rounds=10, random_state=None, verbose=0, optimization_output_file_path=None):
        
        self.model = None
        
        self.early_stopping_rounds = early_stopping_rounds
        
        self.verbose = verbose
        
        self.optimization_output_file_path = optimization_output_file_path
        
        self.random_state = random_state
        
        self.clusters = None
        
        self.cluster_mapping = None

    def fit(self, x):
        
        self.model = OptimizedKMeans(early_stopping_rounds=self.early_stopping_rounds,
                                     verbose=self.verbose,
                                     min_n_observations=1,
                                     max_n_clusters=x.shape[0] - 1,
                                     optimization_output_path=self.optimization_output_file_path,
                                     random_state=self.random_state)
        
        self.cluster_mapping = self.model.fit_predict(x.T)
        
        self.clusters = sorted(np.unique(self.cluster_mapping))
        
        for k, v in Counter(self.cluster_mapping).items():
            if v < 2:
                self.clusters.remove(k)
        
        self.distances = np.min(self.model.transform(x.T), axis=1)
        
        if self.verbose is not None and self.verbose > 0:
            print('gene expression clustering model contains {} clusters'.format(len(self.clusters)))
        
    def transform(self, x):
        
        if isinstance(x, pd.DataFrame):
            x = x.values         
        
        result = []
        
        for c in self.clusters:
            
            selected_genes = [i for i, j in enumerate(self.cluster_mapping == c) if j]
            
            if self.verbose is not None and self.verbose > 1:
                print('{} selected genes for cluster {}'.format(len(selected_genes), c))
            
            x_current = x[:, selected_genes]
            
            x_current_distance = self.distances[selected_genes]

            x_current = x_current[:, np.argsort(x_current_distance)]
            
            current_result = []
            
            for __x__ in x_current:
                current_result.append(auc(np.cumsum(np.sort(x_current_distance)) / np.sum(x_current_distance),
                    np.cumsum(__x__) / np.sum(__x__)))
            
            result.append(current_result)
        
        return np.array(result).T