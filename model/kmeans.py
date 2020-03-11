from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_samples, silhouette_score
import pandas as pd
import numpy as np
import pickle

class OptimizedKMeans(object):
    
    def __init__(self, min_n_clusters=2, max_n_clusters=None, min_n_observations=1, early_stopping_rounds=5, random_state=None, verbose=0, optimization_output_path=None):
        
        assert isinstance(min_n_clusters, int) and min_n_clusters > 1, "min_n_cluster should be an integer greater than one"
        
        assert early_stopping_rounds is None or (isinstance(early_stopping_rounds, int) and early_stopping_rounds > 0), "early_stopping_rounds should be None or integer greater than zero"
        
        assert random_state is None or isinstance(random_state, int), "random_state should be None or integer"
        
        assert isinstance(verbose, int) and verbose >= 0, "verbose should be interger greater than or equals to zero"
        
        assert min_n_observations > 0, 'min_n_observations should be interger greater than zero'
        
        self.min_n_observations = min_n_observations
        
        self.opt_metric = -np.inf
        
        self.min_n_clusters = min_n_clusters
        
        self.max_n_clusters = max_n_clusters
        
        self.early_stopping_rounds = early_stopping_rounds
        
        self.random_state = random_state
        
        self.model = None
        
        self.verbose = verbose
        
        self.n_clusters = None
        
        self.optimization_output_path = optimization_output_path
    
    def fit(self, x):
        
        no_improved_rounds = 0
        
        n_clusters = self.min_n_clusters
        
        if self.optimization_output_path is not None:
            opt_output = {'dimensions': [], 'metric': []}
        
        while True:
            
            if self.verbose > 0:
                print('fitting model for {} clusters'.format(n_clusters))
            
            clusterer = KMeans(n_clusters=n_clusters, random_state=self.random_state)
    
            cluster_labels = clusterer.fit_predict(x)

            #
            # The silhouette_score gives the average value for all the samples.
            # This gives a perspective into the density and separation of the formed
            # clusters
            #
            try:
                silhouette_avg = silhouette_score(x, cluster_labels)
            except ValueError:
                break

            #
            # Compute the silhouette scores for each sample
            #
            sample_silhouette_values = silhouette_samples(x, cluster_labels)

            s_values, c_sizes = [], []

            for i in range(n_clusters):
                #
                # Aggregate the silhouette scores for samples belonging to
                # cluster i, and sort them
                #
                ith_cluster_silhouette_values = \
                    sample_silhouette_values[cluster_labels == i]

                ith_cluster_silhouette_values.sort()

                size_cluster_i = ith_cluster_silhouette_values.shape[0]

                s_values += [np.mean(ith_cluster_silhouette_values)]

                c_sizes += [len(ith_cluster_silhouette_values)]

            s_values = np.array(s_values)

            c_sizes = np.array(c_sizes)

            sd_sizes = np.std(c_sizes)
            
            try:
                if np.min(c_sizes) >= self.min_n_observations:
                    opt_metric = (np.mean(ith_cluster_silhouette_values) - np.std(ith_cluster_silhouette_values)) / sd_sizes;
                else:
                    opt_metric = 0.0
                
            except:
                opt_metric = 0.0
            
            if np.isnan(opt_metric):
                opt_metric = 0.0
            
            if self.verbose > 0:
                print("optimization metric of {}".format(opt_metric))
            
            if self.optimization_output_path is not None:
                
                opt_output['dimensions'].append(clusterer.cluster_centers_)
                
                opt_output['metric'].append(opt_metric)
                
            if opt_metric > self.opt_metric:
                
                if self.verbose > 0:
                    print('*** new best model ***')
                
                self.opt_metric = opt_metric
                
                self.model = clusterer      
                
                self.n_clusters = n_clusters
                
                no_improved_rounds = 0
            
            if self.early_stopping_rounds is not None and no_improved_rounds >= self.early_stopping_rounds:
                
                if (self.verbose > 0):
                    print('stopping optimization: no improvements for {} rounds'.format(no_improved_rounds))
                
                break
                
            elif self.max_n_clusters is not None and n_clusters > self.max_n_clusters:
                
                if self.verbose > 0:
                    print('stopping optmization: max number of clusters {} reached'.format(self.max_n_clusters))
                
                break
                
            else:
                no_improved_rounds += 1
            
            n_clusters += 1
        
        if self.verbose > 0:
            print('\n**********************************************************************')
            print('selected model with score {} and {} clusters'.format(self.opt_metric, self.n_clusters))
            print('**********************************************************************\n')
            
        if self.optimization_output_path is not None:
            with open(self.optimization_output_path, 'wb') as file:
                pickle.dump(opt_output, file)
    
    def transform(self, x):
        return self.model.transform(x)
    
    def predict(self, x):
        return self.model.predict(x)
    
    def fit_transform(self, x):
        self.fit(x)
        return self.transform(x)
    
    def fit_predict(self, x):
        self.fit(x)
        return self.predict(x)
