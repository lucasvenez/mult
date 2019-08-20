from model import OptimizedKMeans
from collections import Counter

import numpy as np

class ClusteredStratifiedKFold(object):
    
    def __init__(self, n_splits=5, shuffle=False, early_stopping_rounds=5, random_state=None):
        
        self.n_splits = n_splits
        
        self.shuffle = shuffle
        
        self.random_state = random_state
        
        self.model = OptimizedKMeans(early_stopping_rounds=early_stopping_rounds, 
                                     min_n_observations=self.n_splits, random_state=self.random_state)
        
    def split(self, X, y):
        
        result = {}
        
        self.model.fit(X)
        
        clusters, distances = self.model.predict(X), np.min(np.abs(self.model.transform(X)), axis=1)
        
        for fold in range(self.n_splits):
            
            result[fold] = [[], []]
            
            for c in np.unique(clusters):

                bool_list = clusters == c

                y_current =  y[bool_list]

                count_per_response_value = list(Counter(y_current).values())
                
                if np.min(count_per_response_value) >= self.n_splits:
                    
                    train_index, valid_index = [], []
                    
                    for _y in np.unique(y_current):
                        
                        bool_list_y = [v and _y == y_ for i, (y_, v) in enumerate(zip(y, bool_list))]
                        
                        frac = int(np.sum(bool_list_y) / self.n_splits)
                        
                        k = 0
                        
                        for i, v in enumerate(bool_list_y):
                            
                            if v:
                            
                                if (k % self.n_splits == fold or (k > frac * self.n_splits and fold == self.n_splits - 1)):
                                    valid_index.append(i)
                            
                                k += 1
                        
                        train_index += [i for i, v in enumerate(bool_list_y) if v and i not in valid_index]

                else:
                    
                    frac = int(np.sum(bool_list) / self.n_splits)
                    
                    valid_index, k = [], 0
                        
                    for i, v in enumerate(bool_list):

                        if v:
                        
                            if (k % self.n_splits == fold or (k > frac * self.n_splits and fold == self.n_splits - 1)):
                                valid_index.append(i)
                            
                            k += 1
                            
                    train_index = [i for i, v in enumerate(bool_list) if v and i not in valid_index]
                
                result[fold][0] += train_index
                
                result[fold][1] += valid_index
                
        return list(result.values())