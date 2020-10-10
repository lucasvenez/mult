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
import numpy as np

class ClusteredSelection(object):
    
    def __init__(self, n_splits=5, shuffle=False, early_stopping_rounds=5, random_state=None):
        
        self.n_splits = n_splits
        
        self.shuffle = shuffle
        
        self.random_state = random_state
        
        self.model = OptimizedKMeans(early_stopping_rounds=early_stopping_rounds, 
                                     min_n_observations=2, random_state=self.random_state)
        
    def select(self, X):
        
        result = []
        
        self.model.fit(X)
        
        clusters, distances = self.model.predict(X), np.sum(np.abs(self.model.transform(X)), axis=1)

        for fold in range(self.n_splits):
            
            for c in np.unique(clusters):

                bool_list = clusters == c
                    
                frac = int(np.sum(bool_list) / self.n_splits)
                    
                valid_index, k = [], 0
                        
                for i, v in enumerate(bool_list):

                    if v:
                        
                        if (k % self.n_splits == fold or (k > frac * self.n_splits and fold == self.n_splits - 1)):
                            valid_index.append(i)
                            
                        k += 1
                
                result += valid_index
            
            break
            
        return result