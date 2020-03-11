from sklearn.model_selection import cross_val_score
from skopt.space import Real, Integer, Categorical
from skopt.utils import use_named_args
from skopt import gp_minimize
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score

import lightgbm as lgb
import numpy as np

def lightgbm_optimizer(x, y, n_calls=50, random_state=None, shuffle=True, 
                       nfolds=2, num_boost_round=100000, early_stopping_rounds=100, 
                       verbose=-1):
    
    space  = [
        Real(1e-6, 1e-1, 'log-uniform', name='learning_rate'),
        Integer(4, 512, name='num_leaves'),
        Integer(4, 20, name='max_depth'),
        Real(0.01, 1.00, name='scale_pos_weight'),
        Real(0.01, 1.00, name='min_child_weight'),
        Real(0.10, 1.00, name='colsample_bytree'),
        Real(0.001, 100, 'log-uniform', name='min_split_gain'),
        Integer(1, 1000, name='min_child_samples'),
        Real(0.01, 0.99, name='subsample'),
        Integer(200000, 800000, name='bin_construct_sample_cnt')
    ]
    
    @use_named_args(space)
    def objective(
                  learning_rate,
                  num_leaves, 
                  max_depth, 
                  scale_pos_weight, 
                  min_child_weight, 
                  colsample_bytree, 
                  min_split_gain, 
                  min_child_samples, 
                  subsample,
                  bin_construct_sample_cnt
                 ):
        try:
            scores = []

            kf = StratifiedKFold(nfolds, shuffle=shuffle, random_state=random_state) 

            params = {
                'learning_rate': learning_rate,
                'num_leaves': int(num_leaves),
                'max_depth': int(max_depth),
                'scale_pos_weight': scale_pos_weight,
                'min_child_weight': min_child_weight,
                'colsample_bytree': colsample_bytree,
                'min_split_gain': min_split_gain,
                'min_child_samples': int(min_child_samples),
                'subsample': subsample,
                'bin_construct_sample_cnt': int(bin_construct_sample_cnt),
                
                'objective':'binary',
                'metric':'auc',
                'is_unbalance':False,
                'nthread': 48,   
                'device': 'gpu',
                'gpu_platform_id': 1,
                'gpu_device_id': 0,
                'verbose': -1,
                'random_state': random_state}

            kfold = StratifiedKFold(nfolds, shuffle=shuffle, random_state=random_state)

            for train_index, valid_index in kfold.split(x, y):
                x_train, y_train = x[train_index,:], y[train_index, 0]
                x_valid, y_valid = x[valid_index,:], y[valid_index, 0]

                lgb_train = lgb.Dataset(x_train, y_train, params={'verbose': -1})
                lgb_valid = lgb.Dataset(x_valid, y_valid, params={'verbose': -1})

                gbm = lgb.train(params, lgb_train, valid_sets=lgb_valid, 
                                num_boost_round=num_boost_round, verbose_eval=False,
                                early_stopping_rounds=early_stopping_rounds);

                y_hat = gbm.predict(x_valid, num_iteration=gbm.best_iteration)

                scores.append(roc_auc_score(y_valid, y_hat))

            return -np.mean([s for s in scores if s is not None])
        
        except ValueError:
            return 0.0
    
    return gp_minimize(objective, space, n_calls=n_calls, random_state=random_state, verbose=(verbose >= 0), n_jobs=-1)
