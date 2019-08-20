from bayes_opt import BayesianOptimization
from sklearn.model_selection import cross_val_score

import lightgbm as lgb

def bayesOpt(train_x, train_y, verbose=0):

    


    
    lgbBO = BayesianOptimization(lgb_evaluate, 
                {'numLeaves':  (5, 50), 
                 'maxDepth': (2, 63), 
                 'scaleWeight': (1, 10000), 
                 'minChildWeight': (0.01, 70), 
                 'subsample': (0.4, 1), 
                 'colSam': (0.4, 1)}, verbose=2)

    lgbBO.maximize(init_points=5, n_iter=5)

    if verbose > 0:
        print(lgbBO.res['max'])