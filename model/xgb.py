import xgboost as xgb
import random


class ExtremeGradientBoosting(object):
    """

    """
    def __init__(self, eta=0.02, max_depth=7, subsample=1., colsample_bytree=1.,
                 objective='binary:logistic', max_iterations=5000, eval_metric='auc',
                 early_stopping_rounds=100, num_class=1,
                 silent=True, seed=random.randint(0, 1000000), verbose_eval=None):

        self.__model = None

        self.__params = {'eta': eta,
                         'max_depth': max_depth,
                         'subsample': subsample,
                         'colsample_bytree': colsample_bytree,
                         'objective': objective,
                         'eval_metric': eval_metric,
                         'seed': seed,
                         'silent': silent,
                         'num_class': num_class}

        self.early_stopping_rounds = early_stopping_rounds

        self.verbose_eval = verbose_eval

        self.max_iterations = max_iterations

        self.trained = False

    def fit(self, train_x, train_y, test_x=None, test_y=None, valid_x=None, valid_y=None):

        d_train = xgb.DMatrix(train_x, train_y[:, 0])

        if valid_x is None or valid_y is None:
            valid_x, valid_y = train_x, train_y

        watchlist = [(xgb.DMatrix(train_x, train_y[:,0]), 'train')]

        if valid_x is not None and valid_y is not None:
            watchlist += [(xgb.DMatrix(valid_x, valid_y), 'valid')]

        if test_x is not None and test_y is not None:
            watchlist += [(xgb.DMatrix(test_x, test_y), 'test')]

        self.__model = xgb.train(self.__params, d_train, self.max_iterations, watchlist,
                                 verbose_eval=self.verbose_eval, early_stopping_rounds=self.early_stopping_rounds)

        self.trained = True

        #return self.__model.evals_result()

    def predict(self, x):

        if self.trained and self.__model is not None:
            return self.__model.predict(xgb.DMatrix(x))