from lightgbm import LGBMModel, Dataset
from optimization import BayesianOptimizer


class SMLA(object):
    
    def __init__(self,
                 optimizer_callable, model_callable,
                 optimizer_default_params={}, model_default_params={},
                 verbose=-1,
                 random_state=None,
                 use_gpu=False):

        assert isinstance(optimizer_default_params, dict)
        assert isinstance(model_default_params, dict)
        assert issubclass(optimizer_callable, BayesianOptimizer)

        #
        self.model_callable = model_callable
        self.model_default_params = model_default_params

        #
        self.optimizer_callable = optimizer_callable
        self.optimizer_default_params = optimizer_default_params

        #
        self.model = None
        self.fitted_shape = None

        #
        self.random_state = random_state
        self.verbose = verbose
        self.use_gpu = use_gpu
    
    def fit(self, x, y, x_valid=None, y_valid=None, early_stopping_rounds=None):

        self.fitted_shape = x.shape

        optimizer = self.optimizer_callable(**self.optimizer_default_params)

        params = optimizer.optimize(x, y)

        self.model = self.model_callable(**params)

        if x_valid is not None and y_valid is not None:
            if issubclass(self.model_callable, LGBMModel):
                self.model.fit(x, y,
                               eval_set=Dataset(x_valid, y_valid),
                               early_stopping_rounds=early_stopping_rounds,
                               verbose=self.verbose)
            else:
                raise NotImplementedError(str(self.model_callable))

        else:
            self.model.fit(x, y)

    def predict(self, x):

        assert x.shape[1] == self.fitted_shape[1], \
            'new data should have same number of features used to fit model'

        return self.model.predict(x)
