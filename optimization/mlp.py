from optimization import BayesianOptimizer

from skopt.space import Real, Integer, Categorical
from skopt.utils import use_named_args

from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score

from sklearn.neural_network import MLPClassifier

import numpy as np

import warnings
warnings.filterwarnings("ignore")


class MLPOptimizer(BayesianOptimizer):

    def optimize(self, x, y):
        """
        Description of each optimized hyperparameter. Checkout original description at
        https://scikit-learn.org/stable/modules/generated/sklearn.neural_network.MLPClassifier.html.
        Accessed on 28/06/2020 at 16:05:37.

        hidden_layer_sizes: tuple, length = n_layers - 2, default=(100,)
            The ith element represents the number of neurons in the ith hidden layer.

        activation{‘identity’, ‘logistic’, ‘tanh’, ‘relu’}, default=’relu’
            Activation function for the hidden layer.

            ‘identity’, no-op activation, useful to implement linear bottleneck, returns f(x) = x

            ‘logistic’, the logistic sigmoid function, returns f(x) = 1 / (1 + exp(-x)).

            ‘tanh’, the hyperbolic tan function, returns f(x) = tanh(x).

            ‘relu’, the rectified linear unit function, returns f(x) = max(0, x)

        solver{‘lbfgs’, ‘sgd’, ‘adam’}, default=’adam’
            The solver for weight optimization.

            ‘lbfgs’ is an optimizer in the family of quasi-Newton methods.

            ‘sgd’ refers to stochastic gradient descent.

            ‘adam’ refers to a stochastic gradient-based optimizer proposed by Kingma, Diederik, and Jimmy Ba

            Note: The default solver ‘adam’ works pretty well on relatively large datasets
            (with thousands of training samples or more) in terms of both training time and validation score.
            For small datasets, however, ‘lbfgs’ can converge faster and perform better.

        alpha: float, default=0.0001
            L2 penalty (regularization term) parameter.

        learning_rate{‘constant’, ‘invscaling’, ‘adaptive’}, default=’constant’
            Learning rate schedule for weight updates.

            ‘constant’ is a constant learning rate given by ‘learning_rate_init’.

            ‘invscaling’ gradually decreases the learning rate at each time step ‘t’ using an inverse scaling exponent
            of ‘power_t’. effective_learning_rate = learning_rate_init / pow(t, power_t)

            ‘adaptive’ keeps the learning rate constant to ‘learning_rate_init’ as long as training loss keeps decreasing.
            Each time two consecutive epochs fail to decrease training loss by at least tol, or fail to increase validation
            score by at least tol if ‘early_stopping’ is on, the current learning rate is divided by 5.

            Only used when solver='sgd'.

        learning_rate_init: double, default=0.001
            The initial learning rate used. It controls the step-size in updating the weights.
            Only used when solver=’sgd’ or ‘adam’.

        power_t: double, default=0.5
            The exponent for inverse scaling learning rate. It is used in updating effective learning rate
            when the learning_rate is set to ‘invscaling’. Only used when solver=’sgd’.

        max_iter: int, default=200
            Maximum number of iterations. The solver iterates until convergence (determined by ‘tol’) or this
            number of iterations. For stochastic solvers (‘sgd’, ‘adam’), note that this determines the number of epochs
            (how many times each data point will be used), not the number of gradient steps.

        tol:float, default=1e-4
            Tolerance for the optimization. When the loss or score is not improving by at least tol for n_iter_no_change
            consecutive iterations, unless learning_rate is set to ‘adaptive’, convergence is considered to be reached
            and training stops.

        momentum: float, default=0.9
            Momentum for gradient descent update. Should be between 0 and 1. Only used when solver=’sgd’.

        beta_1: float, default=0.9
            Exponential decay rate for estimates of first moment vector in adam, should be in [0, 1).
            Only used when solver=’adam’

        beta_2: float, default=0.999
            Exponential decay rate for estimates of second moment vector in adam, should be in [0, 1).
            Only used when solver=’adam’

        epsilon: float, default=1e-8
            Value for numerical stability in adam. Only used when solver=’adam’

        early_stopping: bool, default=False
            Whether to use early stopping to terminate training when validation score is not improving.
            If set to true, it will automatically set aside 10% of training data as validation and terminate
            training when validation score is not improving by at least tol for n_iter_no_change consecutive epochs.
            The split is stratified, except in a multilabel setting. Only effective when solver=’sgd’ or ‘adam’

        validation_fraction: float, default=0.1
            The proportion of training data to set aside as validation set for early stopping. Must be between 0 and 1.
            Only used if early_stopping is True

        n_iter_no_change: int, default=10
            Maximum number of epochs to not meet tol improvement. Only effective when solver=’sgd’ or ‘adam’

            New in version 0.20.

        verbose: bool, default=False
            Whether to print progress messages to stdout.

        warm_start: bool, default=False
            When set to True, reuse the solution of the previous call to fit as initialization, otherwise,
            just erase the previous solution. See the Glossary.

        shuffle: bool, default=True
            Whether to shuffle samples in each iteration. Only used when solver=’sgd’ or ‘adam’.

        random_state: int, RandomState instance, default=None
            Determines random number generation for weights and bias initialization, train-test split if early stopping
            is used, and batch sampling when solver=’sgd’ or ‘adam’. Pass an int for reproducible results
            across multiple function calls. See Glossary.
        """

        space = [Integer(10, 200, name='hidden_layer_sizes'),
                 Integer(1, 3, name='n_hidden_layers'),
                 Categorical(['identity', 'logistic', 'tanh', 'relu'], name='activation'),
                 Categorical(['lbfgs', 'sgd', 'adam'], name='solver'),
                 Real(1e-4, 1e-1, prior='log-uniform', name='alpha'),
                 Categorical(['constant', 'invscaling', 'adaptive'], name='learning_rate'),
                 Real(1e-5, 1e-1, prior='log-uniform', name='learning_rate_init'),
                 Real(1e-5, 1e-1, prior='log-uniform', name='power_t'),
                 Real(1e-4, 1e-1, prior='log-uniform', name='tol'),
                 Real(1e-4, 9e-1, prior='log-uniform', name='momentum'),
                 Real(1e-4, 9e-1, prior='log-uniform', name='beta_1'),
                 Real(1e-4, 9e-1, prior='log-uniform', name='beta_2'),
                 Real(1e-8, 1e-2, prior='log-uniform', name='epsilon')]

        @use_named_args(space)
        def objective(
                hidden_layer_sizes,
                n_hidden_layers,
                activation,
                solver,
                alpha,
                learning_rate,
                learning_rate_init,
                power_t,
                tol,
                momentum,
                beta_1,
                beta_2,
                epsilon):

            try:

                scores, params = [], {
                    'hidden_layer_sizes': (hidden_layer_sizes for _ in range(n_hidden_layers)),
                    'activation': activation,
                    'solver': solver,
                    'alpha': alpha,
                    'learning_rate': learning_rate,
                    'learning_rate_init': learning_rate_init,
                    'power_t': power_t,
                    'tol': tol,
                    'momentum': momentum,
                    'beta_1': beta_1,
                    'beta_2': beta_2,
                    'epsilon': epsilon,
                    'n_iter_no_change': 100,

                    'early_stopping': super().early_stopping_rounds is not None and super().early_stopping_rounds > 1,
                    'validation_fraction': 1/3.,

                    'max_iter': 500,
                    'random_state': self.random_state,
                    'n_jobs': self.n_jobs,
                    'verbose': self.verbose}

                params.update(super().fixed_parameters)

                skf = StratifiedKFold(
                    self.n_folds, shuffle=self.shuffle, random_state=self.random_state)

                for train_index, valid_index in skf.split(x, y):

                    x_train, y_train = x[train_index, :], y[train_index]
                    x_valid, y_valid = x[valid_index, :], y[valid_index]

                    mlp = MLPClassifier(**params)

                    model = mlp.fit(x_train, y_train)

                    y_hat = model.predict_proba(x_valid)

                    scores.append(roc_auc_score(y_valid, y_hat))

                return -np.mean([s for s in scores if s is not None])

            except:
                return 0.0

        return super().execute_optimization(objective, space)
