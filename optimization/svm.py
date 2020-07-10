from optimization import BayesianOptimizer

from skopt.space import Real, Integer, Categorical
from skopt.utils import use_named_args
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score

from sklearn.svm import SVC
import numpy as np

import warnings
warnings.filterwarnings("ignore")


def sigmoid(x, alpha=1.):
    return 1. / (1. + np.exp(-alpha * x))


class SVMOptimizer(BayesianOptimizer):

    def optimize(self, x, y):

        space = [
            Real(1e-6, 1e+6, prior='log-uniform', name='C'),
            Real(1e-6, 1e+1, prior='log-uniform', name='gamma'),
            Integer(1, 8, name='degree'),
            Categorical(['linear', 'poly', 'rbf'], name='kernel')]

        @use_named_args(space)
        def objective(C, gamma, degree, kernel):
            try:
                scores = []

                params = {
                    'C': C,
                    'gamma': gamma,
                    'degree': degree,
                    'kernel': kernel,
                    'random_state': self.random_state}

                params.update(super().fixed_parameters)

                skf = StratifiedKFold(
                    self.n_folds, shuffle=self.shuffle, random_state=self.random_state)

                for train_index, valid_index in skf.split(x, y):

                    try:
                        x_train, y_train = x[train_index, :], y[train_index, 0]
                    except IndexError:
                        x_train, y_train = x[train_index, :], y[train_index].reshape(-1,)

                    try:
                        x_valid, y_valid = x[valid_index, :], y[valid_index, 0]
                    except IndexError:
                        x_valid, y_valid = x[valid_index, :], y[valid_index].reshape(-1,)

                    svm = SVC(**params)

                    svm.fit(x_train, y_train)

                    y_hat = sigmoid(svm.predict(x_valid))

                    scores.append(roc_auc_score(y_valid, y_hat))

                return -np.mean([s for s in scores if s is not None])

            except ValueError:
                return 0.0

        return super().execute_optimization(objective, space)
