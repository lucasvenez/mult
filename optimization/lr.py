from optimization import BayesianOptimizer

from skopt.space import Real, Categorical
from skopt.utils import use_named_args
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score

from sklearn.linear_model import LogisticRegression

import numpy as np

import warnings
warnings.filterwarnings("ignore")


class LogisticRegressionOptimizer(BayesianOptimizer):

    def optimize(self, x, y):
        """
        Description of each optimized hyperparameter. Checkout original description at
        https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LogisticRegression.html.
        Accessed on 28/06/2020 at 15:07:01.

        penalty{‘l1’, ‘l2’, ‘elasticnet’, ‘none’}, default=’l2’
            Used to specify the norm used in the penalization. The ‘newton-cg’, ‘sag’ and ‘lbfgs’
             support only l2 penalties. ‘elasticnet’ is only supported by the ‘saga’ solver. If ‘none’
             (not supported by the liblinear solver), no regularization is applied.

        dual: bool, default=False
            Dual or primal formulation. Dual formulation is only implemented for l2 penalty with liblinear solver.
            Prefer dual=False when n_samples > n_features.

        tol: float, default=1e-4
            Tolerance for stopping criteria.

        C: float, default=1.0
            Inverse of regularization strength; must be a positive float. Like in support vector machines,
            smaller values specify stronger regularization.

        fit_intercept: bool, default=True
            Specifies if a constant (a.k.a. bias or intercept) should be added to the decision function.

        intercept_scaling: float, default=1
            Useful only when the solver ‘liblinear’ is used and self.fit_intercept is set to True. In this case,
            x becomes [x, self.intercept_scaling], i.e. a “synthetic” feature with constant value equal to
            intercept_scaling is appended to the instance vector. The intercept becomes
            intercept_scaling * synthetic_feature_weight.

            Note! the synthetic feature weight is subject to l1/l2 regularization as all other features.
            To lessen the effect of regularization on synthetic feature weight (and therefore on the intercept)
            intercept_scaling has to be increased.

        solver{‘newton-cg’, ‘lbfgs’, ‘liblinear’, ‘sag’, ‘saga’}, default=’lbfgs’
            Algorithm to use in the optimization problem.

            For small datasets, ‘liblinear’ is a good choice, whereas ‘sag’ and ‘saga’ are faster for large ones.

            For multiclass problems, only ‘newton-cg’, ‘sag’, ‘saga’ and ‘lbfgs’ handle multinomial loss; ‘liblinear’ is limited to one-versus-rest schemes.

            ‘newton-cg’, ‘lbfgs’, ‘sag’ and ‘saga’ handle L2 or no penalty

            ‘liblinear’ and ‘saga’ also handle L1 penalty

            ‘saga’ also supports ‘elasticnet’ penalty

            ‘liblinear’ does not support setting penalty='none'

            Note that ‘sag’ and ‘saga’ fast convergence is only guaranteed on features with
            approximately the same scale. You can preprocess the data with a scaler from sklearn.preprocessing.

            New in version 0.17: Stochastic Average Gradient descent solver.

            New in version 0.19: SAGA solve

        :param x:
        :param y:
        :param num_boost_round:
        :param early_stopping_rounds:
        :return:
        """

        space = [
            Categorical(['l1', 'l2', 'elasticnet', 'none'], name='penalty'),
            Categorical([False, True], name='dual'),
            Real(1e-5, 1.00, prior='log-uniform', name='tol'),
            Real(1e-5, 1.00, prior='log-uniform', name='C'),
            Categorical([False, True], name='fit_intercept'),
            Real(1e-5, 1.00, prior='log-uniform', name='intercept_scaling'),
            Categorical(['newton-cg', 'lbfgs', 'liblinear', 'sag', 'saga'], name='solver')]

        @use_named_args(space)
        def objective(
                penalty,
                dual,
                tol,
                C,
                fit_intercept,
                intercept_scaling,
                solver):

            try:

                scores, params = [], {
                    'penalty': penalty,
                    'dual': dual,
                    'tol': tol,
                    'C': C,
                    'fit_intercept': fit_intercept,
                    'intercept_scaling': intercept_scaling,
                    'solver': solver,

                    'random_state': self.random_state,
                    'n_jobs': self.n_jobs,
                    'verbose': self.verbose}

                params.update(super().fixed_parameters)

                skf = StratifiedKFold(
                    self.n_folds, shuffle=self.shuffle, random_state=self.random_state)

                for train_index, valid_index in skf.split(x, y):

                    x_train, y_train = x[train_index, :], y[train_index]
                    x_valid, y_valid = x[valid_index, :], y[valid_index]

                    lr = LogisticRegression(**params)

                    model = lr.fit(x_train, y_train)

                    y_hat = model.predict_proba(x_valid)

                    scores.append(roc_auc_score(y_valid, y_hat))

                return -np.mean([s for s in scores if s is not None])

            except:
                return 0.0

        return super().execute_optimization(objective, space)
