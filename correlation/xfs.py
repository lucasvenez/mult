from sklearn.model_selection import StratifiedKFold
from correlation import feature_selection
import pandas as pd
import numpy as np


class XFS:
    """

    """

    def __init__(self, shuffle=False, random_state=None):

        self.selected_features = []
        self.pvalues = []
        self.entropies = []

        self.shuffle = shuffle
        self.random_state = random_state

    def select(self, x, y, cross_fraction=.25, acceptable_noise=.05, names=None, index=None):
        """

        :param x:
        :param y:
        :param cross_fraction:
        :param acceptable_noise:
        :param names:
        :param index:
        :return:
        """

        assert 0 < cross_fraction < 1.0, \
            'cross fraction parameter should be between (exclusive) 0 and 1'

        cross_fraction = 1. - cross_fraction

        assert 0 <= acceptable_noise <= 1, \
            'acceptable noise parameter should be between (inclusive) 0 and 1'

        assert isinstance(x, (np.ndarray, pd.DataFrame))

        assert isinstance(y, (pd.Series, pd.DataFrame, np.ndarray, list)), \
            'y should be either an np.array, pd.Series, pd.DataFrame, or list'

        assert len(x.shape) == 2, \
            'x should be a two dimensional structure (np.array or pd.DataFrame)'

        if names is not None:

            assert isinstance(names, list), 'names parameter should be a list'

            assert len(names) == x.shape[1], 'names length should be equals to the number of columns in x'

            if index is not None:
                assert isinstance(index, list), 'index parameter should be a list'
            else:
                index = list(range(0, x.shape[0]))

            x = pd.DataFrame(x, columns=names, index=index)

        if not isinstance(y, pd.DataFrame):
            y = pd.DataFrame({'y': y}, index=x.index)

        assert x.shape[0] == y.shape[0], 'x and y should have the same number of rows'

        # splitting data
        kfold = StratifiedKFold(n_splits=max(2, int(1./cross_fraction)),
                                shuffle=self.shuffle, random_state=self.random_state)

        iterations = {}

        for iteration, (index, _) in enumerate(kfold.split(x, y)):

            # x_train, y_train = x.iloc[train_index, :], y.iloc[train_index, 0]
            x_valid, y_valid = x.iloc[index, :], y.iloc[index, 0]

            sf, pv, en = feature_selection(x_valid, y_valid, threshold=acceptable_noise)

            iterations[iteration] = {
                'selected_features': sf,
                'pvalues': pv,
                'entropies': en}

        selected_features = None

        for _, i in iterations.items():
            selected_features = set(i['selected_features']) if selected_features is None \
                else selected_features.intersection(i['selected_features'])

        pvalues, entropies = {}, {}

        for s in selected_features:

            if s not in pvalues:
                pvalues[s] = []
                entropies[s] = []

            for _, i in iterations.items():
                index = i['selected_features'].index(s)
                pvalues[s].append(i['pvalues'][index])
                entropies[s].append(i['entropies'][index])

            pvalues[s] = np.mean(pvalues[s])
            entropies[s] = np.mean(entropies[s])

        for s in sorted(selected_features):
            self.selected_features.append(s)
            self.pvalues.append(pvalues[s])
            self.entropies.append(entropies[s])

        s = [s for _, s in sorted(zip(self.pvalues, self.selected_features))]
        e = [e for _, e in sorted(zip(self.pvalues, self.entropies))]

        self.selected_features = s
        self.entropies = e
        self.pvalues = sorted(self.pvalues)


def xfs(x, y, cross_fraction=0.25, acceptable_noise=0.05, shuffle=True, random_state=None):

    _xfs = XFS(shuffle, random_state)

    _xfs.select(x, y, cross_fraction, acceptable_noise)

    return _xfs.selected_features, _xfs.pvalues, _xfs.entropies
