from correlation import select_genes

import pandas as pd
import numpy as np


class SelectMarker(object):

    @staticmethod
    def select_k_top_markers(entropies):

        if isinstance(entropies, pd.Series):
            entropies = entropies.values.reshape([-1])

        elif isinstance(entropies, list):
            entropies = np.array(entropies)

        entropies = entropies.reshape([-1])

        def q25(e):
            return np.quantile(e, q=.25)

        def q50(e):
            return np.quantile(e, q=.50)

        def q75(e):
            return np.quantile(e, q=.75)

        def std(e):
            return np.std(e, ddof=1)

        betas = np.array([
            -0.8848630975791922, 4.600451633673629, 3.615872144716864, 1.1838667374310157,
            -0.7000817701614678, -0.018821646628496263, -0.2313025930161946, 0.2103729053368249,
            -4.060631004415739, 0.14208357407079072])

        params = np.array([1.0] + [op(entropies) for op in [
            np.mean, std, np.min, np.max, np.sum, q25, q50, q75, len]])

        return int(round(len(entropies) * params.dot(betas), 0))

    @staticmethod
    def select_markers(markers, outcome, threshold=0.0025):
        """
        """

        assert isinstance(markers, pd.DataFrame)

        selected_markers = select_genes(markers, outcome, threshold=threshold)

        return selected_markers
