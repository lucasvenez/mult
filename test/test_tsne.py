import unittest
import preprocessing
import util
import pandas as pd


class TestTSNE(unittest.TestCase):

    def setUp(self):
        self.tsne = preprocessing.tSNE(n_components=2)

    def export_small_dataset(self):

        df = util.read_genomic_data('../input/iss_status.tsv')

        cols = [c for c in df.columns if 'log' not in c.lower()]

        df = df[cols]

        df['Y'] = df['D'] + df['NR_NPD'] * 2 + df['NR_PD'] * 3 + df['RS'] * 4 + df['R_PD'] * 5

        for column in ['D', 'NR_NPD', 'NR_PD', 'RS', 'R_PD']:
            del df[column]

        df.to_csv('../input/iss_status_without_log.tsv', sep='\t', index=True)

    def test_tsne(self):

        df = pd.read_table('../output/iss_status_without_log_valid.tsv', sep='\t').set_index('PUBLIC_ID')

        y = df['Y'].as_matrix()

        del df['Y']

        small_dimension = self.tsne.optimize_predict(df.as_matrix())

        self.tsne.plot(small_dimension, y)
