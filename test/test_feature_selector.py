from model import FeatureSelector

import pandas as pd
import numpy as np
import unittest


class FeatureSelectorTest(unittest.TestCase):
    """

    """

    def test_feature_selection(self):

        #
        #
        #
        genes = pd.read_csv('../data/gene_fpkm.txt', sep='\t')

        genes = genes.drop_duplicates(subset=['GENE_ID'], keep='first')

        genes = genes.rename(columns={'GENE_ID': 'MMRF'})

        genes = genes.set_index('MMRF')

        del genes['Location']

        genes = genes[[i for i in genes.columns if '_1_' in i]]

        genes.columns = [i.split('_')[1] for i in genes.columns]

        genes = genes.T

        genes.index = [int(i) for i in genes.index]

        #
        #
        #
        clinical_data = pd.read_csv('../data/iss_fish_therapy_response.csv', sep='\t')

        clinical_data['MMRF'] = [int(i.replace('MMRF', '')) for i in clinical_data['MMRF']]

        clinical_data = clinical_data.loc[clinical_data['Days-to-Progression'].notnull(), :]

        clinical_data = clinical_data.set_index('MMRF')

        #
        #
        #
        groups = {1: (['SCR'], ['CR', 'VGPR', 'PR', 'SD', 'PD']),
                  2: (['SCR', 'CR'], ['VGPR', 'PR', 'SD', 'PD']),
                  3: (['SCR', 'CR', 'VGPR'], ['PR', 'SD', 'PD']),
                  4: (['SCR', 'CR', 'VGPR', 'PR'], ['SD', 'PD']),
                  5: (['SCR', 'CR', 'VGPR', 'PR', 'SD'], ['PD']),
                  6: (['SCR'], ['CR', 'VGPR'], ['PR', 'SD', 'PD']),
                  7: (['SCR'], ['CR', 'VGPR', 'PR'], ['SD', 'PD'])}

        var_resp = ['Days-to-Progression']

        clinical_data['Greater18Months-Days-to-Progression'] = [int(i > 18 * 30) for i in
                                                                clinical_data['Days-to-Progression']]

        var_resp.append('Greater18Months-Days-to-Progression')

        for i in range(1, 6):
            var_resp.append('Group{}-From-Best-Response-FirstLine'.format(i))

            clinical_data[var_resp[-1]] = clinical_data['Best-Response-FirstLine'].apply(
                lambda x: 0 if x in groups[i][0] else 1)

        clinical_data['Best-Response-FirstLine-ID'] = \
            clinical_data['Best-Response-FirstLine'].map(
                {'SD': 1, 'VGPR': 2, 'CR': 3, 'PR': 3, 'SCR': 4, 'PD': 5, None: 0, np.nan: 0})

        var_resp.append('Best-Response-FirstLine-ID')

        clinical_data = clinical_data[var_resp].dropna()

        #
        #
        #
        fs = FeatureSelector()

        r = fs.fit_transform(genes, clinical_data.iloc[:, 1].dropna())

        self.assertTrue(len(r.columns) < len(genes.columns))
