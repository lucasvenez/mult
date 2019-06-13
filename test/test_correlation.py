from sklearn.metrics import roc_auc_score
from sklearn.model_selection import KFold

import model
import unittest
import analysis
import pandas as pd
import os


class TestCorrelation(unittest.TestCase):

    def test_mic(self):

        genomic_data = None #util.read_genomic_data()

        results = analysis.compute_mic(genomic_data, dependent_variable_names=['D', 'NR_NPD', 'NR_PD', 'RS', 'R_PD'])

        results.to_csv('../output/correlation.tsv', sep='\t', index=True)

    def test_mic_complete_independent_var(self):

        genomic_data = pd.read_table('../output/iss_status_without_log_valid.tsv', sep='\t').set_index('PUBLIC_ID')

        results = analysis.compute_mic(genomic_data, dependent_variable_names=['Y'])

        results.to_csv('../output/correlation_onevar_valid.tsv', sep='\t', index=True)

    def test_mic_complete_independent_var_dummy(self):

        genomic_data = pd.read_table('../output/iss_status_without_log_valid.tsv', sep='\t').set_index('PUBLIC_ID')

        dummies = pd.get_dummies(genomic_data['Y'])
        dummies.rename({1: 'D', 2: 'NR_NPD', 3: 'NR_PD', 4: 'RS', 5: 'R_PD'}, inplace=True)
        del genomic_data['Y']

        genomic_data = pd.concat([genomic_data, dummies], axis=1)

        for _, dummy in dummies.iteritems():

            print('Calculating MIC for {}'.format(dummy.name))

            results = analysis.compute_mic(genomic_data, dependent_variable_names=[dummy.name])

            results.to_csv('../output/correlation_onevar_valid_{}.tsv'.format(dummy.name), sep='\t', index=True)

    def test_mic_per_therapy_and_independent_var_dummy(self):

        #
        # Generating dummies
        #
        genetic_data = pd.read_table('../input/iss_status_without_log_valid.tsv', sep='\t')

        genetic_data['PUBLIC_ID'] = genetic_data['PUBLIC_ID'].map(lambda x: x.replace('_', ''))

        genetic_data.set_index('PUBLIC_ID', inplace=True)

        dummies_iss = pd.get_dummies(genetic_data['Y'])

        dummies_iss.columns = ('D', 'NR_NPD', 'NR_PD', 'RS', 'R_PD')

        del genetic_data['Y']

        dummies = pd.concat([genetic_data, dummies_iss], axis=1)[dummies_iss.columns]

        cli_data = pd.read_table('../input/clin_status.tsv', sep='\t').set_index('ID')[
            ['reponse', 'responseA', 'responseB', 'therapyA', 'therapyB']]

        cli_data.index.rename('PUBLIC_ID', inplace=True)

        dummies_therapy = pd.get_dummies(cli_data['therapyA'])

        cli_data = pd.concat([cli_data, dummies_therapy], axis=1)

        del cli_data['therapyA']

        dummies_therapy_b = pd.get_dummies(cli_data['therapyB'])

        cli_data = pd.concat([cli_data, dummies_therapy_b], axis=1)

        del cli_data['therapyB']

        dummies_response_a = pd.get_dummies(cli_data['responseA'])

        cli_data = pd.concat([cli_data, dummies_response_a], axis=1)

        del cli_data['responseA']

        dummies = dummies.join(cli_data)

        del genetic_data['D_PT_iss']

        genetic_data = genetic_data.join(dummies)

        remove_col = []

        for index, col in genetic_data.iteritems():
            if col.dtype == 'object':
                remove_col += [col.name]

        for name in remove_col:
            del genetic_data[name]

        for therapy in dummies_therapy.columns:

            print('Correlation {}'.format(therapy))

            result = None

            for dep_var in dummies_response_a.columns:

                filter_ = dummies[therapy] == 1

                tmp = analysis.compute_mic(genetic_data[filter_], dependent_variable_names=[dep_var])

                tmp.set_index('IND_VAR', inplace=True)

                del tmp['DEP_VAR']

                tmp.columns = [dep_var]

                if result is None:
                    result = tmp
                else:
                    result = result.join(tmp)

                print(result.head())

            result.to_csv('../output/mic_response_{}.tsv'.format(therapy), sep='\t')

    def test_mic_per_therapy_and_independent_var_dummy_with_groups(self):

        #
        # Generating dummies
        #

        genetic_data = pd.read_table('../input/iss_status_without_log_valid.tsv', sep='\t')

        genetic_data['PUBLIC_ID'] = genetic_data['PUBLIC_ID'].map(lambda x: x.replace('_', ''))

        genetic_data = genetic_data.rename(index=str, columns={'PUBLIC_ID': 'ID'})

        genetic_data.set_index('ID', inplace=True)

        dummies_iss = pd.get_dummies(genetic_data['Y'])

        dummies_iss.columns = ('D', 'NR_NPD', 'NR_PD', 'RS', 'R_PD')

        del genetic_data['Y']

        dummies = genetic_data.join(dummies_iss)[dummies_iss.columns]

        cli_data = pd.read_table('../input/clin_status.tsv', sep='\t').set_index('ID')

        cli_data = genetic_data.join(cli_data)[cli_data.columns]

        dummies_therapy = pd.get_dummies(cli_data['therapyA'])

        cli_data = cli_data.join(dummies_therapy)

        dummies_therapy_b = pd.get_dummies(cli_data['therapyB'])

        cli_data = cli_data.join(dummies_therapy_b)

        dummies_response_a = pd.get_dummies(cli_data['responseA'])

        dummies_response_a.columns = ['CR', 'PR', 'PD', 'SD', 'SCR', 'VGPR']

        cli_data = cli_data.join(dummies_response_a)

        groups_response_a = self.generate_groups_from_response(dummies_response_a)

        cli_data = cli_data.join(groups_response_a)

        cli_data.to_csv('../output/cli_data_with_groups.tsv', sep='\t')

        dummies = dummies.join(cli_data)

        del genetic_data['D_PT_iss']

        genetic_data = genetic_data.join(dummies)

        if not os.path.exists('../output/full_genetic_data_with_groups.tsv'):
            genetic_data.to_csv('../output/full_genetic_data_with_groups.tsv', sep='\t', index=False)

        remove_col = []

        for index, col in genetic_data.iteritems():
            if col.dtype == 'object':
                remove_col += [col.name]

        for name in remove_col:
            del genetic_data[name]

        for therapy in dummies_therapy.columns:

            functions = [
                #analysis.compute_kruskal, analysis.compute_mine, analysis.compute_wilcox,
                         analysis.compute_distcorr]
                #analysis.compute_pearson]

            for func in functions:

                func_name = func.__name__.split('_')[1]

                print('{} for {} therapy'.format(func_name, therapy))

                result = None

                for dep_var in groups_response_a.columns:

                    filter_ = dummies[therapy] == 1

                    tmp = func(genetic_data[filter_], dependent_variable_names=[dep_var])

                    if tmp is not None:
                        tmp.set_index('IND_VAR', inplace=True)

                        del tmp['DEP_VAR']

                        tmp.columns = [dep_var]

                        if result is None:
                            result = tmp
                        else:
                            result = result.join(tmp)

                if result is not None:

                    base_dir = '../output/group/{0}/'.format(func_name)

                    if not os.path.exists(base_dir):
                        os.makedirs(base_dir)

                    result.to_csv('{}/{}.tsv'.format(base_dir, therapy), sep='\t')

    def generate_groups_from_response(self, dummies):

        '''
        (SCR)            vs (CR,VGPR,PR,SD,PD)
        (SCR,CR)         vs (VGPR,PR,SD,PD)
        (SCR,CR,VGPR)    vs (PR,SO,PD)
        (SCR,CR,VGPR,PR) vs (SD,PD)
        (PR,SD,PD)       vs (CR,VGPR) vs SCR
        '''

        scr_vs_others = [int(i) for i in dummies['SCR'] == 1]

        scr_cr_vs_others = [int(i) for i in (dummies['CR'] == 1) | (dummies['SCR'] == 1)]

        scr_cr_vgpr_vs_others = [int(i) for i in (dummies['CR'] == 1) | (dummies['SCR'] == 1) | (dummies['VGPR'] == 1)]

        scr_cr_vgpr_pr_vs_others = [int(i) for i in (dummies['CR'] == 1) | (dummies['SCR'] == 1) |
                                               (dummies['VGPR'] == 1) | (dummies['PR'] == 1)]

        pr_sd_pd_vs_cr_vgpr_scr = [int(i) for i in ((dummies['PR'] == 1) | (dummies['SD'] == 1) | (dummies['PD'] == 1)) * 0 +
                                              ((dummies['CR'] == 1) | (dummies['VGPR'] == 1)) * 1 +
                                              ((dummies['SCR'] == 1) * 2)]

        result = pd.DataFrame({
            'G1': scr_vs_others,
            'G2': scr_cr_vs_others,
            'G3': scr_cr_vgpr_vs_others,
            'G4': scr_cr_vgpr_pr_vs_others,
            'G5': pr_sd_pd_vs_cr_vgpr_scr},
            index=dummies.index
        )

        return result.fillna(-1)
