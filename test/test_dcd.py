from sklearn.model_selection import KFold
from sklearn.metrics import roc_auc_score, accuracy_score, confusion_matrix
from sklearn.preprocessing import *

from mlworkflow import dcd2

import unittest
import os

import pandas as pd


class ModelTest(unittest.TestCase):

    def test_experiment_01(self):

        print('Loading and processing data')

        df = pd.read_csv('../output/cli_data_with_groups.tsv', sep='\t').set_index('ID')

        therapies = ['Bor', 'Bor-Cyc-Dex+Bor-Dex', 'Bor-Cyc-Dex', 'Bor-Dex+Bor', 'Bor-Dex+Bor-Cyc-Dex',
                     'Bor-Dex+Bor-Len-Dex+Len', 'Bor-Dex+Bor-Len-Dex', 'Bor-Dex', 'Bor-Len-Dex+Bor-Dex',
                     'Bor-Len-Dex+Len', 'Bor-Len-Dex', 'Len', 'Len-Dex+Bor-Len-Dex', 'Len-Dex']

        drugs = set()

        for therapy in therapies:
            for part in therapy.split('+'):
                for drug in part.split('-'):
                    drug = drug.lower()
                    df[drug] = 0
                    drugs.add(drug)


        for index, row in df.iterrows():
            for therapy in therapies:
                if row[therapy] > 0:
                    for part in therapy.split('+'):
                        for drug in part.split('-'):

                            drug = drug.lower()

                            df.loc[index, drug] += 1

        df[list(drugs)] = df[list(drugs)].fillna(-1)

        df_therapies = df[therapies]

        df_genetic = pd.read_csv('../input/iss_status_without_log_valid.tsv', '\t').rename(columns={'PUBLIC_ID': 'ID'})
        df_genetic['ID'] = df_genetic['ID'].str.replace('_', '')
        df_genetic.set_index('ID', inplace=True)

        df_genetic = df_genetic[[col for col in df_genetic.columns if 'FPKM' in str(col)]]
        df_genetic.columns = [col.replace('.', '').replace('-', '') for col in df_genetic.columns]

        rootdir = '../output/group/'

        selected_vars = {'all': set()}

        n_vars = 5

        for _, dirs, _ in os.walk(rootdir):

            for dir in dirs:

                selected_vars[dir] = set()

                for _, _, files in os.walk(os.path.join(rootdir, dir)):

                    for file in files:

                        file = os.path.join(rootdir, dir, file)

                        if 'G2' in pd.read_csv(file, sep='\t', nrows=2).columns:

                            ascending = True if dir in {'wilcox', 'kruskal'} else False

                            df_correlation = pd.read_csv(file, sep='\t')

                            df_correlation = df_correlation.loc[['FPKM' in val for val in df_correlation['IND_VAR']],:]

                            if dir == 'mine':
                                df_correlation['G2'] = df_correlation['G2'].apply(lambda x: float(eval(x)[0]))

                            elif dir == 'pearson':
                                df_correlation['G2'] = df_correlation['G2'].apply(lambda x: float(abs(x)))

                            else:
                                df_correlation['G2'] = df_correlation['G2'].astype(float)

                            vars = df_correlation.sort_values(by='G2', ascending=ascending)['IND_VAR']

                            vars = [var.replace('.', '').replace('-', '') for var in (vars.as_matrix().tolist())]

                            selected_vars['all'] = selected_vars['all'].union(set(vars[:n_vars]))

                            selected_vars[dir] = selected_vars[dir].union(set(vars[:n_vars]))

        print('Start training')

        #
        # Executing cross-validation
        #
        kfold = KFold(n_splits=10, random_state=1)

        result = pd.DataFrame({'model': [], 'selector': [], 'fold': [], 'auc': [], 'acc': [],
                               'tp': [], 'tn': [], 'fp': [], 'fn': [], 'n_vars': n_vars, 'vars': []})

        for selector in selected_vars:

            if selector != 'all':
                continue

            print('Selector: {}'.format(selector))
            #
            # Building datasets
            #
            x = df_genetic[list(selected_vars[selector])].join(df[list(drugs)])

            y = x.join(df['G2'])['G2'].astype(int).as_matrix().reshape([-1, 1])

            x = x.as_matrix()

            for fold, (train_index, test_index) in enumerate(kfold.split(x, y)):

                predictions = None

                print('> Fold {}'.format(fold + 1))

                x_train, y_train = x[train_index, :], y[train_index, :]

                x_test, y_test = x[test_index, :], y[test_index, :]

                scaler = StandardScaler()

                x_train = scaler.fit_transform(x_train)

                x_test = scaler.transform(x_test)

                y_hat = dcd2(x_train, y_train, x_test, y_test, selector, fold + 1)

                #
                # Exporting predictions
                #

                for a, b in zip(y_test, y_hat):

                    row = pd.DataFrame({'y_true': a, 'y_hat': b})

                    if predictions is None:
                        predictions = row
                    else:
                        predictions = predictions.append(row)

                predictions.to_csv('../output/results/MMDCD_ALL_128_{}.csv'.format(fold + 1 if fold > 8 else '0{}'.format(fold + 1)), sep=',', index=False)

                auc = roc_auc_score(y_test, y_hat)

                y_hat[y_hat >= .5] = 1

                y_hat[y_hat < .5] = 0

                acc = accuracy_score(y_test, y_hat)

                tn, fp, fn, tp = confusion_matrix(y_test, y_hat).ravel()

                result = result.append(pd.DataFrame({
                    'model': ['dcd'], 'selector': [selector],
                    'fold': [fold + 1], 'auc': [auc], 'acc': [acc],
                    'tp': [tp], 'tn': [tn], 'fp': [fp], 'fn': [fn],
                    'n_vars': [n_vars], 'vars': [','.join(list(selected_vars[selector]))]}))

                result.to_csv('../output/experiment_one_dcd_model.tsv', sep='\t', index=False)