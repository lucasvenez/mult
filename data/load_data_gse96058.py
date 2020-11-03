from data import load_data_gse

from contextlib import closing

import os
import shutil
import numpy as np
import pandas as pd
import urllib.request as request


def processing_gse96058(clinical):

    assert isinstance(clinical, pd.DataFrame), 'Invalid clinical type. It should be a pandas data frame.'

    # cleaning clinical markers
    clinical = clinical.replace({'NA': None, 'na': None})

    del clinical['scan-b_external_id']

    clinical['instrument_model'] = clinical['instrument_model'].replace({
        'HiSeq 2000': 0, 'NextSeq 500': 1})

    lymph_dummies = pd.get_dummies(clinical['lymph_node_group'])
    lymph_dummies.columns = ['lymph_node_group_' + c for c in lymph_dummies.columns]
    clinical = pd.concat([clinical, lymph_dummies], axis=1)
    del clinical['lymph_node_group']

    clinical['lymph_node_status'] = clinical['lymph_node_status'].replace({
        'NodeNegative': 0, 'NodePositive': 1})
    clinical['nhg'] = clinical['nhg'].replace({'G1': 1, 'G2': 2, 'G3': 3})
    clinical['nhg_prediction_mgc'] = clinical['nhg_prediction_mgc'].replace({'G2': 2, 'G3': 3})

    pam50_dummies = pd.get_dummies(clinical['pam50_subtype'])
    pam50_dummies.columns = ['pam50_subtype_' + c for c in pam50_dummies.columns]
    clinical = pd.concat([clinical, pam50_dummies], axis=1)
    del clinical['pam50_subtype']

    for c in clinical.columns:
        clinical[c] = clinical[c].astype(float)

    #
    outcome = pd.DataFrame((clinical['overall_survival_days'] >=
                            clinical['overall_survival_days'].mean()).astype(float))

    outcome.columns = ['risk_group']

    # removing clinical markers invalid for
    # building high and low risk predictors
    for column in ['overall_survival_days', 'overall_survival_event']:
        del clinical[column]

    # removing estimated values
    for c in clinical.columns:
        if 'prediction' in c or 'model' in c:
            del clinical[c]

    return clinical, outcome


def load_data_gse96058(verbose=-1, read_as_ndarray=False):
    """
    This method loads the data set of the project GSE96058 available at
    https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE96058.

    :param verbose: (int) print logging messages if greater than 0 (default: -1)
    :param read_as_ndarray: (bool) reads data as pandas data frame if false and
    as numpy ndarray if True (default: False)

    :return:
        - clinical (pd.DataFrame): contains a set of clinical markers associated to lung patients,
        - genes (pd.DataFrame): contains gene expression levels associated to lung patients,
        - outcome (pd.DataFrame): contains one variable grouping patients in high (0) and low (1) risk
    """
    clinical, _, outcome = load_data_gse('GSE96058', processing_gse96058, verbose, read_as_ndarray)

    genes = get_gene_expressions(list(clinical.index))

    return clinical, genes, outcome


def get_gene_expressions(columns):

    base_path = os.path.join(os.path.dirname(__file__),  'GSE96058')

    final_genes_filename = os.path.join(base_path, 'genes.csv')

    if not os.path.exists(final_genes_filename):

        if not os.path.exists(base_path):
            os.makedirs(base_path)

        filename = 'GSE96058_gene_expression_3273_samples_and_136_replicates_transformed.csv.gz'

        ftp_url = 'ftp://ftp.ncbi.nlm.nih.gov/geo/series/GSE96nnn/GSE96058/suppl/' \
                  'GSE96058_gene_expression_3273_samples_and_136_replicates_transformed.csv.gz'

        if not os.path.exists(os.path.join(base_path, filename)):
            with closing(request.urlopen(ftp_url)) as r:
                with open(os.path.join(base_path, filename), 'wb') as f:
                    shutil.copyfileobj(r, f)

        genes = pd.read_csv(os.path.join(base_path, filename), sep=',')

        genes = genes.rename(columns={'Unnamed: 0': 'ID'}).set_index('ID')

        genes.columns = columns

        if not os.path.isfile(final_genes_filename):
            genes.to_csv(os.path.join(final_genes_filename))

    else:

        genes = pd.read_csv(final_genes_filename, sep=',', index_col='ID')

    genes = genes.T

    return genes.apply(lambda x: np.exp(x))
