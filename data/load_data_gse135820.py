from data import load_data_gse

import numpy as np
import pandas as pd


def processing_gse135820(clinical):

    assert isinstance(clinical, pd.DataFrame), 'Invalid clinical type. It should be a pandas data frame.'

    # removing markers
    for column in ['sample_region_cellularity', 'sample_region_necrosis', 'year_of_diagnosis',
                   'study_entry_delay', 'experiment_site', 'site_id', 'progression-free_survival_time',
                   'residual_disease_status']:
        del clinical[column]

    clinical = clinical.replace({'NA': None, 'unknown': None, 'Unknown': None})

    # cleaning clinical markers
    clinical['stage'] = clinical['stage'].replace({'unknown': None, 'low': 1, 'high': 2})
    clinical['vital_status'] = clinical['vital_status'].replace({'dead': 0, 'alive': 1})

    diagnosis_dummies = pd.get_dummies(clinical['diagnosis'], 'diagnosis')
    diagnosis_dummies.columns = [c.replace(' ', '_').replace('-', '_').lower() for c in diagnosis_dummies.columns]
    clinical = pd.concat([clinical.drop('diagnosis', 1),  diagnosis_dummies], axis=1)

    race_dummies = pd.get_dummies(clinical['race_ethnicity'], 'race_ethnicity')
    race_dummies.columns = [c.replace(' ', '_').replace('-', '_').lower() for c in race_dummies.columns]
    clinical = pd.concat([clinical.drop('race_ethnicity', 1),  race_dummies], axis=1)

    mutation_dummies = pd.get_dummies(clinical['brca1_and_brca2_germline_mutation_status'], 'mutation')
    mutation_dummies.columns = [c.replace(' ', '_').replace('-', '_').lower() for c in mutation_dummies.columns]
    clinical = pd.concat([clinical.drop('brca1_and_brca2_germline_mutation_status', 1),  mutation_dummies], axis=1)

    anatomical_dummies = pd.get_dummies(clinical['anatomical_site'], 'anatomical_site')
    anatomical_dummies.columns = [c.replace(' ', '_').replace('-', '_').lower() for c in anatomical_dummies.columns]
    clinical = pd.concat([clinical.drop('anatomical_site', 1), anatomical_dummies], axis=1)

    # removing samples without required features
    for col in ['age_at_diagnosis', 'vital_status', 'stage']:
        clinical = clinical[clinical[col].notnull()]

    clinical = clinical.astype(float)

    # remove redundant columns
    clinical = clinical.drop(columns=['diagnosis_non_hgsoc'])

    # generating the outcome
    outcome = pd.DataFrame((
        (clinical['overall_survival_time'] >= clinical['overall_survival_time'].mean()) &
        (clinical['vital_status'] == 1)).astype(float))
    outcome.columns = ['risk_group']

    # removing features that are a direct mapping to outcome (avoiding data leak)
    clinical = clinical.drop(columns=['vital_status', 'overall_survival_time'])

    return clinical, outcome


def load_data_gse135820(verbose=-1, read_as_ndarray=False):
    """
    This method loads the data set of the project GSE135820 available at
    https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE135820. This projects
    reports the development and validation of a predictor of high-grade-serousOvarian carcinoma,
    based on 4077 samples with clinical and gene expression features.

    :param verbose: (int) print logging messages if greater than 0 (default: -1)
    :param read_as_ndarray: (bool) reads data as pandas data frame if false and
    as numpy ndarray if True (default: False)
    :return:
        - clinical (pd.DataFrame): contains a set of clinical markers associated to lung patients,
        - genes (pd.DataFrame): contains gene expression levels associated to lung patients,
        - outcome (pd.DataFrame): contains one variable grouping patients in high (0) and low (1) risk
    """
    clinical, genes, outcome = load_data_gse('GSE135820', processing_gse135820, verbose, read_as_ndarray)

    return clinical, genes.apply(lambda x: np.exp(x)), outcome


if __name__ == '__main__':
    load_data_gse135820()
