from data import load_data_gse

import pandas as pd

def processing_gse94873(clinical):
    clinical = clinical.replace({'NA':None})
    
    clinical = clinical.rename(columns={'immunotherapy_response_(responder=1_nonresponder=0)':'immunotherapy_responder',
                            'survival_status_12_months_(1=alive_0=dead)':'survival_status'})

    clinical['gender'] = clinical['gender'].replace({'FEMALE':'female', 'MALE':'male'})
    clinical['cancer_stage'] = clinical['cancer_stage'].replace({'IIIC': 1, 'IV M1A': 2, 'IV M1B': 3, 'IV M1C': 4})

    clinical = pd.concat([clinical.drop('gender', 1),  pd.get_dummies(clinical['gender'])], axis=1)

    for col in ['tissue', 'male']: 
        del clinical[col]
    
    clinical = clinical.astype(float)
    
    outcome = clinical['survival_status']
    
    return clinical, outcome

def load_data_gse94873(verbose=-1, read_as_ndarray=False):
    """
    This method loads the data set of the project GSE94873 available at
    https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE94873. This project
    reports the study of models to predict clinical response of patients with
    advanced melanoma.

    :param verbose: (int) print logging messages if greater than 0 (default: -1)
    :param read_as_ndarray: (bool) reads data as pandas data frame if false and
    as numpy ndarray if True (default: False)
    :return:
        - clinical (pd.DataFrame): contains a set of clinical markers associated to Melanoma patients,
        - genes (pd.DataFrame): contains gene expression levels associated to Melanoma patients,
        - outcome (pd.DataFrame): contains one variable grouping patients in dead (0) and alive (1)
    """
    clinical, genes, outcome = load_data_gse('GSE94873', processing_gse94873, verbose, read_as_ndarray)

    return clinical, genes, outcome