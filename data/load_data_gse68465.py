from data import load_data_gse

import pandas as pd


def processing_gse68465(clinical):

    assert isinstance(clinical, pd.DataFrame), 'Invalid clinical type. It should be a pandas data frame.'

    # cleaning clinical markers
    clinical = clinical.replace({'na': None, 'NA': None, 'Unknown': None})

    clinical['disease_state'] = clinical['disease_state'].replace({
        'Normal': 0, 'Lung Adenocarcinoma': 1})

    clinical['gender'] = clinical['sex'].replace({'Male': 0, 'Female': 1})
    del clinical['sex']

    clinical['age_at_diagnosis'] = clinical['age'].astype(float)
    del clinical['age']

    clinical['race'] = pd.get_dummies(clinical['race'].replace({'Unknown': None, 'Not Reported': None}))

    clinical['vital_status'] = clinical['vital_status'].replace({'Dead': 0, 'Alive': 1})

    clinical['clinical_treatment_adjuvant_chemo'] = clinical['clinical_treatment_adjuvant_chemo'].replace(
        {'Unknown': None, 'No': 0, 'Yes': 1})

    clinical['clinical_treatment_adjuvant_rt'] = clinical['clinical_treatment_adjuvant_rt'].replace({
        'Unknown': None, 'No': 0, 'Yes': 1})

    def to_integer(x):
        try:
            return int(''.join(_ for _ in x if _.isdigit()))
        except:
            return None

    clinical['disease_stage'] = clinical['disease_stage'].apply(to_integer)

    clinical['first_progression_or_relapse'] = clinical['first_progression_or_relapse'].replace(
        {'No': 0, 'Yes': 1, 'Unknown': None})

    clinical['smoking_history'] = clinical['smoking_history'].replace({
        'Never smoked': 0, 'Smoked in the past': 1, 'Currently smoking': 2, 'Unknown': None})

    clinical['surgical_margins'] = clinical['surgical_margins'].replace({
        'ALL MARGINS PATHOLOGICALLY NEGATIVE': 0,
        'MICROSCOPICALLY POSITIVE MARGINS OR MICROSCOPIC RESIDUAL DISEASE': 1})

    for column in ['organism_part', 'months_to_first_progression']:
        del clinical[column]

    clinical['histologic_grade'] = clinical['histologic_grade'].replace({
        'POORLY DIFFERENTIATED': 0, 'Moderate Differentiation': 1, 'WELL DIFFERENTIATED': 2})

    #
    for a in clinical.columns:
        clinical[a] = clinical[a].astype(float)

    #
    for column in ['vital_status', 'months_to_last_contact_or_death']:
        clinical = clinical[~clinical[column].isnull()]

    #
    outcome = pd.DataFrame((clinical['months_to_last_contact_or_death'] >=
                            clinical['months_to_last_contact_or_death'].mean()).astype(float))

    outcome.columns = ['risk_group']

    # removing clinical markers invalid for
    # building high and low risk predictors
    for column in ['first_progression_or_relapse', 'mths_to_last_clinical_assessment',
                   'vital_status', 'months_to_last_contact_or_death']:
        del clinical[column]

    return clinical, outcome


def load_data_gse68465(verbose=-1, read_as_ndarray=False):
    """
    This method loads the data set of the project GSE68465 available at
    https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE68465. This project
    reports a large, training/testing, multi-site, blinded validation study to
    characterize the performance of several prognostic models based on
    gene expression for 442 lung adenocarcinomas.

    :param verbose: (int) print logging messages if greater than 0 (default: -1)
    :param read_as_ndarray: (bool) reads data as pandas data frame if false and
    as numpy ndarray if True (default: False)

    :return:
        - clinical (pd.DataFrame): contains a set of clinical markers associated to lung patients,
        - genes (pd.DataFrame): contains gene expression levels associated to lung patients,
        - outcome (pd.DataFrame): contains one variable grouping patients in high (0) and low (1) risk
    """
    clinical, genes, outcome = load_data_gse('GSE68465', processing_gse68465, verbose, read_as_ndarray)

    return clinical, genes, outcome
