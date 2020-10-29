import pandas as pd
import numpy as np

from data import load_data_gse


def processing_gse136400(clinical):

    df = clinical.copy()

    delete = ['eventid', 'imidgiven', 'time_btwn', 'sampleprotocol', 'gepprocesslocation',
              'ftprotocol', 'ftprotocolbase', 'datepulled', 'datelastcontact', 'sampledatetime', 'sampledatetime__1',
              'iss1', 'iss2', 'iss3', 'protocolsample', 'celfilernbx', 'datapulled', 'sampleid', 'chipdate',
              'sampleprotocol.1', 'datebasesample', 'dateenrolled', 'datesample', 'cell_type', 'protocol_fill',
              'inregimenstep', 'datechip', 'location.signal', 'datesamplernbx', 'descname', 'celfilernas', 'agelte65',
              'refchip', 'celname', 'wide_pointgoup', 'wide_pointgroup', 'point', 'pc_bmbxtxt', 'issr', 'sorting',
              'wide_include', 'tissue', 'censorforosdate', 'censorforpfsdate', 'sampletype', 'processed_site',
              'chiptype', 'tc', 'fnpatid', 'transplant', 'isexpress', 'censos.ar', 'censpfs.ar', 'agelte75',
              'hyperdiploidclinicalfish', 'amp1qclinicalfish', 'del13qclinicalfish', 'del16qclinicalfish',
              'del17pclinicalfish', 't_11_14clinicalfish', 't_4_14clinicalfish', 'monthspfs.ar', 'uams.cd1',
              'uams.cd2', 'uams.hy', 'uams.lb', 'uams.mf', 'uams.ms', 'uams.pr', 'monthsos.ar', 'monthspfs.ar2',
              'datelastresponse', 'maxpfs', 'monthsobs', 'monthsos', 'responselast', 'imidpostresponse', 'imid_date',
              'imidimproved', 'pfsgt18', 'maxpfsgt18mon', 'maxpfsgt24mon', 'maxpfsgt36mon'
    ]

    for c in delete:
        del df[c]

    df = df.replace({'FALSE': 0, 'TRUE': 1, 'NA': None,
                     '.': None, np.nan: None, 'Not Evaluable': None,
                     'PR_Unconfirmed': 'PR', 'NULL': None, 'Refused': None})

    def from_to_imids(x):
        try:
            return 'revlimid' if x.lower().startswith('revlimid') else 'thalidomide'
        except:
            return None

    df['imids'] = df['imids'].apply(from_to_imids)

    df['iss'] = df['iss'].apply(lambda x: {'I': 1, 'II': 2, 'III': 3}.get(x, None))

    df['gender'] = df['gender'].apply(lambda x: {'Female': 1, 'Male': 0}.get(x))

    # df['vital_status'] = df['datedeath'].isnull()
    del df['datedeath']

    df = df[df['race'].apply(lambda x: 'native' not in str(x).lower())]
    race_dummies = pd.get_dummies(df['race'].str.lower().str.replace(' / ', '_').str.replace(' ', '_'), 'race')
    del df['race']
    df = pd.concat([df, race_dummies], axis=1)

    imids = df['imids']
    pointgroup = df['pointgroup']
    descprotocol = df['descprotocol']
    gender = df['gender']
    cluster = df['cluster']

    del df['imids']
    del df['pointgroup']
    del df['descprotocol']
    del df['gender']
    del df['cluster']

    df = df.astype(float)

    df['imids'] = imids
    df['pointgroup'] = pointgroup
    df['descprotocol'] = descprotocol
    df['gender'] = gender
    df['cluster'] = cluster

    for c in ['gender', 'ageatsampledate', 'imids', 'iss', 'ldh', 'b2m']:
        df = df[~df[c].isnull()]

    for c in df.columns:
        if df[c].isnull().sum() / df.shape[0] > .05:
            del df[c]

    outcome = pd.DataFrame({'risk_group': (df['monthspfs'] > df['monthspfs'].mean())}).astype(float)
    del df['monthspfs']

    df['transplant'] = (df['numberoftransplants'] > 0).astype(int)
    del df['numberoftransplants']

    df = df.rename(columns={'ageatsampledate': 'age_at_diagnosis', 'imids': 'treatment'})

    return df, outcome


def load_data_gse136400(verbose=-1, read_as_ndarray=False):
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
    clinical, genes, outcome = load_data_gse('GSE136400', processing_gse136400, verbose, read_as_ndarray)

    clinical, genes = clinical.fillna(0), genes.fillna(0).apply(lambda x: np.exp(x))

    index = clinical.join(genes, how='inner').join(outcome, how='inner').index

    return clinical.loc[index, :], genes.loc[index, :], outcome.loc[index, :]
