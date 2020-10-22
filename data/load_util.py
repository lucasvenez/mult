import os
import GEOparse
import pandas as pd


def load_data_from_geo(geo_id, dst_path='.', silent=True):
    """

    :param geo_id:
    :param dst_path:
    :param silent:
    :return:
    """

    dataset = GEOparse.get_GEO(geo_id, destdir=dst_path, silent=silent)

    # Parsing platforms
    platforms = {}
    for gpl_name, gpl in dataset.gpls.items():
        platforms[str(gpl_name).upper()] = gpl.table
    # loading characteristic keys

    keys = set()
    for gsm_name, gsm in dataset.gsms.items():
        for item in gsm.metadata['characteristics_ch1']:
            k, _ = item.split(': ')
            k = k.lower().strip().replace(' ', '_').replace('/', '_')
            keys.add(k)

    # parsing metadata, characteristics, and genes
    metadata, genes, characteristics = {'ID': []}, None, {'ID': []}
    for gsm_name, gsm in dataset.gsms.items():
        # metadata

        metadata['ID'].append(gsm_name)
        for key, value in gsm.metadata.items():
            if key != 'characteristics_ch1':
                key = key.lower().strip()
                if key not in metadata:
                    metadata[key] = []
                metadata[key].append(', '.join(value))

                # genes

        tmp = gsm.table[['ID_REF', 'VALUE']].set_index('ID_REF')

        tmp.columns = [str(gsm_name).upper()]
        if genes is None:
            genes = tmp
        else:
            genes = genes.join(tmp, how='left')

        # characteristics

        local_keys = set()

        characteristics['ID'].append(gsm_name)

        for item in gsm.metadata['characteristics_ch1']:

            k, v = item.split(': ')
            k = k.lower().strip().replace(' ', '_').replace('/', '_')
            local_keys.add(k)
            if k not in characteristics:
                characteristics[k] = []
            characteristics[k].append(v if v != '--' else None)

        for k in keys.difference(local_keys):
            characteristics[k].append(None)
    metadata = pd.DataFrame(metadata).set_index('ID')
    characteristics = pd.DataFrame(characteristics).set_index('ID')

    genes = genes.reset_index().rename(columns={'ID_REF': 'ID'}).set_index('ID')

    genes = genes.T

    return characteristics, genes, metadata, platforms


def load_data_gse(geo_id, processing_function, verbose=-1, read_as_ndarray=False):
    """


    :param geo_id: (str) Gene expression Omnimbus (GEO) ID respecting the pattern GSE[0-9]+
    :param processing_function: (callable) function processing clinical data and returning one data frame with
    cleaned clinical data and clinical outcome (e.g., high and low risk classes)
    :param verbose: (int) print logging messages if greater than 0 (default: -1)
    :param read_as_ndarray: (bool) reads data as pandas data frame if false and
    as numpy ndarray if True (default: False)

    :return:
        - clinical (pd.DataFrame): contains a set of clinical markers associated to lung patients,
        - genes (pd.DataFrame): contains gene expression levels associated to lung patients,
        - outcome (pd.DataFrame): contains one variable grouping patients in high (0) and low (1) risk
    """

    # validating geo_id value
    assert isinstance(geo_id, str) and geo_id.startswith('GSE') and geo_id.replace('GSE', '').isnumeric(), \
        'invalid GEO ID number. It should matches with GSE[0-9]+ regular expression.'

    # validating processing_function value
    assert callable(processing_function), 'processing_function should be a callable object'

    # defining function constants
    GEO_ID, CURRENT_PATH = geo_id, os.path.join(os.path.dirname(__file__))
    DATASET_PATH = os.path.join(CURRENT_PATH, GEO_ID)

    if not os.path.exists(DATASET_PATH):
        os.makedirs(DATASET_PATH)

    clinical_path = os.path.join(DATASET_PATH, 'clinical.csv')
    gene_path = os.path.join(DATASET_PATH, 'gene.csv')
    outcome_path = os.path.join(DATASET_PATH, 'outcome.csv')

    if os.path.isfile(clinical_path) and os.path.isfile(gene_path) and os.path.isfile(outcome_path):
        clinical = pd.read_csv(clinical_path, index_col='ID')
        genes = pd.read_csv(gene_path, index_col='ID').T
        outcome = pd.read_csv(outcome_path, sep=',', index_col='ID')

    else:

        characteristics, genes, metadata, platforms = load_data_from_geo(
            geo_id=GEO_ID,
            dst_path=CURRENT_PATH,
            silent=verbose <= 0)

        clinical, outcome = processing_function(characteristics)

        # exporting databases
        clinical.to_csv(clinical_path, sep=',', index=True)
        genes.T.to_csv(gene_path, sep=',', index=True)
        outcome.to_csv(outcome_path, sep=',', index=True)

    if read_as_ndarray:
        clinical = clinical.values
        genes = genes.values
        outcome = outcome.values

    return clinical, genes, outcome
