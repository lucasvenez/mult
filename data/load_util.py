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

    # loading keys
    keys, metadata_keys = set(), set()
    for gsm_name, gsm in dataset.gsms.items():

        # gettin all characteristics key
        for item in gsm.metadata['characteristics_ch1']:
            k, _ = item.strip().split(': ', 1)
            k = k.lower().strip().replace(' ', '_').replace('/', '_')
            keys.add(k)

        # getting all metadata keys
        for key, value in gsm.metadata.items():
            k = key.lower().strip().replace(' ', '_').replace('/', '_')
            if key != 'characteristics_ch1':
                metadata_keys.add(k)

    # parsing metadata, characteristics, and genes
    metadata, genes, characteristics = {'ID': [], **{c: [] for c in metadata_keys}}, None, {'ID': [],
                                                                                            **{c: [] for c in keys}}

    for gsm_name, gsm in dataset.gsms.items():

        # metadata
        metadata_local_keys = set()
        metadata['ID'].append(gsm_name)

        for key, value in gsm.metadata.items():

            if key != 'characteristics_ch1':

                key = key.lower().strip().replace(' ', '_').replace('/', '_')

                if key not in metadata_local_keys:
                    metadata[key].append(', '.join(value))
                    metadata_local_keys.add(key)

        for k in metadata_keys.difference(metadata_local_keys):
            metadata[k].append(None)

        if gsm.table.shape[0] > 0:

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

            k, v = item.split(': ', 1)

            k = k.lower().strip().replace(' ', '_').replace('/', '_')

            if k not in local_keys:
                local_keys.add(k)
                characteristics[k].append(v if v != '--' else None)

        for k in keys.difference(local_keys):
            characteristics[k].append(None)

    metadata = pd.DataFrame(metadata).set_index('ID')
    characteristics = pd.DataFrame(characteristics).set_index('ID')

    if genes is not None:
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
    current_path = os.path.join(os.path.dirname(__file__))
    dataset_path = os.path.join(current_path, geo_id)

    if not os.path.exists(dataset_path):
        os.makedirs(dataset_path)

    clinical_path = os.path.join(dataset_path, 'clinical.csv')
    gene_path = os.path.join(dataset_path, 'gene.csv')
    outcome_path = os.path.join(dataset_path, 'outcome.csv')

    if os.path.isfile(clinical_path) and os.path.isfile(gene_path) and os.path.isfile(outcome_path):
        clinical = pd.read_csv(clinical_path, index_col='ID')
        genes = pd.read_csv(gene_path, index_col='ID').T
        outcome = pd.read_csv(outcome_path, sep=',', index_col='ID')

    else:

        characteristics, genes, metadata, platforms = load_data_from_geo(
            geo_id=geo_id, dst_path=os.path.join(current_path, geo_id), silent=verbose <= 0)

        clinical, outcome = processing_function(characteristics)

        # exporting databases
        if isinstance(clinical, pd.DataFrame):
            clinical.to_csv(clinical_path, sep=',', index=True)

        if isinstance(genes, pd.DataFrame):
            genes.T.to_csv(gene_path, sep=',', index=True)

        if isinstance(outcome, pd.DataFrame):
            outcome.to_csv(outcome_path, sep=',', index=True)

    if read_as_ndarray:

        if isinstance(clinical, pd.DataFrame):
            clinical = clinical.values

        if isinstance(genes, pd.DataFrame):
            genes = genes.values

        if isinstance(outcome, pd.DataFrame):
            outcome = outcome.values

    outcome = outcome[outcome.notnull()]

    if genes is not None and outcome is not None and clinical is not None:
        index = clinical.join(genes, how='inner').join(outcome, how='inner').index

    elif outcome is not None and clinical is not None:
        index = clinical.join(outcome, how='inner').index

    else:
        index = genes.join(outcome, how='inner').index

    if genes is not None:
        genes = genes.dropna(axis=1, how='any')
        genes = genes.loc[index, :]

    if clinical is not None:
        clinical = clinical.loc[index, :].fillna(0)

    if outcome is not None:
        outcome = outcome.loc[index, :]

    return clinical, genes, outcome
