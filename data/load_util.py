import GEOparse
import pandas as pd


def load_data_from_geo(geo_id, dst_path='.'):

    dataset = GEOparse.get_GEO(geo_id, destdir=dst_path, silent=True)

    # Parsing platforms
    platforms = {}
    for gpl_name, gpl in dataset.gpls.items():
        platforms[gpl_name] = gpl.table
    # loading characteristic keys

    keys = set()
    for gsm_name, gsm in dataset.gsms.items():
        for item in gsm.metadata['characteristics_ch1']:
            k, _ = item.split(': ')
            keys.add(k)

    # parsing metadata, characteristics, and genes
    metadata, genes, characteristics = {'ID': []}, None, {'ID': []}
    for gsm_name, gsm in dataset.gsms.items():
        # metadata

        metadata['ID'].append(gsm_name)
        for key, value in gsm.metadata.items():
            if key != 'characteristics_ch1':
                if key not in metadata:
                    metadata[key] = []
                metadata[key].append(', '.join(value))

                # genes

        tmp = gsm.table[['ID_REF', 'VALUE']].set_index('ID_REF')

        tmp.columns = [gsm_name]
        if genes is None:
            genes = tmp
        else:
            genes = genes.join(tmp, how='left')

        # characteristics

        local_keys = set()

        characteristics['ID'].append(gsm_name)

        for item in gsm.metadata['characteristics_ch1']:

            k, v = item.split(': ')
            local_keys.add(k)
            if k not in characteristics:  # if k not in result:
                characteristics[k] = []

            characteristics[k].append(v if v != '--' else None)
        for k in keys.difference(local_keys):
            characteristics[k].append(None)
    metadata = pd.DataFrame(metadata).set_index('ID')
    characteristics = pd.DataFrame(characteristics).set_index('ID')

    genes = genes.T

    genes.index.name = 'ID'

    return characteristics, genes, metadata, platforms
