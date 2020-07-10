from constants import FISH_VARIABLE_NAMES

import pandas as pd
import os

CLINICAL_OUTCOME = 'response_best_response_first_line'


def load_data():

    # loading clinical data
    clinical = pd.read_csv(os.path.join(os.path.dirname(__file__), 'clinical.tsv'), sep='\t', index_col='ID')

    # split fish markers
    fish = clinical[FISH_VARIABLE_NAMES].copy()

    # split clinical outcomes
    outcome = clinical.iloc[:, :4].copy()
    
    for column in outcome.columns:
        del clinical[column]
        
    for f in FISH_VARIABLE_NAMES:
        del clinical[f]
        
    del clinical['therapy_first_line_class']
    
    treatments = pd.get_dummies(clinical[['therapy_first_line']])
    
    treatments.columns = [c.replace('therapy_first_line_', '') for c in treatments.columns]
    
    # loading and pre-processing gene expression markers
    gene_expressions = pd.read_csv(os.path.join(os.path.dirname(__file__), 'gene_count.tsv'), sep='\t', index_col='ID')
    
    # removing invalid gene expressions
    for g in gene_expressions.loc[:, gene_expressions.sum() == 0].columns:
        del gene_expressions[g]

    gene_expressions = gene_expressions.dropna(axis=1, how='any')
    
    # selecting patients with both clinical and gene expression markers
    selected_index = clinical.join(gene_expressions, how='inner').index

    # removing treatment with less than 10 samples
    for treatment in treatments.loc[:, treatments.sum() < 10].columns:
        del treatments

    return clinical.loc[selected_index], fish.loc[selected_index], \
           gene_expressions.loc[selected_index], treatment.loc[selected_index], \
           outcome[[CLINICAL_OUTCOME]].loc[selected_index]
