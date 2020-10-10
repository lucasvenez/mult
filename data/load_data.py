# Copyright 2020 The MuLT Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

from constants import FISH_VARIABLE_NAMES

import pandas as pd
import os

CLINICAL_OUTCOME = 'response_best_response_first_line'


def load_data():

    # loading clinical data
    clinical = pd.read_csv(os.path.join(os.path.dirname(__file__), 'clinical.tsv'), sep='\t', index_col='ID')

    # split fish markers
    fish = clinical[FISH_VARIABLE_NAMES]
    
    fish = fish.replace({'Not Detected': 0, 'Detected': 1})

    # split clinical outcomes
    outcome = clinical.iloc[:, :4].copy()
    
    for column in outcome.columns:
        del clinical[column]
        
    for f in FISH_VARIABLE_NAMES:
        del clinical[f]
        
    del clinical['therapy_first_line_class']
    del clinical['family_cancer']

    # formatting treatments
    treatments = pd.get_dummies(clinical[['therapy_first_line']].fillna('Non-therapy'))
    # treatments = clinical[['therapy_first_line']].fillna('Non-therapy')
    # treatments['therapy_first_line'] = treatments['therapy_first_line'].astype('category').cat.codes

    # clinical = clinical[~(clinical.isnull().sum(axis=1) > 8)]

    del clinical['therapy_first_line']
    
    # treatments.columns = [c.replace('therapy_first_line_', '') for c in treatments.columns]
    
    # loading and pre-processing gene expression markers
    gene_expressions = pd.read_csv(os.path.join(os.path.dirname(__file__), 'gene_count.tsv'), sep='\t', index_col='ID')
    
    # selecting patients with both clinical and gene expression markers
    selected_index = clinical.join(gene_expressions, how='inner').index
    
    clinical = clinical.loc[selected_index]
    fish = fish.loc[selected_index]
    gene_expressions = gene_expressions.loc[selected_index]
    treatments = treatments.loc[selected_index]
    outcome = outcome[[CLINICAL_OUTCOME]].loc[selected_index]
    
    clinical = clinical.loc[~outcome.iloc[:, 0].isnull(), :]
    fish = fish.loc[~outcome.iloc[:, 0].isnull(), :]
    gene_expressions = gene_expressions.loc[~outcome.iloc[:, 0].isnull(), :]
    treatments = treatments.loc[~outcome.iloc[:, 0].isnull(), :]
    outcome = outcome.loc[~outcome.iloc[:, 0].isnull(), :]
    
    # removing invalid gene expressions
    for g in gene_expressions.loc[:, gene_expressions.sum() == 0].columns:
        del gene_expressions[g]

    gene_expressions = gene_expressions.dropna(axis=1, how='any')
    
    # removing treatment with less than 10 samples
    for treatment in treatments.loc[:, treatments.sum() < 10].columns:
        del treatments[treatment]

    def ecog_ps_to_int(ecog_ps):
        try:
            for i in range(1, 5):
                if str(i) in ecog_ps:
                    return i
        except:
            pass
        return None
        
    clinical['ecog_ps'] = clinical['ecog_ps'].apply(ecog_ps_to_int)
    
    clinical['first_line_transplant'] = clinical['first_line_transplant'].replace('Yes', 1).replace('No', 0)
    
    clinical['gender'] = clinical['gender'].replace('Male', 0).replace('Female', 1)

    # representing categorical features as one hot encoding (dummy)
    for c in clinical.columns:
        if str(clinical[c].dtype) == 'object':
            dummy = pd.get_dummies(clinical[c])
            dummy.columns = [c.lower() + '_' + d.lower().replace('/', '_').replace(' ', '_') for d in dummy.columns]
            clinical = pd.concat([clinical, dummy], axis=1)
            del clinical[c]

    # returning cleaned data set
    return clinical, fish, gene_expressions, treatments, outcome
