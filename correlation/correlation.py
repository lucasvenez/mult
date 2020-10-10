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

from multiprocessing import Pool
from scipy.stats import ks_2samp
from scipy.stats import kruskal
from scipy.stats import entropy
from minepy import MINE

import multiprocessing

import scipy.spatial.distance as distance
import scipy.stats as statistical

import pandas as pd
import numpy as np


def entropy_stats(values):

    if isinstance(values, list) or isinstance(values, np.ndarray):
        values = pd.Series(values)

    assert isinstance(values, pd.Series)

    return entropy(values.value_counts(), base=2)


def select_genes_mic(genes, response, threshold=0.05, verbose=0):
    #
    # Gene Selection
    #
    
    excluded_genes = []
    
    gene_table = {'variable': [], 'score': []}
    
    all_ = genes.copy()
    all_['__response__'] = response
    
    mic_result = compute_mic(all_, ['__response__'])
    
    for index, row in mic_result.iterrows():
        gene_table['variable'].append(row['IND_VAR'])
        gene_table['score'].append(row['mic'])

    gene_table = pd.DataFrame(gene_table).set_index('variable')
    
    threshold = np.quantile(gene_table['score'].values, 1 - threshold)
    
    selected_genes = list(gene_table[gene_table['score'] >= threshold].sort_values(by='score', ascending=False).index)
    
    if verbose > 0:
        print('select_genes_mic selected {} variables in for the correlation step'.format(len(selected_genes)))
    
    pairwise_pearson = genes[selected_genes].corr().abs()
    
    for gi, g in enumerate(selected_genes[:-1]):
        
        gene_pearson = pairwise_pearson.loc[[g], :].iloc[:, (gi+1):]

        excluded_genes += list(gene_pearson.loc[:, (gene_pearson.values > .75)[0]].columns)
    
    selected_genes = [s for s in selected_genes if s not in excluded_genes]
    
    mics = [gene_table.loc[s, 'score'] for s in selected_genes]
    
    if verbose > 0:
        print('select_genes_mic selected {} variables in for the pairwise step'.format(len(mics)))
    
    return selected_genes, mics


def kruskal_pvalue(i):

    var, negatives, positives = i

    try:
        p = kruskal(negatives, positives)[1]
    except ValueError:
        p = 1.0

    e = entropy_stats(list(negatives) + list(positives))

    return var, p, e


def ks2samp_pvalue(i):

    var, negatives, positives = i

    try:
        p = ks_2samp(negatives, positives)[1]
    except ValueError:
        p = 1.0

    e = entropy_stats(list(negatives) + list(positives))

    return var, p, e


def select_genes(genes, response, threshold=0.05):
    #
    # Gene Selection
    #
    
    excluded_genes = []
    
    gene_table = {'variable': [], 'score': [], 'entropy': []}
    
    if isinstance(response, pd.DataFrame):
        response = response.values

    inputs = []

    for c in genes:

        gene_values = genes[c].values
        negatives = gene_values[response == 0]
        positives = gene_values[response == 1]

        if len(negatives) > 0 and len(positives) > 0:
            inputs.append((c, negatives, positives))

    with Pool(10) as pool:
        results = pool.map(ks2samp_pvalue, inputs)

    for r in results:

        var, pvalue, e = r

        gene_table['variable'].append(var)

        gene_table['score'].append(pvalue)

        gene_table['entropy'].append(e)

    gene_table = pd.DataFrame(gene_table).set_index('variable')
    
    selected_genes = list(gene_table[gene_table['score'] < threshold].sort_values(by='score').index)

    pairwise_pearson = genes[selected_genes].corr().abs()
    
    for gi, g in enumerate(selected_genes[:-1]):
        
        gene_pearson = pairwise_pearson.loc[[g],:].iloc[:, (gi+1):]
        
        excluded_genes += list(gene_pearson.loc[:, (gene_pearson.values > .95)[0]].columns)
    
    result = [s for s in selected_genes if s not in excluded_genes]
    
    pvalues = [gene_table.loc[s, 'score'] for s in result]

    es = [gene_table.loc[s, 'entropy'] for s in result]
    
    return result, pvalues, es


def compute(dataset, dependent_variable_names, func):

    assert isinstance(dependent_variable_names, list)
    
    assert isinstance(dataset, pd.DataFrame) and dataset.shape[0] > 0
    
    assert callable(func)
    
    try:
    
        dependent_variables = dataset[dependent_variable_names]

        for name in dependent_variable_names:
            del dataset[name]

        num_cores = multiprocessing.cpu_count() * 3

        inputs = [(independent_var.values, dependent_var.values, independent_var.name, dependent_var.name)
                                                   for i, independent_var in dataset.iteritems()
                                                   for j, dependent_var in dependent_variables.iteritems()
                                                   if len(dependent_var.unique()) > 1]

        if len(inputs) > 0:

            with multiprocessing.Pool(num_cores) as p:
                results = p.map(func, inputs)

            results_dict = {'IND_VAR': [], 'DEP_VAR': [], func.__name__: []}

            for i in results:
                if i is not None:
                    key = next(iter(i.keys()))
                    results_dict['IND_VAR'] += [key[0]]
                    results_dict['DEP_VAR'] += [key[1]]
                    results_dict[func.__name__] += [i[key]]

            df = pd.DataFrame(results_dict)

            return df

        return None
    
    except Exception as e:
        print(e)
        return None


def compute_distcorr(dataset, dependent_variable_names):
    return compute(dataset, dependent_variable_names, distcorr)


def compute_mine(dataset, dependent_variable_names):
    return compute(dataset, dependent_variable_names, mine)


def compute_mic(dataset, dependent_variable_names):
    return compute(dataset, dependent_variable_names, mic)


def compute_wilcox(dataset, dependent_variable_names):
    return compute(dataset, dependent_variable_names, wilcox)


def compute_kruskal(dataset, dependent_variable_names):
    return compute(dataset, dependent_variable_names, kruskal)


def compute_pearson(dataset, dependent_variable_names):
    return compute(dataset, dependent_variable_names, pearson)


def compute_ks(dataset, dependent_variable_names):
    return compute(dataset, dependent_variable_names, ks)


def mic(pair):
        
    try:
        
        assert len(pair) == 4 and isinstance(pair, tuple)

        x, y, x_name, y_name = pair

        mine = MINE()

        mine.compute_score(x, y)

        return {(x_name, y_name): mine.mic()}
    
    except:
        
        return None


def mine(pair):

    assert len(pair) == 4 and isinstance(pair, tuple)

    try:
    
        x, y, x_name, y_name = pair

        mine_ = MINE()

        mine_.compute_score(x, y)

        result = {(x_name, y_name): (mine_.mic(), mine_.mas(), mine_.mcn(), mine_.mev(), mine_.tic())}

        return result
    
    except:
        
        return None


def wilcox(pair):

    assert len(pair) == 4 and isinstance(pair, tuple)

    try:
    
        result = None

        x, y, x_name, y_name = pair
      
        if len(x[y == 0]) > 0 and len(x[y == 1]) > 0:

            wilcox_result = statistical.wilcoxon(x[y == 0], x[y == 1], zero_method='wilcox')[1]

            result = {(x_name, y_name): wilcox_result}

        return result
    
    except Exception as e:
        
        print(e)
        
        return None


'''
def kruskal(pair):

    assert len(pair) == 4 and isinstance(pair, tuple)

    try:
    
        result = None

        x, y, var, group = pair

        var = var.replace('-', '.')
        
        group = group.replace('-', '.')

        df = pd.DataFrame({'var': x.astype(float), 'group': y.astype(float)})

        if df.shape[0] > 0 and df.shape[1] > 1:

            if len(df['group'].unique()) > 1:

                df = com.convert_to_r_dataframe(df=df, strings_as_factors=True)

                f = stats.formula('var ~ group')

                pvalue = stats.kruskal_test(f, data=df)[2][0]

                result = {(var, group): pvalue}

        return result
    
    except:
        
        return None
'''


def distcorr(pair):

    assert len(pair) == 4 and isinstance(pair, tuple)

    try:
    
        X, Y, x_name, y_name = pair

        result = None

        if len(pd.unique(Y)) > 1:

            dcor = distance.correlation(X, Y)

            result = {(x_name, y_name): dcor}

        return result
    
    except:
        
        return None


def pearson(pair):

    assert len(pair) == 4 and isinstance(pair, tuple)

    try:
    
        X, Y, x_name, y_name = pair

        result = None

        if len(pd.unique(Y)) > 1:
            try:
                result = {(x_name, y_name): statistical.pearsonr(X, Y)[0]}
            except:
                result = None

        return result
    
    except:
        
        return None

def ks(pair):

    assert len(pair) == 4 and isinstance(pair, tuple)

    try:
    
        result = None

        x, y, x_name, y_name = pair
      
        if len(x[y == 0]) > 0 and len(x[y == 1]) > 0:

            ks_result = ks_2samp(x[y == 0], x[y == 1])[1]

            result = {(x_name, y_name): ks_result}

        return result
    
    except Exception as e:
        
        print(e)
        
        return None

