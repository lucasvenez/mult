from minepy import MINE
from scipy.stats import ks_2samp
#from rpy2.robjects.packages import importr

import multiprocessing
import pandas as pd
#import pandas.rpy.common as com
import scipy.spatial.distance as distance
import scipy.stats as statistical

#stats = importr('stats')
#base = importr('base')


def select_genes(genes, response, threshold=0.05):
    #
    # Gene Selection
    #
    
    excluded_genes = []
    
    gene_table = {'variable': [], 'score': []}

    response = response.values
    
    for c in genes:

        gene_values = genes[c].values

        if len(gene_values[response == 0]) > 0 and len(gene_values[response == 1]) > 0:

            ks_pvalue = ks_2samp(gene_values[response == 0], gene_values[response == 1])[1]

            gene_table['variable'].append(c)

            gene_table['score'].append(ks_pvalue)

    gene_table = pd.DataFrame(gene_table).set_index('variable')
    
    selected_genes = list(gene_table[gene_table['score'] < threshold].sort_values(by='score').index)
    
    pairwise_pearson = genes[selected_genes].corr().abs()
    
    for gi, g in enumerate(selected_genes[:-1]):
        
        gene_pearson = pairwise_pearson.loc[[g],:].iloc[:, (gi+1):]
        
        excluded_genes += list(gene_pearson.loc[:,(gene_pearson.values > .75)[0]].columns)
    
    return set(selected_genes).difference(set(excluded_genes))

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

