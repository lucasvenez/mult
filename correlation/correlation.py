from minepy import MINE
#from rpy2.robjects.packages import importr

import multiprocessing
import pandas as pd
#import pandas.rpy.common as com
import scipy.spatial.distance as distance
import scipy.stats as statistical

#stats = importr('stats')
#base = importr('base')


def compute(dataset, dependent_variable_names, func):

    assert isinstance(dependent_variable_names, list)
    assert isinstance(dataset, pd.DataFrame) and dataset.shape[0] > 0
    assert callable(func)

    dependent_variables = dataset[dependent_variable_names]

    for name in dependent_variable_names:
        del dataset[name]

    num_cores = multiprocessing.cpu_count() - 1

    inputs = [(independent_var.as_matrix(), dependent_var.as_matrix(), independent_var.name, dependent_var.name)
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


def mic(pair):

    assert len(pair) == 4 and isinstance(pair, tuple)

    x, y, x_name, y_name = pair

    mine = MINE()

    mine.compute_score(x, y)

    return {(x_name, y_name): mine.mic()}


def mine(pair):

    assert len(pair) == 4 and isinstance(pair, tuple)

    x, y, x_name, y_name = pair

    mine_ = MINE()

    mine_.compute_score(x, y)

    result = {(x_name, y_name): (mine_.mic(), mine_.mas(), mine_.mcn(), mine_.mev(), mine_.tic())}

    return result


def wilcox(pair):

    assert len(pair) == 4 and isinstance(pair, tuple)

    result = None

    x, y, x_name, y_name = pair

    if len(x[y == 0] > 0) and len(x[y == 1]) > 0:

        result_greater = stats.wilcox_test(base.as_numeric(x[y == 0].tolist()),
                                           base.as_numeric(x[y == 1].tolist()), alternative='greater')[2]

        result_less = stats.wilcox_test(base.as_numeric(x[y == 0].tolist()),
                                        base.as_numeric(x[y == 1].tolist()), alternative='less')[2]

        wilcox_result = min(float(result_greater[0]), float(result_less[0]))

        result = {(x_name, y_name): wilcox_result}

    return result


def kruskal(pair):

    assert len(pair) == 4 and isinstance(pair, tuple)

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


def distcorr(pair):

    assert len(pair) == 4 and isinstance(pair, tuple)

    X, Y, x_name, y_name = pair

    result = None

    if len(pd.unique(Y)) > 1:

        dcor = distance.correlation(X, Y)

        result = {(x_name, y_name): dcor}

    return result


def pearson(pair):

    assert len(pair) == 4 and isinstance(pair, tuple)

    X, Y, x_name, y_name = pair

    result = None

    if len(pd.unique(Y)) > 1:
        try:
            result = {(x_name, y_name): statistical.pearsonr(X, Y)[0]}
        except:
            result = None

    return result
