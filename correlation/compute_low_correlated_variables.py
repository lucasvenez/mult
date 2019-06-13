def select_variables_based_on_pearson(file_path='data/output/genetic_correlation.tsv'):

    variables, to_be_deleted = set(), set()

    with open(file_path) as file:

        count_var, previous_var = 0, None

        for line in file:

            try:

                var1, pearson, var2 = line.split('\t')

                var2 = var2.replace('\n', '')

                if pearson != '':

                    pearson = abs(float(pearson))

                    if pearson >= .75:
                        to_be_deleted = to_be_deleted.union({var2})

                    if previous_var != var1:

                        variables = variables.union({var1})

                        count_var += 1

                        print('Computing var {} of {}'.format(count_var, 27167))

                    previous_var = var1

            except Exception as e:
                pass

    import pandas as pd

    return pd.DataFrame({'variable': list(variables.difference(to_be_deleted))})
