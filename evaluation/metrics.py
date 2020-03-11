import pandas as pd
import numpy as np

def classification_metrics(tn, fp, fn, tp):
    
    sensitivity = (tp / float(tp + fn)) if tp + fn > 0 else 1

    precision =  (tp / float(tp + fp)) if tp + fp > 0 else 1

    specificity = (tn / float(tn + fp)) if tn + fp > 0 else 1

    ks = abs(sensitivity + specificity - 1.)

    ifp = (float(tp + fp) / tp) if tp > 0 else -np.inf

    accuracy = (tp + tn) / (tp + tn + fp + fn)
    
    return {'accuracy': accuracy, 'precision': precision, 'sensitivity': sensitivity, 'specificity': specificity}

def ks_score(y_true, y_hat):
    
    try:
        
        deciles = [-1] + list(np.quantile(y_hat, [.1, .2, .3, .4, .5, .6, .7, .8, .9, 1]))

        result = []

        for i in y_hat:
            for index, (a, b) in enumerate(zip(deciles[:-1], deciles[1:])):
                if a < i <= b:
                    result.append(index + 1)
                    continue

        result = pd.DataFrame({'decile': result})

        result['positive'] = y_true
        result['negative'] = 1 - y_true

        negative_sum = result['negative'].sum()
        positive_sum = result['positive'].sum()

        result = result.groupby(by=['decile']).sum()

        result['positive_percentage'] = result['positive'] / positive_sum
        result['negative_percentage'] = result['negative'] / negative_sum

        result['positive_percentage_cumsum'] = result['positive_percentage'].cumsum()
        result['negative_percentage_cumsum'] = result['negative_percentage'].cumsum()

        result['ks'] = np.abs(result['positive_percentage_cumsum'] - result['negative_percentage_cumsum'])

        return result['ks'].max()
    
    except ValueError:
        return 0.0