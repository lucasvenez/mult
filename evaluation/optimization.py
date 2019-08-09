import numpy as np

def optimize_threshold(y_true, y_, by=0.01):

    t, max_metric = None, -np.inf

    for i in np.arange(0.00, max(y_), by):

        y_hat = np.copy(y_)

        filter__ = y_hat >= i

        y_hat[filter__], y_hat[~filter__] = 1, 0

        tn, fp, fn, tp = confusion_matrix(y_true, y_hat).ravel()

        sensitivity = (tp / float(tp + fn)) if tp + fn > 0 else 1

        specificity = (tn / float(tn + fp)) if tn + fp > 0 else 1

        ks = abs(sensitivity + specificity - 1.)
        
        auc = roc_auc_score(y_true, y_hat)
        
        metric = ks

        if metric > max_metric and metric is not np.inf:

            max_metric = metric

            t = i

    return t