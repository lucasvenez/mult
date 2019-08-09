import pandas as pd

def classification_metrics(t, auc, tn, fp, fn, tp, title='Metric'):
    
    sensitivity = (tp / float(tp + fn)) if tp + fn > 0 else 1

    precision =  (tp / float(tp + fp)) if tp + fp > 0 else 1

    specificity = (tn / float(tn + fp)) if tn + fp > 0 else 1

    ks = abs(sensitivity + specificity - 1.)

    ifp = (float(tp + fp) / tp) if tp > 0 else -np.inf

    accuracy = (tp + tn) / (tp + tn + fp + fn)
    
    row = pd.DataFrame({title: [t], 'AUC': auc, 'Accuracy': accuracy, 
                        'Precision': precision, 'Sensitivity': sensitivity, 'Specificity': specificity,
                        'KS': ks, 'IFP': ifp})
    
    return row    