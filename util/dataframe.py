import pandas as pd


def to_data_frame(x, prefix, index):
    
    assert len(x) == len(index), 'x and index shapes should match, but they have {} and {}, repsectively'.format(len(x), len(index))
    
    assert prefix is not None, 'prefix should not be None'
    
    df = pd.DataFrame(x, index=index)
    
    df.columns = [prefix + str(i) for i in range(x.shape[1])]
    
    return df