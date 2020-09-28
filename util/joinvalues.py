import pandas as pd


def join_values(values):

    assert isinstance(values, list) or isinstance(values, tuple)

    result = None

    for v in values:

        if isinstance(v, pd.Series):
            v = pd.DataFrame(v)

        assert isinstance(v, pd.DataFrame)

        c = v.astype(str).apply(''.join, axis=1)

        result = c if result is None else result + c

    return result
