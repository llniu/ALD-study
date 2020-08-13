import pandas as pd

def combine_value_counts(X:pd.DataFrame):
    """Pass a selection of columns to combine it's value counts.
    
    This performs no checks. Make sure the scale of the variables
    you pass is comparable.
    """
    _df = pd.DataFrame()
    for col in X.columns:
        _df[col] = X[col].value_counts()
    freq_targets = _df.sort_index()
    return freq_targets

def create_dichotome(series: pd.Series, cutoff_ge):
    """Define a dichtome (binary) variable from a continous feature."""
    return (series.dropna() >= cutoff_ge).astype(int)