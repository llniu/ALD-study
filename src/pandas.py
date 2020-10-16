import pandas as pd


def combine_value_counts(X: pd.DataFrame, dropna=True):
    """Pass a selection of columns to combine it's value counts.

    This performs no checks. Make sure the scale of the variables
    you pass is comparable.


    Parameters
    ----------
    X: pandas.DataFrame
        DataFrame (view) with columns which value counts should be
        concatenated.
    dropna : bool, optional
        Whether to keep missing values in a column, by default True
        This does mean the result has not NA index.

    Returns
    -------
    X: pandas.DataFrame
        DataFrame (view) with columns which value counts should be
        concatenated.
    """
    _df = pd.DataFrame()
    for col in X.columns:
        _df = _df.join(X[col].value_counts(dropna=dropna), how='outer')
    freq_targets = _df.sort_index()
    return freq_targets


def create_dichotome(series: pd.Series, cutoff_ge):
    """Define a dichtome (binary) variable from a continous feature.

    Parameters
    ----------
    series : pd.Series
        pandas.Series with values to dicotomize
    cutoff_ge : int, float
        Greater equal (ge) for which a one is assigned.
        Everything below will be set to zero.

    Returns
    -------
    pandas.Series
        Series with assignement based on cutoff to zero or one.
    """
    return (series.dropna() >= cutoff_ge).astype(int)
