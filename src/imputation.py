"""Imputation of proteomics data."""
import numpy as np
import pandas as pd

NP_LOG_FCT = np.log2

def log2(row: pd.Series):
    """Apply log Transformation to values."""
    return NP_LOG_FCT(row.where(row != 0.0))

RANDOM_SEED = 123

IMPUTATION_MEAN_SHIFT    = 1.8
IMPUTATION_STD_SHRINKAGE = 0.3

def imputation_normal_distribution(log_intensities: pd.Series, mean_shift=IMPUTATION_MEAN_SHIFT, std_shrinkage=IMPUTATION_STD_SHRINKAGE):
    """Impute missing log-transformed intensity values of DDA run.

    Parameters
    ----------
    log_intensities: pd.Series
        Series of normally distributed values. Here usually log-transformed
        protein intensities.
    mean_shift: integer, float
        Shift the mean of the log_intensities by factors of their standard
        deviation to the negative.
    std_shrinkage: float
        Value greater than zero by which to shrink (or inflate) the 
        standard deviation of the log_intensities.
    """
    np.random.seed(RANDOM_SEED)
    if not isinstance(log_intensities, pd.Series):
        try:
            log_intensities.Series(log_intensities)
            logger.warning("Series created of Iterable.")
        except:
            raise ValueError(
                "Plese provided data which is a pandas.Series or an Iterable")
    if mean_shift < 0:
        raise ValueError(
            "Please specify a positive float as the std.-dev. is non-negative.")
    if std_shrinkage <= 0:
        raise ValueError(
            "Please specify a positive float as shrinkage factor for std.-dev.")
    if std_shrinkage >= 1:
        logger.warning("Standard Deviation will increase for imputed values.")

    mean = log_intensities.mean()
    std = log_intensities.std()

    mean_shifted = mean - (std * mean_shift)
    std_shrinked = std * std_shrinkage

    return log_intensities.where(log_intensities.notna(),
                                 np.random.normal(mean_shifted, std_shrinked))