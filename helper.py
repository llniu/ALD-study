import numpy as np
import pandas as pd
import sklearn


NP_LOG_FCT = np.log2

def log2(row: pd.Series):
    """Apply log Transformation to values."""
    return NP_LOG_FCT(row.where(row != 0.0))

RANDOMSEED = 123

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
    np.random.seed(RANDOMSEED)
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

def create_dichotome(series: pd.Series, cutoff_ge):
    """Define a dichtome (binary) variable from a continous feature."""
    return (series.dropna() >= cutoff_ge).astype(int)


class ThresholdClassifier(sklearn.base.BaseEstimator, sklearn.base.ClassifierMixin):
    """Classification based on predefined thresholds.

    Class can use several thresholds on variables in data.
    Class is defined in order to use previous CV-functionality.
    """
    def __init__(self, threshold={}, cutoff=0.5):
        self.cutoff = cutoff
        self.threshold = threshold

    def fit(self, data, target=None):
        self.markers = set(self.threshold.keys())
        """Nothing to fit"""
        if not self.markers.issubset(set(data.columns)): #len(self.markers & set(data.columns)) == len(self.markers):
            raise ValueError("Data does not contain all specified thresholds: {}".format(self.markers - set(data.columns)))
        if data.loc[:, self.threshold.keys()].isna().any().any():
            raise ValueError("Data does contain missing values. Please impute values.")

    def predict_proba(self, data):
        """Classify for each thresholds and then aggregate results
        by summation."""
        select_markers, thresholds = self.threshold.keys(), self.threshold.values()
        result = data.loc[:,select_markers] > list(thresholds)
        prob_c1 = result.mean(axis=1)
        prob_c0 = 1.0 - prob_c1
        result = pd.DataFrame({'prob c0': prob_c0, 'prob c1': prob_c1})
        return result.values

    def predict(self, data):
        """Predicts the class assignment based on the threshold provided or set."""
        prob = self.predict_proba(data)
        return (prob[:,1] >= self.cutoff).astype(int)


import pandas as pd
from sklearn.feature_selection import mutual_info_classif
from sklearn.feature_selection import SelectKBest

class FeatureSelector():
    """Namespace for feature selection.
    Uses mutal information to select k best features.
    Can combine the best for a set of targets to a combined maximum.

    Parameters
    ----------
    k: int
        top-k features for each endpoint
    protein_gene_data: pandas.DataFrame (shape: X_N, 1)
        Optional mapping of index of DataFrame passed to fit method
        to values in protein_gene_data. Here this is the associated gene-name
        to a protein.

    """

    def __init__(self, k=10, protein_gene_data=None):
        self.k = k
        self.protein_gene_id = protein_gene_data
        if protein_gene_data is not None:
            self.endpoints_features_ = pd.DataFrame()
        else:
            self.endpoints_features_ = None

    def fit(self, X: pd.DataFrame, y: pd.Series, col_name='target'):
        mask_samples_in_both = X.index.intersection(y.index)
        k_best = SelectKBest(mutual_info_classif, k=self.k)
        k_best.fit(X.loc[mask_samples_in_both], y=y.loc[mask_samples_in_both])

        selected_ = k_best.get_support()
        selected_ = X.columns[selected_]
        result = self.protein_gene_id.loc[selected_]
        result = result.fillna('NoGene')
        result.columns = [col_name]
        if self.endpoints_features_ is not None:
            self.endpoints_features_ = self.endpoints_features_.join(result, how='outer')
        else:
            print("Not able to aggregate as no protein_gene_data was passed.")
        return result


class MainExecutorCV():
    """Class to call cross-validation."""

    def __init__(self, proteomics_data, clinical_data, cutoffs_clinic, clf_sklearn, endpoints_defined=['F2', 'F3', 'S1', 'I2']):
        self.data_proteomics = proteomics_data
        self.data_clinic = clinical_data
        self.cutoffs_clinic = cutoffs_clinic
        self.endpoints_defined = endpoints_defined
        self.clf_sklearn = clf_sklearn

    @staticmethod
    def cutoff_classifier(cutoffs:dict)-> dict:
        """Takes a dictionary of key:cutoff values and returns
        univariate ThresholdClassifiers for each key-cutoff-pair."""
        clf_threshold = {}
        for key, value in cutoffs.items():
            clf_threshold[key] = ThresholdClassifier(threshold={key:value})
        return clf_threshold

    def run_evaluation(self, y:pd.Series, endpoint:str, additional_markers:list, proteins_selected:pd.Index, verbose=False):
        """Custom function to run standarda analysis for an endpoint based on
        predefined cutoffs, specified clinical variables"""
        assert endpoint in self.endpoints_defined

        cutoffs_endpoint = self.cutoffs_clinic[endpoint].dropna().to_dict()
        clf_endpoint_threshold = self.cutoff_classifier(cutoffs_endpoint)
        if verbose: display(clf_endpoint_threshold)

        X = self.data_clinic.loc[y.index, self.cutoffs_clinic[endpoint].keys()]
        if verbose: display(X.describe())

        results = {}
        auc_scores = {}

        for key, clf in clf_endpoint_threshold.items():
            _X = X[key].to_frame().dropna()
            _y = y.loc[_X.index].dropna()
            assert _X.isna().sum().sum() == 0
            assert _y.isna().sum() == 0

            _res, _auc_roc = run_cv_binary({f'{endpoint}_marker_{key}':clf}, X=_X, y=_y)
            results.update(_res)
            auc_scores.update(_auc_roc)

        for key in additional_markers:
            _X = X[key].to_frame().dropna()
            _y = y.loc[_X.index].dropna()
            assert _X.isna().sum().sum() == 0
            assert _y.isna().sum() == 0

            for key_clf, clf in self.clf_sklearn.items():
                _res, _auc_roc = run_cv_binary({f'{endpoint}_marker_{key}_{key_clf}':clf}, X=_X, y=_y)
                results.update(_res)
                auc_scores.update(_auc_roc)

        _X = self.data_proteomics[proteins_selected.index]
        in_both = y.index.intersection(_X.index)
        _X = _X.loc[in_both]
        _y = y.loc[in_both]

        _res, _auc_roc = run_cv_binary(self.clf_sklearn, X=_X, y=_y, prefix=f'{endpoint}_prot_')
        results.update(_res)
        auc_scores.update(_auc_roc)

        return results, auc_scores
