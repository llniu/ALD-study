import os
import logging
logger = logging.getLogger()

from collections import defaultdict
from collections.abc import Iterable

import numpy as np
import pandas as pd

from sklearn.model_selection import cross_validate
from sklearn.preprocessing import PolynomialFeatures

import sklearn.metrics as sklm
from sklearn.metrics import roc_curve, auc
from sklearn.metrics import precision_recall_curve
from sklearn.model_selection import RepeatedStratifiedKFold

from src.threshold_classifier import ThresholdClassifier

CV_FOLDS = 5
CV_REPEATS = 10
RANDOM_SEED = 123


def run_cv_binary_simple(clf_dict: dict, X: pd.DataFrame, y: pd.Series, cv=5,
                         scoring=['precision', 'recall', 'f1',
                             'balanced_accuracy', 'roc_auc'],
                         prefix='',
                         return_estimator=False) -> dict:
    """Run Cross Validation (cv) for binary classification example
    for a set of classifiers.


    Inputs
    ------
    clf_dict: dict
        Dictionary with keys and scikit-learn classifiers as values.
    X: 2D-array, pd.DataFrame
        Input data
    y: 1D-array, pd.Series
        Targets for classification
    cv: int
        Number of splits for Cross-Validation.
    prefix: str
        Prefix for clf-key for custom naming.

    Returns
    -------
    dict: dict with keys of clf_dict and computed results for each run.
    """
    cv_results = {}
    roc_curve_results = {}
    for key, clf in clf_dict.items():
        key = prefix + key
        cv_results[key] = cross_validate(clf, X, y=y, cv=cv, scoring=scoring,
                                         return_estimator=return_estimator)
        cv_results[key]['num_feat'] = X.shape[-1]
        cv_results[key]['n_obs'] = len(y)
    return cv_results

#ToDo: Write a (doc)test and or see if this is can be done differently


def _get_cv_means(results_dict: dict) -> pd.DataFrame:
    """Convert results of runs to averages and standard deviation.

    Parameters
    ----------
    results_dict : dict
        Takes as input a dictionary where each key holds a list of results.
        Normally the number of results should be the same, but it is not envforced.

            {'model1': {'metric1': array(value1, value2, value3),
                        'metric2': array(value1, value2, value3)},
            {'model2': {'metric1': array(value1, value2, value3),
                        'metric2': array(value1, value2, value3)}

    Returns
    -------
    pd.DataFrame
        pandas.DataFrame holding the results.
    """
    results = pd.DataFrame(results_dict)

    if 'estimator' in results.index:
        results = results.drop('estimator')

    cv_means = results.applymap(np.mean).T
    cv_std = results.applymap(np.std).T

    # is there a pandas way?
    order = list(cv_means.columns)
    columns = []
    for x in order:
        columns += [x, x + '_std']

    cv_results = cv_means.join(cv_std, rsuffix='_std', sort=True)
    cv_results = cv_results[columns]

    levels = [cv_means.columns, ['mean', 'std']]
    multi_index = pd.MultiIndex.from_product(
        levels, names=['variable', 'statistics'])
    cv_results.columns = multi_index
    return cv_results


scorer_dict = {}
scoring = ['precision', 'recall', 'f1', 'balanced_accuracy', 'roc_auc']
scorer_dict = {metric: metric+'_score' for metric in scoring}
scorer_dict = {key: getattr(sklm, metric)
                            for key, metric in scorer_dict.items()}


def run_cv_binary(clf_dict: dict, X: pd.DataFrame, y: pd.Series,
                  scoring=scorer_dict,
                  cv=None,
                  verbose=False,
                  prefix='',
                  folder: str = None,
                  save_predictions: bool = False) -> dict:
    """Run Cross Validation (cv) for binary classification example
    for a set of classifiers.


    Parameters
    ----------
    clf_dict : dict
        Dictionary with keys and scikit-learn classifiers as values.
    X : 2D-array, pd.DataFrame
        Input data
    y : 1D-array, pd.Series
        Targets for classification
    splits : List of pandas.Index tuples denoting global training and test indices.
    cv : int, cross-validation generator or an iterable, default=None
        Number of splits for Cross-Validation.
    prefix : str
        Prefix for clf-key for custom naming.

    Returns
    -------
    dict: dict with keys of clf_dict and computed results for each run.
    """
    cv_results = {}
    roc_curve_results = defaultdict(list)
    precision_recall_results = defaultdict(list)

    if cv is None:
        raise ValueError(
            "Please provide an Iterable of pandas.Index tuples or integer")
    elif isinstance(cv, int):
         rskf = RepeatedStratifiedKFold(
             n_splits=cv, n_repeats=CV_REPEATS, random_state=RANDOM_SEED)
         cv_train_test_indices = rskf.split(X, y)
         cv_train_test_indices = [(X.index[train_indices], X.index[test_indices])
                                   for train_indices, test_indices in cv_train_test_indices]
         logger.warning(
             'Splits based on provided data to fit, not globally. Do not compare between models.')
    elif isinstance(cv, Iterable):
        # assert isinstance(cv, Iterable)
        cv_train_test_indices = cv

    for key_clf, clf in clf_dict.items():
        key_clf = prefix + key_clf

        _cv_results = defaultdict(list)

        for i, (train_index, test_index) in enumerate(cv_train_test_indices):
            X_train = X.loc[X.index.intersection(train_index)]
            X_test = X.loc[X.index.intersection(test_index)]
            y_train = y.loc[y.index.intersection(train_index)]
            y_test = y.loc[y.index.intersection(test_index)]

            # drop-na only here, not before passing to the CV helper fct
            # this will garuantuee that for each run the clf are
            # trained at least on a precisly defined subset

            clf.fit(X_train, y_train)
            y_pred = clf.predict(X_test)
            y_score = clf.predict_proba(X_test)

            if save_predictions:
                target_comp_df = pd.DataFrame(
                    {'y_test': y_test, 'y_test_pred':  y_score[:, 1]})
                _df = pd.DataFrame(
                    {'y_train': y_train, 'y_train_pred': clf.predict_proba(X_train)[:, 1]})
                target_comp_df = target_comp_df.join(_df, how='outer')

                if folder == None:
                    folder = 'model_scores'
                os.makedirs(folder, exist_ok=True)
                if i == 0:
                    _fname = os.path.join(folder, f'{key_clf}.csv')
                else:
                    _fname = os.path.join(folder, f'{key_clf}_{i}.csv')
                target_comp_df.to_csv(_fname)

            for metric_name, metric_fct in scorer_dict.items():
                if metric_name == 'roc_auc':
                    _cv_results[metric_name].append(
                        metric_fct(y_test, y_score[:, 1]))
                else:
                    _cv_results[metric_name].append(metric_fct(y_test, y_pred))

            _cv_results['num_feat'].append(X.shape[-1])
            _cv_results['n_obs'].append(len(y))

            #additonal features requested: set verbose
            if verbose:
                _cv_results['prop_y_train'].append(y_train.mean())
                _cv_results['prop_y_test'].append(y_test.mean())

                _cv_results['y_test'].append(
                    pd.Series(y_score[:, 1], index=X_test.index))

            # save fpr, tpr and cutoffs
            # roc_auc_2 will be the same as roc_auc
            fpr, tpr, cutoffs = roc_curve(y_test, y_score[:, 1])
            roc_curve_results[key_clf].append((fpr, tpr, cutoffs))
            _cv_results['roc_auc_2'].append(auc(fpr, tpr))

            precision, recall, thresholds = precision_recall_curve(
                y_test, y_score[:, 1])
            average_precision = sklm.average_precision_score(
                y_test, y_score[:, 1])
            precision_recall_results[key_clf].append(
                (precision, recall, thresholds, average_precision))

        cv_results[key_clf] = dict(_cv_results)

    return cv_results, dict(roc_curve_results), dict(precision_recall_results)


class MainExecutorCV():
    """Class to call cross-validation."""

    def __init__(self,
                 proteomics_data: pd.DataFrame,
                 clinical_data: pd.DataFrame,
                 cutoffs_clinic: pd.DataFrame,
                 clf_sklearn: dict,
                 demographics: pd.DataFrame,
                 endpoints_defined: Iterable = ['F2', 'F3', 'S1', 'I2']):
        """Executor of Cross Validation of this project. Can be seen as a stateful main function.

        Parameters
        ----------
        proteomics_data : pd.DataFrame
            Proteomics data for samples. Expected to be imputed and without missings.
            Rows: Samples, Columns: Protein Intensities
        clinical_data : pd.DataFrame
            Clinical features. Should include features specified by `cutoffs_clinc`
            and `demographics`.
        cutoffs_clinic : pd.DataFrame
            [description]
        clf_sklearn : dict
            Dictionary with sklearn classifiers to consider.
            {model-key: sklearn-model-instance}
        demographics : pd.DataFrame
            `pandas.DataFrame` holding additional (demographic) features for samples.
        endpoints_defined : list, optional
            [description], by default ['F2', 'F3', 'S1', 'I2']
        """
        self.data_proteomics = proteomics_data
        self.data_clinic = clinical_data
        self.cutoffs_clinic = cutoffs_clinic
        self.demographics = demographics.dropna()
        self.endpoints_defined = endpoints_defined
        self.clf_sklearn = clf_sklearn

    @staticmethod
    def cutoff_classifier(cutoffs: dict) -> dict:
        """Takes a dictionary of key:cutoff values and returns
        univariate ThresholdClassifiers for each key-cutoff-pair."""
        clf_threshold = {}
        for key, value in cutoffs.items():
            clf_threshold[key] = ThresholdClassifier(threshold={key: value})
        return clf_threshold

    def run_evaluation(self,
                       y: pd.Series,
                       endpoint: str,
                       additional_markers: list,
                       proteins_selected: pd.Index,
                       add_demographics=False,
                       interactions_degree=1,
                       verbose=False,
                       evaluator_fct=run_cv_binary,
                       cv=CV_FOLDS, n_repeats=CV_REPEATS):
        """Custom function to run standarda analysis for an endpoint based on
        predefined cutoffs, specified clinical variables"""
        assert endpoint in self.endpoints_defined, 'Unknown endpoint. Select one of '
        f'{", ".join(self.endpoints_defined)}'

        cutoffs_endpoint = self.cutoffs_clinic[endpoint].dropna().to_dict()
        clf_endpoint_threshold = self.cutoff_classifier(cutoffs_endpoint)
        if verbose:
            display(clf_endpoint_threshold)

        X = self.data_clinic.loc[y.index, self.cutoffs_clinic[endpoint].keys()]
        if verbose:
            display(X.describe())

        results = {}
        auc_scores = {}
        prc_scores = {}

        # Threshold Classifier (single clinical variable)
        for key, clf in clf_endpoint_threshold.items():
            _X = X[key].to_frame().dropna()
            _y = y.loc[_X.index].dropna()
            assert _X.isna().sum().sum() == 0
            assert _y.isna().sum() == 0

            _res, _auc_roc, _auc_prc = evaluator_fct(
                {f'{endpoint}_marker_{key}': clf}, X=_X, y=_y, cv=cv)
            results.update(_res)
            auc_scores.update(_auc_roc)
            prc_scores.update(_auc_prc)

        #additional marker models (single clinical variable, trained)
        #add cutoff models based on data
        #ToDo: Explain that all markers are now trained (thresholds are adapted)
        additional_markers.extend(list(cutoffs_endpoint.keys()))
        for key in additional_markers:
            _X, _y = self._select_features(X[key], y, add_demographics)

            for key_clf, clf in self.clf_sklearn.items():
                _res, _auc_roc, _auc_prc = evaluator_fct(
                    {f'{endpoint}_marker_{key}_{key_clf}': clf}, X=_X, y=_y, cv=cv)
                results.update(_res)
                auc_scores.update(_auc_roc)
                prc_scores.update(_auc_prc)

        #proteomics models (based on provided protein selection)
        _X, _y = self._select_features(
            self.data_proteomics[proteins_selected.index], y, add_demographics)

        #Add interaction to _X
        if interactions_degree > 1:
            assert isinstance(interactions_degree, int), "Please pass an interaction_degree of type int, not {}".format(
                type(interactions_degree))
            poly_features = PolynomialFeatures(
                degree=interactions_degree, include_bias=False)
            _X = pd.DataFrame(poly_features.fit_transform(_X), index=_X.index)

        _res, _auc_roc, _auc_prc = evaluator_fct(
            self.clf_sklearn, X=_X, y=_y, prefix=f'{endpoint}_prot_', cv=cv)
        results.update(_res)
        auc_scores.update(_auc_roc)
        prc_scores.update(_auc_prc)

        return results, auc_scores, prc_scores

    def _select_features(self, X, y, add_demographics):
            if isinstance(X, pd.Series):
                X = X.to_frame()
            _X = X.dropna()
            in_both = y.index.intersection(_X.index)
            _X = _X.loc[in_both]
            _y = y.loc[in_both]

            if add_demographics:
                _index_tmp = _X.index
                _X = _X.join(self.demographics).dropna()
                _y = y.loc[_X.index]
                _intersection, _diff_to_1 = self._get_common_indices(_index_tmp, _X.index)

            assert _X.isna().sum().sum() == 0
            assert _y.isna().sum() == 0

            return _X, _y

    @staticmethod
    def _get_common_indices(index_1, index_2):
        _intersection = index_1.intersection(index_2)
        _diff_to_1 = index_1.difference(_intersection)
        if len(_diff_to_1) > 0:
            logger.warning("Sample with clinical features not in demographics: {}".format(", ".join(_diff_to_1)))
        return _intersection, _diff_to_1
