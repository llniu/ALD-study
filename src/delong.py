"""Helper function based on delong package by yandexdataschool:
https://github.com/yandexdataschool/roc_comparison
"""
import os
import logging


import pandas as pd

from roc_comparison import compare_auc_delong_xu

logger = logging.getLogger()
logger.setLevel(logging.INFO)


def calc_p_value_delong_xu(model_1: str, model_2: str, folder_dumps: str,
                           verbose=False, index_col='Sample ID'):
    """Calculate p_value based on the a previous dump of model scores.

    Parameters
    ----------
    model_1 : str
        model_name of model contained folder_dumps to  be loaded from disk.
    model_2 : str
        model_name of model contained folder_dumps to  be loaded from disk.
    folder_dumps : str
        Path du model dumps with `index_col` saved. The index is used to find
        the common subset of samples on which the test can be based.
    verbose : bool, optional
        logging logging.INFO statements, by default False
    index_col : str, optional
        Index column to be used when loading data into a
        pandas.DataFrame, by default 'Sample ID'

    Returns
    -------
    float
         p-value for DeLong AUC-ROC test.
    """
    model_1_name = " ".join(model_1.split('.csv')[0].split('_'))
    model_2_name = " ".join(model_2.split('.csv')[0].split('_'))
    if verbose:
        logger.info("Compare {} to {}\n".format(model_1_name, model_2_name))

    # compare_auc_delong_xu.delong_roc_test()
    model_1 = pd.read_csv(os.path.join(
        folder_dumps, model_1), index_col=index_col)
    model_2 = pd.read_csv(os.path.join(
        folder_dumps, model_2), index_col=index_col)
    in_both = model_2.y_test.dropna().index.intersection(
        model_1.y_test.dropna().index)

    model_1_omitted_ids = model_1.y_test.dropna().index.difference(in_both)

    model_2_omitted_ids = model_2.y_test.dropna().index.difference(in_both)
    if verbose:
        logger.info("Omitting {} from test set of {}: {}\n".format(
            len(model_1_omitted_ids), model_1_name, ', '.join(model_1_omitted_ids)))
        logger.info("Omitting {} from test set of {}: {}\n".format(
            len(model_2_omitted_ids), model_2_name, ', '.join(model_2_omitted_ids)))
        logger.info("Comparison based on {} in total, which are: {}".format(
            len(in_both), ", ".join(in_both)))

    assert all(model_1.loc[in_both].y_test.dropna() ==
               model_1.loc[in_both].y_test.dropna())

    log10_pvalue = compare_auc_delong_xu.delong_roc_test(
        ground_truth=model_1.loc[in_both].y_test.dropna(),
        predictions_one=model_2.y_test_pred.loc[in_both],
        predictions_two=model_1.y_test_pred.loc[in_both]
    )

    return 10**log10_pvalue[0][0]
