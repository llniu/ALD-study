import numpy as np
from sklearn.metrics import roc_curve, auc


def plot_performance(ax, result, metric, title, _process_index=None):
    """Plot mean and standard deviation (stddev) of metrics.

    Parameters
    ----------
    ax : matplotlib.Axes
        Axes to draw on.
    result : pandas.DataFrame
        results. Rows are models. Each metric has a mean and stddev in a MultiIndex
        columns object of the type ('metric', ('mean', 'stddev'))
    metric : pandas.DataFrame
        The metric to select from the columns of the `result` DataFrame.
    title : str     
        Title of the axes
    _process_index : function, optional
        Function to process model names, by default None

    Returns
    -------
    matplotlib.Axes 
        Return reference to the passed ax of the argument `ax`
    """
    df = result.copy()
    df = df.sort_values(by=[(metric, 'mean')])
    colors = np.where(['prot' in row for row in df.index], 'darkred', 'white')
    if _process_index is not None:
        df.index = _process_index(df.index)
    y = df.index
    width = df[(metric, 'mean')]
    xerr = df[(metric, 'std')]
    ax.set_xlim(0, 1.1)
    ax.tick_params(labelsize=15)
    ax.barh(y=y, width=width, xerr=xerr, capsize=4,
            color=colors, height=0.6, edgecolor='black')

    metric_name = " ".join(metric.split('_')).capitalize()
    if metric == 'f1':
        metric_name += ' score'
    ax.set_title('{}\n{}'.format(title, metric_name), fontsize=15)
    return ax


def plot_roc_curve(ax, runs_roc_auc_scores, endpoint=''):
    tprs = []
    base_fpr = np.linspace(0, 1, 101)
    roc_aucs = []
    for fpr, tpr, threshold in runs_roc_auc_scores:
        roc_auc = auc(fpr, tpr)
        roc_aucs.append(roc_auc)

        ax.plot(fpr, tpr, 'royalblue', alpha=0.05)

        tpr = np.interp(base_fpr, fpr, tpr)
        tpr[0] = 0.0
        tprs.append(tpr)

    tprs = np.array(tprs)
    mean_tprs = tprs.mean(axis=0)
    std = tprs.std(axis=0)

    tprs_upper = mean_tprs + std
    tprs_lower = mean_tprs - std

    mean_rocauc = np.mean(roc_aucs).round(2)
    sd_rocauc = np.std(roc_aucs).round(2)
    se_rocauc = sd_rocauc/np.sqrt(len(roc_aucs))

    CI = (mean_rocauc-1.96 * se_rocauc, mean_rocauc + 1.96 * se_rocauc)

    ax.plot(base_fpr, mean_tprs, color='royalblue',
            label='Mean ROC\n(AUC = {}±{})'.format(mean_rocauc, sd_rocauc))
    ax.fill_between(base_fpr, tprs_lower, tprs_upper,
                    color='grey', alpha=0.4, label='±1 std. dev')

    ax.plot([0, 1], [0, 1], 'r--')
    ax.set_xlim([-0.01, 1.02])
    ax.set_ylim([-0.01, 1.02])
    ax.set_ylabel('True Positive Rate', fontsize=15)
    ax.set_xlabel('False Positive Rate', fontsize=15)
    ax.tick_params(labelsize=15)
    ax.legend(fontsize=12)
    ax.set_title('{}\nProteomic panel'.format(endpoint), fontsize=15)
    # print("95% CI:{}".format(CI))
    return ax


def plot_prc_curve(ax, runs_roc_auc_scores, endpoint=''):
    """[summary]

    Args:
        ax ([type]): [description]
        runs_roc_auc_scores ([type]): [description]
        endpoint (str, optional): [description]. Defaults to ''.
    """
    precisions = []
    base_recall = np.linspace(0, 1, 101)
    avg_precision = []
    for precision, recall, thresholds, _average_precision in runs_roc_auc_scores:

        avg_precision.append(_average_precision)

        ax.plot(recall, precision, 'royalblue', alpha=0.05)

        precision = np.interp(base_recall, recall[::-1], precision[::-1])
        precision[-1] = 0.0
        precisions.append(precision)
    ax.set(xlabel="Recall", ylabel="Precision")

    precisions = np.array(precisions)
    mean_precisions = precisions.mean(axis=0)
    std_precisions = precisions.std(axis=0)

    precisions_upper = mean_precisions + std_precisions
    precisions_lower = mean_precisions - std_precisions

    mean_avg_prec = np.mean(avg_precision).round(2)
    sd_avg_prec = np.std(avg_precision).round(2)
    se_avg_prec = sd_avg_prec/np.sqrt(len(avg_precision))

    CI = (mean_avg_prec-1.96 * se_avg_prec, mean_avg_prec + 1.96 * se_avg_prec)

    ax.plot(base_recall, mean_precisions, color='royalblue',
            label='Mean Avg. Prec.= {}±{})'.format(mean_avg_prec, sd_avg_prec))
    ax.fill_between(base_recall, precisions_lower, precisions_upper,
                    color='grey', alpha=0.4, label='±1 std. dev')

    ax.plot([0, 1], [1, 0], 'r--')
    ax.set_xlim([-0.01, 1.02])
    ax.set_ylim([-0.01, 1.02])
    ax.set_ylabel('Precision', fontsize=15)
    ax.set_xlabel('True Positive Rate', fontsize=15)
    ax.tick_params(labelsize=15)
    ax.legend(fontsize=12)
    ax.set_title('{}\nProteomic panel'.format(endpoint), fontsize=15)
    print("95% CI:{}".format(CI))
