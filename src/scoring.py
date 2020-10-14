
import pandas as pd
import sklearn.metrics as sklm


class ConfusionMatrix():
    """Wrapper for `sklearn.metrics.confusion_matrix`"""

    def __init__(self, y_true, y_pred):
        self.cm_ = sklm.confusion_matrix(y_true, y_pred)

    @property
    def as_dataframe(self):
        """Create pandas.DataFrame and return.
        Names rows and columns."""
        if not hasattr(self, 'df'):
            self.df = pd.DataFrame(self.cm_)
            self.df.index.name = 'true'
            self.df.columns.name = 'pred'
        return self.df

    @property
    def as_array(self):
        """Return sklearn.metrics.confusion_matrix array"""
        return self.cm_

    def __str__(self):
        """sklearn.metrics.confusion_matrix __str__"""
        return str(self.cm_)

    def __repr__(self):
        """sklearn.metrics.confusion_matrix __repr__"""
        return repr(self.cm_)
