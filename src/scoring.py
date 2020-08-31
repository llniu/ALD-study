
import pandas as pd
import sklearn
import sklearn.metrics as sklm

class ConfusionMatrix():
    """Wrapper for `sklearn.metrics.confusion_matrix`"""
    def __init__(self, y_true, y_pred):
        self.cm_ = sklm.confusion_matrix(y_true, y_pred)
    
    @property
    def as_dataframe(self):
        if not hasattr(self, 'df'):
            self.df = pd.DataFrame(self.cm_)
            self.df.index.name = 'true'
            self.df.columns.name = 'pred'
        return self.df
    
    @property
    def as_array(self):
        return self.cm_
    
    def __str__(self):
        return str(self.cm_)
    
    def __repr__(self):
        return repr(self.cm_)
