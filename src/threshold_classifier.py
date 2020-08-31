import sklearn
import pandas as pd

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
