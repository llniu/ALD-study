import sklearn
import pandas as pd


class ThresholdClassifier(sklearn.base.BaseEstimator, sklearn.base.ClassifierMixin):
    """Classification based on predefined thresholds.

    Class can use several thresholds on variables in data.
    Class is defined in order to use previous Cross-Validation-functionality
    from sklearn.
    """

    def __init__(self, threshold={}, cutoff=0.5):
        """Init ThresholdClassifier

        Parameters
        ----------
        threshold : dict, optional
            key-value pairs of thresholds for clinical markers, by default {}
        cutoff : float, optional
            [description], by default 0.5
        """
        self.cutoff = cutoff
        self.threshold = threshold

    def fit(self, data, target=None):
        """Misnomer for Threshold Classification. Nothing has to be fitted
        as the thresholds are predefined for each clinical marker.

        Performs some checks.

        Parameters
        ----------
        data : pandas.DataFrame
            Clincial data containing the measured values for clinical markers.
        target : str, optional
            Endpoint, which is not used as nothing is trained.
            Defined for compliance with interface, by default None

        Raises
        ------
        ValueError
            If clinical marker specified on construction is not part of the data.
        ValueError
            If clinical data contains missing values. 
        """
        #ToDo: Call checks from method.        
        self.markers = set(self.threshold.keys())
        """Nothing to fit"""
        if not self.markers.issubset(set(data.columns)):
            raise ValueError("Data does not contain all specified thresholds: {}".format(
                self.markers - set(data.columns)))
        if data.loc[:, self.threshold.keys()].isna().any().any():
            raise ValueError("Data does contain missing values. Please impute values.")

    def predict_proba(self, data):
        """Classify for each thresholds and then aggregate results 
        by taking the mean."""
        select_markers, thresholds = self.threshold.keys(), self.threshold.values()
        result = data.loc[:, select_markers] > list(thresholds)
        prob_c1 = result.mean(axis=1)
        prob_c0 = 1.0 - prob_c1
        result = pd.DataFrame({'prob c0': prob_c0, 'prob c1': prob_c1})
        return result.values

    def predict(self, data, cutoff=None):
        """Predicts the class assignment based on the threshold provided or set."""
        prob = self.predict_proba(data)
        if cutoff is None:
            cutoff = self.cutoff
        return (prob[:, 1] >= cutoff).astype(int)
