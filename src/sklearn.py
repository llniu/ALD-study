import pandas as pd

from sklearn.feature_selection import mutual_info_classif
from sklearn.feature_selection import SelectKBest

class FeatureSelector():
    """Namespace for feature selection.
    Uses mutal information to select k best features. 
    Can combine the best for a set of targets to a combined maximum.
    """
    def __init__(self, k=10, protein_gene_data=None):
        """Initialize FeatureSelector.

        Parameters
        ----------
        k : int, optional
            top-k features for each endpoint, by default 10
        protein_gene_data : pandas.DataFrame (shape: X_N, 1), optional
            Optional mapping of index of DataFrame passed to fit method
            to values in protein_gene_data. Here this is the associated gene-name
            to a protein., by default None
        """        
        self.k = k
        self.protein_gene_id = protein_gene_data
        self.endpoints_scores = {}
        
    def fit(self, X: pd.DataFrame, y: pd.Series, col_name='target'):
        mask_samples_in_both = X.index.intersection(y.index)
        k_best = SelectKBest(mutual_info_classif, k=self.k)
        k_best.fit(X.loc[mask_samples_in_both], y=y.loc[mask_samples_in_both])
        self.endpoints_scores[col_name] = pd.Series(k_best.scores_, index=X.columns, name=col_name)
        
        return self.get_k_best(col_name, self.k)
    
    def __getitem__(self, col_name):
        return self.endpoints_scores[col_name]
    
    def get_k_best(self, col_name, k):
        """Get the k-best features based on the analysis

        Parameters
        ----------
        col_name : str
            target column for comparison.
        k : int
            k-best features to be returned

        Returns
        -------
        pandas.DataFrame
            DataFrame with index containing the k-best features. 
            Ordered by value. If gene IDs are available, these
            are outputted instead of the Mutual Information score.
        """        
        selected_ = self.endpoints_scores[col_name].nlargest(k)
        if self.protein_gene_id is not None:
            result = self.protein_gene_id.loc[selected_.index]
            result = result.fillna('NoGene')
        else:
            result = selected_.to_frame()
        result.columns = [col_name]
        return result    
