import pandas as pd


class FinalPredictor():
    """Documenting what is needed for predicting from final models."""

    def __init__(self,
                 data_clinic: pd.DataFrame,
                 data_proteomics: pd.DataFrame,
                 final_models: dict,
                 features_dict: dict,
                 endpoints: dict):
        """Assign data and final models for predictions.

        Parameters
        ----------
        data_clinic: pd.DataFrame
            Full clinical data
        data_proteomics: pd.DataFrame
            full proteomics data. These will be used to predict.
        final_models: dict
            Dictornary containing the final prediction models of proteomics data.
        features_dict: dict
            Custom dictionary containing feature information by endpoint (-> endpoints)
            Each endpoint contains a 'proteins' keyword which is a Series of
            protein-values, which index is used to
            select the features. -> Entirelly internal from src.sklearn.FeatureSelector
        endpoints: list
            Selection of endpoints to consider. Endpoints should be part of features_dict.
        """
        self.data_cli_full = data_clinic
        self.data_proteomics = data_proteomics
        assert set(endpoints) <= set(
            features_dict), 'Missing endpoints in features_dict: {}'.format(
            ", ".join(set(endpoints) - set(features_dict))
        )
        assert len(set(final_models) - set(endpoints)
                   ) == 0, 'Missing final model for endpoints: {}'.format(
                       ", ".join(set(final_models) - set(endpoints))
        )
        self.features_dict = features_dict
        self.final_models = final_models
        self.endpoints = endpoints

    def predict(self, indices):
        """Predict given set of models by init.

        indices: pd.Index
            Selection of samples to predict on proteomics data.
        """
        return self._predict(indices, self.predict_series)

    def predict_score(self, indices):
        """Predict model score given set of models by init.

        indices: pd.Index
            Selection of samples to predict on proteomics data.
        """
        return self._predict(indices, self.predict_score_series)

    def _predict(self, indices, fct):
        """[summary]

        Parameters
        ----------
        indices : pandas.Index
            set of sample indices
        fct : function
            Function to process a series of prediciton. Probably assigning
            indices: self.predict_series.

        Returns
        -------
        pandas.DataFrame
            Return a predictions for all endpoints.
        """
        """wrap sklearn prediction fct."""
        _indices = self.data_proteomics.index.intersection(indices)
        # #ToDo add logging of discarded samples
        _df = pd.DataFrame()
        for endpoint in self.endpoints:
            _data = self.data_proteomics.loc[
                _indices,
                self.features_dict[endpoint]['proteins'].index]
            _df = _df.join(
                fct(self.final_models[endpoint], _data, name=endpoint), how='outer')
        return _df

    @staticmethod
    def predict_series(model, data: pd.DataFrame, name='prediction') -> pd.Series:
        """Wrapper for sklearn model.predict. Helper function to assign
        the index of the samples to the predictions of the sklearn model provided.

        model: sklearn.Predictor
            sklearn predictor able to predict for passed data.
        data: pd.DataFrame
            DataFrame to be passed to model. Check that features
            correspond to what the model was trained on.
        name: str
            Name of the returned pd.Series

        Returns
        -------
        pandas.Series
            Prediction of binary class assignment (0 vs 1)
        """
        return pd.Series(model.predict(data), index=data.index, name=name)

    @staticmethod
    def predict_score_series(model, data: pd.DataFrame, name='prediction') -> pd.Series:
        """Wrapper for model.predict. Helper function to assign
        the index of the samples to the predictions of the models.

        model : sklearn.Predictor
            sklearn predictor able to predict for passed data.
        data : pd.DataFrame
            DataFrame to be passed to model. Check that features
            correspond to what the model was trained on.
        name : str, optional
            Name of the returned pd.Series, by default 'prediction'

        Returns
        -------
        pandas.Series
            Prediction scores for being in class 1 (having a disease)
        """
        return pd.Series(model.predict_proba(data)[:, 1], index=data.index, name=name)
