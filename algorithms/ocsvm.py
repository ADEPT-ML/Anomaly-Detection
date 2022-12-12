"""An implementation of the One-Class SVM shallow model."""
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.svm import OneClassSVM

from .interface.algorithm_config import *
from .interface.algorithm_information import *
from .interface.algorithm_interface import *
from .util.shallow_model_util import remove_seasonality, post_processing


class Algorithm(AlgorithmInterface):
    """Algorithm class that contains the One-Class SVM implementation."""

    def __init__(self) -> None:
        self.model = OneClassSVM(kernel="rbf", gamma=0.01, nu=0.01)
        self.scaler = StandardScaler()
        self.info = AlgorithmInformation(name="One-Class SVM", deep=False, explainable=False)
        self.config = AlgorithmConfig()

    @property
    def information(self) -> AlgorithmInformation:
        """Returns an AlgorithmInformation object containing all information regarding the algorithm.

        Returns:
            An AlgorithmInformation object containing all information regarding the algorithm.
        """
        return self.info

    @property
    def configuration(self) -> AlgorithmConfig:
        """Returns an AlgorithmConfig object containing all information regarding the config options for the algorithm.

        Returns:
            An AlgorithmConfig object containing all information regarding the config options.
        """
        return self.config

    def calc_anomaly_score(self, data: pd.DataFrame, config: dict) -> tuple[list, list, list, float]:
        """Calculates an anomaly score for the given data.

        Removes the seasonality and scales the given data.
        Fits the model and calculates anomaly scores on the adjusted model.
        Calculates the new indices and passes the resulting data to post-processing.

        Args:
            data: A dataframe containing a sensor data slice.
            config: The user specified configuration data.

        Returns:
            An empty deep anomaly score, the calculated anomaly scores, the timestamps and the threshold.
        """
        processed_data = data.apply(lambda x: remove_seasonality(x), axis=0).dropna()
        scaled_data = self.scaler.fit_transform(processed_data)
        self.model.fit(scaled_data)
        scores = self.model.decision_function(scaled_data)
        indices = data.iloc[336:-336].index.tolist()
        return post_processing(scores, indices)
