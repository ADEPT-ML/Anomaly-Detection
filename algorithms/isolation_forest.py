"""An implementation of the Isolation Forest shallow model."""
import pandas as pd
from sklearn.ensemble import IsolationForest

from .interface.algorithm_config import *
from .interface.algorithm_information import *
from .interface.algorithm_interface import *
from .util.shallow_model_util import remove_seasonality, post_processing


class Algorithm(AlgorithmInterface):
    """Algorithm class that contains the Isolation Forest implementation."""

    def __init__(self) -> None:
        self.info = AlgorithmInformation(name="Isolation Forest", deep=False, explainable=False)
        percentile_setting = SliderSetting(id="contamination",
                                           name="Contamination",
                                           description="The expected percentage of anomalies in the data.",
                                           default=1,
                                           step=0.1,
                                           lowBound=0.1,
                                           highBound=10)
        season_setting = ToggleSetting(id="season",
                                       name="Remove seasonality",
                                       description="Will remove the seasonality from the data if enabled.",
                                       default=True)
        self.config = AlgorithmConfig([percentile_setting, season_setting])

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

    def calc_anomaly_score(self, data: pd.DataFrame, building: str, config: dict) -> tuple[list, list, list, float]:
        """Calculates an anomaly score for the given data.

        Removes the seasonality from the given data.
        Fits the model and calculates anomaly scores on the adjusted model.
        Calculates the new indices and passes the resulting data to post-processing.

        Args:
            data: A dataframe containing a sensor data slice.
            building: The name of the building from which the data originates
            config: The user specified configuration data.

        Returns:
            An empty deep anomaly score, the calculated anomaly scores, the timestamps and the threshold.
        """
        contamination = config["contamination"] / 100
        model = IsolationForest(n_estimators=500, n_jobs=None, contamination=contamination)
        processed_data = data.apply(lambda x: remove_seasonality(x), axis=0).dropna() if config["season"] else data
        model.fit(processed_data)
        scores = model.decision_function(processed_data)
        indices = (data.iloc[336:-336] if config["season"] else data).index.tolist()
        return post_processing(scores, indices)
