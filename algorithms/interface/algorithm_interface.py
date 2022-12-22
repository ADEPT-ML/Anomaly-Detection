"""Contains the algorithm base class."""
from abc import ABC, abstractmethod

import pandas as pd

from .algorithm_config import *
from .algorithm_information import *

__all__ = ["AlgorithmInterface"]


class AlgorithmInterface(ABC):
    """The base class for all anomaly detection algorithms."""

    @property
    @abstractmethod
    def information(self) -> AlgorithmInformation:
        """Returns an AlgorithmInformation object containing all information regarding the algorithm.

        Returns:
            An AlgorithmInformation object containing all information regarding the algorithm.
        """
        pass

    @property
    @abstractmethod
    def configuration(self) -> AlgorithmConfig:
        """Returns an AlgorithmConfig object containing all information regarding the config options for the algorithm.

        Returns:
            An AlgorithmConfig object containing all information regarding the config options.
        """
        pass

    @abstractmethod
    def calc_anomaly_score(self, data: pd.DataFrame, building: str, config: dict) -> tuple[list, list, list, float]:
        """Calculates the anomaly score for the given data.

        Args:
            data: A dataframe containing a sensor data slice.
            building: The name of the building from which the data originates
            config: The user specified configuration data.

        Returns:
            The deep anomaly score if available, the calculated anomaly scores, the timestamps and the threshold.
        """
        pass
