from abc import ABC, abstractmethod

import pandas as pd

from .algorithm_config import *
from .algorithm_information import *

__all__ = ["AlgorithmInterface"]


class AlgorithmInterface(ABC):
    @property
    @abstractmethod
    def information(self) -> AlgorithmInformation:
        pass

    @property
    @abstractmethod
    def configuration(self) -> AlgorithmConfig:
        pass

    @abstractmethod
    def calc_anomaly_score(self, data: pd.DataFrame) -> tuple[list, list, list, float]:
        pass
