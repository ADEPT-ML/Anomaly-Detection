from dataclasses import dataclass
from abc import ABC, abstractmethod
import pandas as pd


@dataclass
class AlgorithmInformation():
    name: str
    deep: bool
    explainable: bool
class AlgorithmInterface(ABC):
    @property
    @abstractmethod
    def information(self) -> AlgorithmInformation:
        pass

    @abstractmethod
    def calc_anomaly_score(self, data: pd.DataFrame) -> tuple[list, list, list, float]:
        pass
