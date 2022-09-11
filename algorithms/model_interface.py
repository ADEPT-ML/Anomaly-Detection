from abc import ABC, abstractmethod
import pandas as pd
import numpy as np


class ModelInterface(ABC):

    @abstractmethod
    def calc_anomaly_score(self, data: pd.DataFrame) -> pd.DataFrame:
        pass
