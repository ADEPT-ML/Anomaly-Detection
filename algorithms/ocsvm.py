import pandas as pd
import sklearn as sk
from sklearn.exceptions import NotFittedError
from sklearn.preprocessing import StandardScaler
from sklearn.svm import OneClassSVM

from .interface.algorithm_config import *
from .interface.algorithm_interface import *
from .interface.algorithm_information import *
from .util.deseason import deseasoning
from .util.shallow_post_processing import shallow_model_post_processing


class Algorithm(AlgorithmInterface):

    def __init__(self, kernel_fun: str = "rbf", gamma_val: float = 0.01, nu_val: float = 0.01) -> None:
        self.model = OneClassSVM(kernel=kernel_fun, gamma=gamma_val, nu=nu_val)
        self.scaler = StandardScaler()
        self.info = AlgorithmInformation(name="One-Class SVM", deep=False, explainable=False)
        self.config = AlgorithmConfig()

    @property
    def information(self) -> AlgorithmInformation:
        return self.info

    @property
    def configuration(self) -> AlgorithmConfig:
        return self.config

    def calc_anomaly_score(self, data: pd.DataFrame, config: dict) -> tuple[list, list, list, float]:
        deseasoned = data.apply(lambda x: deseasoning(x), axis=0)
        try:
            sk.utils.validation.check_is_fitted(self.model)
        except NotFittedError:
            self.train_model(data)
        scaled_data = self.scaler.fit_transform(deseasoned.dropna())
        scores = self.model.decision_function(scaled_data)
        idx = data.iloc[336:-336].index
        error, timestamps, threshold = shallow_model_post_processing(pd.DataFrame(data=scores, index=idx))
        return [], error, timestamps, threshold

    def train_model(self, data: pd.DataFrame):
        deseasoned = data.apply(lambda x: deseasoning(x), axis=0)
        scaled_data = self.scaler.fit_transform(deseasoned.dropna())
        self.model.fit(scaled_data)
