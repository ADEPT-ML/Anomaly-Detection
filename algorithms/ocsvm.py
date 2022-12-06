import pandas as pd
import sklearn as sk
from sklearn.svm import OneClassSVM
from sklearn.exceptions import NotFittedError
from sklearn.preprocessing import StandardScaler

from .util.deseason import deseasoning


class OCSVM():
    def __init__(self, kernel_fun: str = "rbf", gamma_val: float = 0.01, nu_val: float = 0.01) -> None:
        self.model = OneClassSVM(kernel=kernel_fun, gamma=gamma_val, nu=nu_val)
        self.scaler = StandardScaler()

    def calc_anomaly_score(self, data: pd.DataFrame) -> pd.DataFrame:
        deseasoned = data.apply(lambda x: deseasoning(x), axis=0)
        try:
            sk.utils.validation.check_is_fitted(self.model)
        except NotFittedError:
            self.train_model(data)
        scaled_data = self.scaler.fit_transform(deseasoned.dropna())
        scores = self.model.decision_function(scaled_data)
        idx = data.iloc[336:-336].index
        return pd.DataFrame(data=scores, index=idx)

    def train_model(self, data: pd.DataFrame):
        deseasoned = data.apply(lambda x: deseasoning(x), axis=0)
        scaled_data = self.scaler.fit_transform(deseasoned.dropna())
        self.model.fit(scaled_data)
