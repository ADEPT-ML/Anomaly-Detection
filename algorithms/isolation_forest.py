import pandas as pd
import sklearn as sk
from sklearn.ensemble import IsolationForest
from sklearn.exceptions import NotFittedError

from .util.deseason import deseasoning
from .util.shallow_post_processing import shallow_model_post_processing


class IsolationForestModel:
    def __init__(self, n_trees: int = 500, thread_amount: int = None, cont: float = 0.01) -> None:
        self.model = IsolationForest(n_estimators=n_trees, n_jobs=thread_amount, contamination=cont)

    def calc_anomaly_score(self, data: pd.DataFrame) -> pd.DataFrame:
        deseasoned = data.apply(lambda x: deseasoning(x), axis=0)
        try:
            sk.utils.validation.check_is_fitted(self.model)
        except NotFittedError:
            self.train_model(data)
        scores = self.model.decision_function(deseasoned.dropna())
        idx = data.iloc[336:-336].index
        return shallow_model_post_processing(pd.DataFrame(data=scores, index=idx))

    def train_model(self, data: pd.DataFrame):
        deseasoned = data.apply(lambda x: deseasoning(x), axis=0)
        self.model.fit(deseasoned.dropna())
