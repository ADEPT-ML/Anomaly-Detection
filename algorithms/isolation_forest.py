from tokenize import String
import sklearn as sk
from sklearn.exceptions import NotFittedError
import pandas as pd
import numpy as np
from sklearn.ensemble import IsolationForest
from algorithms.deseason import deseasoning


class IsolationForestModel:
    def __init__(self, n_trees: int = 500 , nCores: int = None, cont: String = 0.01) -> None:
        self.model = IsolationForest(n_estimators =  n_trees, n_jobs = nCores, contamination = cont)

    def calc_anomaly_score(self, data: pd.DataFrame) -> pd.DataFrame:
        deseasoned = data.apply(lambda x : deseasoning(x), axis = 0)
        try:
            sk.utils.validation.check_is_fitted(self.model)
        except NotFittedError: 
            self.train_model(data)
        scores = self.model.decision_function(deseasoned.dropna())
        idx = data.iloc[336:-336].index
        return pd.DataFrame(data = scores, index = idx)
    
    def train_model(self, data: pd.DataFrame):
        deseasoned = data.apply(lambda x : deseasoning(x), axis = 0)
        self.model.fit(deseasoned.dropna())
    

