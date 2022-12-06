import pandas as pd
from statsmodels.tsa.seasonal import seasonal_decompose


def deseasoning(data: pd.DataFrame, model_type: str = "additive", period=672) -> pd.DataFrame:
    if data.dtypes == float:
        decomp = seasonal_decompose(data, model=model_type, period=period)
    else:
        return data
    return decomp.resid
