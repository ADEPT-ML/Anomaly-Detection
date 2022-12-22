"""Contains all functions for used by multiple shallow models."""
import pandas as pd
from statsmodels.tsa.seasonal import seasonal_decompose


def remove_seasonality(data: pd.DataFrame, model_type: str = "additive", period=672) -> pd.DataFrame:
    """Removes the seasonality from the data.

    Uses the given model type and period if specified.

    Args:
        data: The timeseries data as a Pandas DataFrame.
        model_type: Either "additive" or "multiplicative".
        period: Amount of values that belong to one period.

    Returns:
        The given data with removed seasonality.
    """
    if data.dtypes == float:
        return seasonal_decompose(x=data, model=model_type, period=period).resid
    else:
        return data


def post_processing(anomaly_score: list, indices: list) -> tuple[list, list, list, float]:
    """Calculates the required output data for shallow models.

    Uses the resulting data of the shallow models to create the timestamps,
    threshold and anomaly score in the required format.
    As shallow models do not produce the required data for deep error values an empty list is returned instead.

    Args:
        anomaly_score: The anomaly score output of the shallow model
        indices: An updated list of the timestamp values

    Returns:
        An empty deep anomaly score, the calculated anomaly scores, the timestamps and the threshold.
    """
    output_timestamps = [str(e) for e in indices]
    output_threshold = max(anomaly_score)
    output_error = [-e + output_threshold for e in anomaly_score]
    return [], output_error, output_timestamps, output_threshold
