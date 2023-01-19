"""Contains functions for finding and parsing anomalies."""
from dataclasses import dataclass


@dataclass
class Anomaly:
    """Dataclass that contains all information to describe an anomaly."""
    length: int
    index: int
    error: float


def find_anomalies(anomaly_score: list[float], threshold: float) -> list[Anomaly]:
    """Detects all anomalies based on the anomaly_score and threshold.

    Args:
        anomaly_score: A list of anomaly scores.
        threshold: The specified threshold.

    Returns:
        A list of Anomaly objects.
    """
    anomalies = []
    current_anomaly = False
    for i in range(len(anomaly_score)):
        if anomaly_score[i] > threshold:
            if current_anomaly:
                anomaly = anomalies[-1]
                anomaly.length += 1
                anomaly.error = max(anomaly.error, anomaly_score[i])
            else:
                anomalies.append(Anomaly(length=1, index=i, error=anomaly_score[i]))
                current_anomaly = True
        else:
            current_anomaly = False
    return sorted(anomalies, key=lambda x: x.error, reverse=True)


def parse_anomalies(anomalies: list[Anomaly], timestamps: list[str]) -> list[dict]:
    """Creates a dictionary representation of the anomalies and their timestamps.

    Args:
        anomalies: A list of Anomaly objects.
        timestamps: The timestamps of the data slice.

    Returns:
        The dictionary representation of the anomalies.
    """
    return [
        {"timestamp": timestamps[e.index + e.length // 2], "type": "Point" if e.length == 1 else f"Area - {e.length}"}
        for e in
        anomalies]
