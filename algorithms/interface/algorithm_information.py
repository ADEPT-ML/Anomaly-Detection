"""Contains the algorithm information dataclass."""
from dataclasses import dataclass

__all__ = ["AlgorithmInformation"]


@dataclass
class AlgorithmInformation:
    """Dataclass that contains all information regarding the anomaly detection algorithm."""
    name: str
    deep: bool
    explainable: bool
