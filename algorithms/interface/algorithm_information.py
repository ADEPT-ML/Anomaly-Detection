from dataclasses import dataclass

__all__ = ["AlgorithmInformation"]


@dataclass
class AlgorithmInformation():
    name: str
    deep: bool
    explainable: bool
