"""Contains all function for the dynamic algorithm loading."""
import sys


def fetch_algorithms() -> list:
    """Creates a list with all anomaly detection algorithms.

    Includes only classes that adhere to the specifications.
    Every anomaly detection algorithms needs to inherit from the AlgorithmInterface.
    Anomaly detection algorithm classes that contain compile time errors are also excluded.

    Returns:
        A list with all anomaly detection algorithms.
    """
    fetched_algorithms = []
    for m in sys.modules:
        if "algorithms." in m:
            try:
                algo = sys.modules[m].Algorithm()
                fetched_algorithms.append(algo)
            except Exception:
                pass
    return fetched_algorithms


def sort_algorithms(algorithms) -> list:
    """Sorts the algorithms alphabetically and groups them by their deep value.

    Args:
        algorithms: A list of algorithm instances.

    Returns:
        A sorted list of algorithm instances.
    """
    return sorted(algorithms, key=lambda x: (x.information.deep, x.information.name))


def create_algorithm_json(algorithms: list) -> list[dict]:
    """Creates the json representation of the specified list of anomaly detection algorithms.

    Args:
        algorithms: A list of algorithm instances.

    Returns:
        The json representation of the algorithms.
    """
    return [algorithm_format(i, e.information, e.configuration) for i, e in enumerate(algorithms)]


def algorithm_format(algo_id: int, info, config) -> dict:
    """Creates a dictionary based on the specified anomaly detection algorithm information.

    Args:
        algo_id: The id of the algorithm.
        info: The information property of the algorithm instance.
        config: The configuration property of the algorithm instance.

    Returns:
        A dictionary that represents the anomaly detection algorithm.
    """
    return {"id": algo_id, "name": info.name, "explainable": info.explainable, "config": config}
