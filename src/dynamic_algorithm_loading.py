import sys


def fetch_algorithms():
    fetched_algorithms = []
    for m in sys.modules:
        if "algorithms." in m:
            try:
                temp = sys.modules[m].Algorithm()
                fetched_algorithms.append(temp)
            except Exception:
                pass
    return fetched_algorithms


def sort_algorithms(algorithms):
    return sorted(algorithms, key=lambda x: (x.information.deep, x.information.name))


def create_algorithm_json(algorithms):
    return [algorithm_format(i, e.information, e.configuration) for i, e in enumerate(algorithms)]


def algorithm_format(id, info, config):
    return {"id": id, "name": info.name, "explainable": info.explainable, "config": config}
