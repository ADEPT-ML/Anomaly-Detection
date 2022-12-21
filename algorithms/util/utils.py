"""Contains several general functions useful for all anomaly detection algorithms."""
import hashlib


def replace_nan(data: list) -> list:
    """Replaces all NaN in the list with 0.

    Args:
        data: The list in which all NaN should be replaced.

    Returns:
        The list with all replaced NaN.
    """
    return [e if e == e else 0 for e in data]


def create_unique_representation(building: str, sensors: list[str]) -> str:
    """Creates a unique id for the building and the sensor selection.

    Args:
        building: The name of the selected building.
        sensors: A list of all selected sensor names.

    Returns:
        A string that uniquely identifies the selection.
    """
    combined = f"{building}_{'_'.join(sensors)}"
    return hashlib.md5(combined.encode('utf-8')).hexdigest()
