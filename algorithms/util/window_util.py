"""Contains all functions related to sliding windows."""
import numpy as np


def sliding_window(data: np.array, window_size: int, stride: int) -> list:
    """Runs the sliding window algorithm on the specified data.

    Uses the specified window size and stride value.

    Args:
        data: The raw data on which the algorithm is run.
        window_size: The size of the window.
        stride: The amount by which the window is moved each time.

    Returns:

    """
    return [data[i:i + window_size] for i in range(0, len(data) - window_size + 1, stride)]


def window_diff(window_a: np.array, window_b: np.array) -> np.array:
    """Calculates the absolute difference between the individual values of two arrays.

    Args:
        window_a: The first window.
        window_b: The second window.

    Returns:
        An array of the same size with the absolute difference of each value pair.
    """
    return np.abs(window_a - window_b)
