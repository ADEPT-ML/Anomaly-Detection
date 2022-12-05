import tensorflow as tf
import pandas as pd
import numpy as np
import os

path = os.path.join('algorithms', 'models', 'LSTM')
model = tf.keras.models.load_model(path, compile=False)


def predict(data: pd.DataFrame):
    data = (data - data.mean(axis=0)) / data.std(axis=0)  # Normalizing data
    data = data.to_numpy()

    window_size = 50
    slide_factor = 50
    sliding_window_list = sliding_window(data, window_size, slide_factor)
    windows = tf.constant(sliding_window_list)
    output_data = model.predict(windows)
    differences = [window_diff_numpy(sliding_window_list[i], output_data[i]) for i in range(len(windows))]
    errors = []
    size = window_size // slide_factor
    for i in range(len(data)):
        lower = max(0, i // slide_factor - (size - 1))
        upper = min(len(differences), (i // slide_factor) + 1)
        error_sum = 0
        count = upper - lower
        if count == 0:
            continue
        for j in range(lower, upper):
            offset = i - j * slide_factor
            differences_j_ = differences[j]
            error_sum = error_sum + differences_j_[offset]
        errors.append(error_sum / count)

    return errors


def sliding_window(data: list, window_size: int, slide_factor: int) -> list:
    return [data[i:i + window_size] for i in range(0, len(data) - window_size + 1, slide_factor)]


def window_diff_numpy(a: np.array, b: np.array) -> np.array:
    return np.abs(a - b)
