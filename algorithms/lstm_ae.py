import os

import numpy as np
import pandas as pd
import tensorflow as tf

from .interface.algorithm_config import *
from .interface.algorithm_interface import *
from .interface.algorithm_information import *
from .util import utils


class Algorithm(AlgorithmInterface):

    def __init__(self):
        self.path = os.path.join('algorithms', 'models', 'LSTM')
        self.model = tf.keras.models.load_model(self.path, compile=False)
        self.info = AlgorithmInformation(name="LSTM Autoencoder", deep=True, explainable=True)
        self.config = Config()

    @property
    def information(self) -> AlgorithmInformation:
        return self.info

    @property
    def configuration(self) -> Config:
        return self.config

    def calc_anomaly_score(self, data: pd.DataFrame) -> tuple[list, list, list, float]:
        deep_errors = []
        for col in data.keys():
            errors = self.predict(data[[col]].reset_index(drop=True))
            deep_errors.append(utils.replace_nan(utils.unpack_array(errors)))
        output_error = [sum(j[i] for j in deep_errors) for i in range(len(deep_errors[0]))]
        output_threshold = np.percentile(output_error, 99.5)
        output_timestamps = [str(e) for e in data.index][:len(output_error)]
        return deep_errors, output_error, output_timestamps, output_threshold

    def predict(self, data: pd.DataFrame):
        data = (data - data.mean(axis=0)) / data.std(axis=0)  # Normalizing data
        data = data.to_numpy()

        window_size = 50
        slide_factor = 50
        sliding_window_list = self.sliding_window(data, window_size, slide_factor)
        windows = tf.constant(sliding_window_list)
        output_data = self.model.predict(windows)
        differences = [self.window_diff_numpy(sliding_window_list[i], output_data[i]) for i in range(len(windows))]
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

    def sliding_window(self, data: list, window_size: int, slide_factor: int) -> list:
        return [data[i:i + window_size] for i in range(0, len(data) - window_size + 1, slide_factor)]

    def window_diff_numpy(self, a: np.array, b: np.array) -> np.array:
        return np.abs(a - b)
