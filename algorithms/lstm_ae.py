"""An implementation of the LSTM Autoencoder deep model."""
import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1'  # noqa

import numpy as np
import pandas as pd
import tensorflow as tf

from .interface.algorithm_config import *
from .interface.algorithm_information import *
from .interface.algorithm_interface import *
from .util import window_util


class Algorithm(AlgorithmInterface):
    """Algorithm class that contains the LSTM-AE implementation."""

    def __init__(self) -> None:
        self.path = os.path.join('algorithms', 'models', 'LSTM')
        self.model = tf.keras.models.load_model(self.path, compile=False)
        self.info = AlgorithmInformation(name="LSTM Autoencoder", deep=True, explainable=True)
        slider_setting = SliderSetting(id="percentile",
                                       name="Percentile",
                                       description="The threshold percentile for the anomaly detection.",
                                       default=99.5,
                                       step=0.1,
                                       lowBound=95,
                                       highBound=99.9)
        float_setting = FloatSetting(id="constant",
                                     name="Constant",
                                     description="The threshold constant for the anomaly detection.",
                                     default=1)
        slider_option = Option(name="Percentile", settings=[slider_setting])
        float_option = Option(name="Constant", settings=[float_setting])
        option_setting = OptionSetting(id="dropdown",
                                       name="Threshold",
                                       description="Select the threshold mode.",
                                       default="Percentile",
                                       options=[slider_option, float_option])
        self.config = AlgorithmConfig([option_setting])

    @property
    def information(self) -> AlgorithmInformation:
        """Returns an AlgorithmInformation object containing all information regarding the algorithm.

        Returns:
            An AlgorithmInformation object containing all information regarding the algorithm.
        """
        return self.info

    @property
    def configuration(self) -> AlgorithmConfig:
        """Returns an AlgorithmConfig object containing all information regarding the config options for the algorithm.

        Returns:
            An AlgorithmConfig object containing all information regarding the config options.
        """
        return self.config

    def calc_anomaly_score(self, data: pd.DataFrame, building: str, config: dict) -> tuple[list, list, list, float]:
        """Calculates the anomaly score for the given data.

        Args:
            data: A dataframe containing a sensor data slice.
            building: The name of the building from which the data originates
            config: The user specified configuration data.

        Returns:
            The deep anomaly score, the calculated anomaly scores, the timestamps and the threshold.
        """
        deep_errors = []
        for key in data.keys():
            errors = self.predict(data[[key]].reset_index(drop=True))
            deep_errors.append(errors)
        output_error = [sum(j[i] for j in deep_errors) for i in range(len(deep_errors[0]))]
        output_threshold = float(self.parse_threshold(config, output_error))
        output_timestamps = [str(e) for e in data.index][:len(output_error)]
        return deep_errors, output_error, output_timestamps, output_threshold

    def predict(self, data: pd.DataFrame) -> list:
        """Uses the model to calculate the error scores for the supplied data.

        Args:
            data: A dataframe containing a sensor data slice.

        Returns:
            The error scores for the data.
        """
        data = (data - data.mean(axis=0)) / data.std(axis=0)  # Normalizing data
        data = data.to_numpy()

        window_size = slide_factor = 50
        sliding_window_list = window_util.sliding_window(data, window_size, slide_factor)
        windows = tf.constant(sliding_window_list)
        output_data = self.model.predict(windows, batch_size=1024, verbose=0)
        differences = [window_util.window_diff(sliding_window_list[i], output_data[i]) for i in range(len(windows))]
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
            errors.append(float(error_sum / count))

        return errors

    @staticmethod
    def parse_threshold(config: dict, error: list) -> float:
        """Parses the threshold value from the config.

        Uses either a constant value or a percentile based on the supplied data.

        Args:
            config: The specified config for the anomaly prediction.
            error: The calculated anomaly scores.

        Returns:
            The selected threshold float value.
        """
        match config["dropdown"]:
            case "Percentile":
                return np.percentile(error, config["percentile"])
            case "Constant":
                return config["constant"]
