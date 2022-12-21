"""An implementation of the DAGMM deep model."""
import os
from subprocess import Popen, DEVNULL, STDOUT
from typing import NoReturn

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1'  # noqa

import numpy as np
import pandas as pd
import tensorflow as tf

from fastapi import HTTPException

from .interface.algorithm_config import *
from .interface.algorithm_information import *
from .interface.algorithm_interface import *
from .util.window_util import sliding_window
from .util.utils import create_unique_representation


class Algorithm(AlgorithmInterface):
    """Algorithm class that contains the DAGMM implementation."""

    def __init__(self) -> None:
        self.path = os.path.join("algorithms", "models", "DAGMM")
        self.training_file = os.path.join("algorithms", "models", "training")
        self.window_size = 96
        self.score_limit = 100
        self.batch_size = 1024
        self.training_message = "No model exists for the selected data. Training has started. Check back later."
        self.busy_message = "No model exists for the selected data and training has not been started because a model " \
                            "is already being trained. Please try again later. "
        self.generic_model = tf.keras.models.load_model(os.path.join(self.path, "universal"), compile=False)
        self.info = AlgorithmInformation(name="DAGMM", deep=True, explainable=False)
        percentile_setting = SliderSetting(id="percentile",
                                           name="Percentile",
                                           description="The threshold percentile for the anomaly detection.",
                                           default=99.5,
                                           step=0.1,
                                           lowBound=95,
                                           highBound=100)
        constant_setting = SliderSetting(id="constant",
                                         name="Constant",
                                         description="The threshold constant for the anomaly detection.",
                                         default=0,
                                         step=0.5,
                                         lowBound=-64,
                                         highBound=64)
        stride_setting = SliderSetting(id="stride",
                                       name="Stride",
                                       description="The window stride.",
                                       default=1,
                                       step=1,
                                       lowBound=1,
                                       highBound=96)
        slider_option = Option(name="Percentile", settings=[percentile_setting])
        float_option = Option(name="Constant", settings=[constant_setting])
        option_setting = OptionSetting(id="dropdown",
                                       name="Threshold",
                                       description="Select the threshold mode.",
                                       default="Percentile",
                                       options=[slider_option, float_option])
        single_mode = ToggleSetting(id="generic", name="Generic Model",
                                    description="Uses a generic model for the prediction.",
                                    default=False)
        self.config = AlgorithmConfig([option_setting, stride_setting, single_mode])

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
        """Calculates an anomaly score for the given data.

        Loads the model and parameters based on the specified configuration and sensor selection.
        Attempts to start the training if no model is present.

        Args:
            data: A dataframe containing a sensor data slice.
            building: The name of the building from which the data originates
            config: The user specified configuration data.

        Returns:
            The deep anomaly score if present, the calculated anomaly scores, the timestamps and the threshold.

        Raises:
            HTTPException: 202 when the training has started. 503 when another model is already training.
        """
        generic, stride = self.parse_config(config)
        if generic:
            return self.generic_dagmm(data, config, stride)

        model_hash = create_unique_representation(building, data.columns.tolist())
        model_path = os.path.join(self.path, str(model_hash))

        if not self.is_trained(model_path):
            self.train_model(model_path, data)

        model = tf.keras.models.load_model(model_path, compile=False)
        indices = data.index.tolist()
        data = data.to_numpy()
        sliding_window_list = sliding_window(data, self.window_size, stride)
        windows = tf.constant(sliding_window_list)
        output_data = model.predict(windows, batch_size=self.batch_size, verbose=0)
        errors = self.calculate_scores(output_data, stride)
        threshold = self.parse_threshold(config, errors)
        return [], errors, indices, threshold

    def train_model(self, model_path: str, data: pd.DataFrame) -> NoReturn:
        """Attempts to start the training process.

        Creates the model directory if necessary.
        Writes the training data to a file and starts the training process as a separate process.

        Args:
            model_path: The path to the model directory.
            data: The training data.

        Raises:
            HTTPException: 202 when the training has started. 503 when another model is already training.
        """
        if os.path.exists(self.training_file):
            raise HTTPException(status_code=503, detail=self.busy_message)
        if not (os.path.isdir(model_path)):
            os.mkdir(model_path)
        data.to_csv(os.path.join(model_path, "data.csv"), index=False)
        train_script = os.path.join(self.path, 'train', 'train.py')
        Popen(f"python3 {train_script} {model_path}", stdout=DEVNULL, stderr=STDOUT, shell=True)
        with open(self.training_file, "w"):
            pass
        raise HTTPException(status_code=202, detail=self.training_message)

    def generic_dagmm(self, data: pd.DataFrame, config: dict, stride: int) -> tuple[list, list, list, float]:
        """Calculates the anomaly score using the generic DAGMM model.

        Args:
            data: A dataframe containing a sensor data slice.
            config: The user specified configuration data.
            stride: The amount by which the window is moved each time.

        Returns:
            The deep anomaly score, the calculated anomaly scores, the timestamps and the threshold.
        """
        deep_errors = []
        for key in data.keys():
            errors = self.predict_generic(data[[key]].reset_index(drop=True), stride)
            deep_errors.append(errors)
        output_error = [sum(j[i] for j in deep_errors) for i in range(len(deep_errors[0]))]
        output_timestamps = [str(e) for e in data.index][:len(output_error)]
        threshold = self.parse_threshold(config, output_error)
        return deep_errors, output_error, output_timestamps, threshold

    def predict_generic(self, data, stride: int) -> list[float]:
        """Creates a prediction using the generic model.

        Args:
            data: A dataframe containing a sensor data slice.
            stride: The amount by which the window is moved each time.

        Returns:
            The anomaly scores for the specified data.
        """
        data = (data - data.mean(axis=0)) / data.std(axis=0)
        data = data.to_numpy()
        sliding_window_list = sliding_window(data, self.window_size, stride)
        windows = tf.constant(sliding_window_list)
        output_data = self.generic_model.predict(windows, batch_size=self.batch_size, verbose=0)
        errors = self.calculate_scores(output_data, stride)
        return errors

    def calculate_scores(self, results: list, stride: int) -> list[float]:
        """Calculates the anomaly scores based on the model output and parameters.

        Considers the window size and stride value to compute all individual anomaly scores.

        Args:
            results: The results of the DAGMM model.
            stride: The amount by which the window is moved each time.

        Returns:
            The resulting anomaly scores.
        """
        results = np.clip(results, -self.score_limit, self.score_limit)
        errors = []
        amount = self.window_size - stride + (len(results) * stride)
        for i in range(amount):
            elements = results[max(0, ((i - self.window_size) // stride) + 1):min(len(results), (i // stride) + 1)]
            errors.append(float(sum(elements) / len(elements)))
        return errors

    @staticmethod
    def parse_config(config: dict) -> tuple[bool, int]:
        """Extracts the generic toggle and stride value from the config.

        Args:
            config: The specified config for the anomaly prediction.

        Returns:
            The generic and stride settings.
        """
        return config["generic"], config["stride"]

    @staticmethod
    def parse_threshold(config: dict, scores: list) -> float:
        """Parses the threshold value from the config.

        Uses either a constant value or a percentile based on the supplied data.

        Args:
            config: The specified config for the anomaly prediction.
            scores: The resulting anomaly scores.

        Returns:
            The selected threshold float value.
        """
        match config["dropdown"]:
            case "Percentile":
                return np.percentile(scores, config["percentile"])
            case "Constant":
                return config["constant"]

    @staticmethod
    def is_trained(model_path: str) -> bool:
        """Verifies if the model is trained.

        Training is deemed complete if a file named completed exists in the model directory.

        Args:
            model_path: The path to the model directory.

        Returns:
            True if the model is trained, false otherwise.
        """
        return os.path.isfile(os.path.join(model_path, "completed"))
