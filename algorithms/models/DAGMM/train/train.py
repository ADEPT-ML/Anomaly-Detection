import os
import sys

import pandas as pd

import Autoencoders
import DAGMM

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1'
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"  # Disable GPU as training is usually faster on CPU for used models
import tensorflow as tf

# ======================================================================================================================
#    _____ ____  _   _ ______ _____ _____
#   / ____/ __ \| \ | |  ____|_   _/ ____|
#  | |   | |  | |  \| | |__    | || |  __
#  | |   | |  | | . ` |  __|   | || | |_ |
#  | |___| |__| | |\  | |     _| || |__| |
#   \_____\____/|_| \_|_|    |_____\_____|
#
# ======================================================================================================================


initial_learning_rate = 0.003
use_lr_decay = True
decay_factor = 0.95
decay_after_epochs = 1

early_stopping_metric = "loss"
early_stopping_patience = 10
early_stopping_delta = 0.1

batch_size = 256
sequence_length = 96
max_epochs = 500

gmm_component_count = 5
regularization_strength = 1e-7
energy_importance = 0.1


# ======================================================================================================================
#  _______ _____            _____ _   _   _____          _____ __  __ __  __
# |__   __|  __ \     /\   |_   _| \ | | |  __ \   /\   / ____|  \/  |  \/  |
#    | |  | |__) |   /  \    | | |  \| | | |  | | /  \ | |  __| \  / | \  / |
#    | |  |  _  /   / /\ \   | | | . ` | | |  | |/ /\ \| | |_ | |\/| | |\/| |
#    | |  | | \ \  / ____ \ _| |_| |\  | | |__| / ____ \ |__| | |  | | |  | |
#    |_|  |_|  \_\/_/    \_\_____|_| \_| |_____/_/    \_\_____|_|  |_|_|  |_|
#
# ======================================================================================================================

def train(path: str, data: pd.DataFrame):
    feature_count = len(data.columns)

    # Prepare training data
    train_data_dataset = create_training_data(data)
    learning_rate = calculate_learning_rate(data)

    dagmm_model, training_model = create_models(feature_count)

    optimizer = tf.keras.optimizers.RMSprop(clipnorm=5, learning_rate=learning_rate)
    training_model.compile(optimizer=optimizer)

    callback_list = [
        tf.keras.callbacks.EarlyStopping(
            monitor=early_stopping_metric,
            patience=early_stopping_patience,
            min_delta=early_stopping_delta,
            verbose=1,
            restore_best_weights=True
        )
    ]

    print("Fitting started!", flush=True)
    training_model.fit(train_data_dataset, epochs=max_epochs, callbacks=callback_list)

    raw_dataset = tf.keras.utils.timeseries_dataset_from_array(data,
                                                               targets=None,
                                                               sequence_length=sequence_length)
    inference_model = DAGMM.InferenceDAGMM(dagmm_model, raw_dataset, train_data_dataset)
    inference_model.predict(train_data_dataset.take(1))  # Build model
    inference_model.save(path)
    with open(os.path.join(path, "completed"), "w"):
        pass
    remove_training_identifier(path)


def create_models(feature_count):
    input_layer = tf.keras.Input(shape=(None, feature_count))
    encoder_state, prediction = Autoencoders.simplified_leiden_cnn_ae(input_layer, feature_count, 5)
    dagmm_model = DAGMM.build_model(input_layer=input_layer,
                                    encoder_state=encoder_state,
                                    prediction=prediction,
                                    gmm_component_count=gmm_component_count)
    training_model = DAGMM.TrainingDAGMM(dagmm_model,
                                         regularization_loss_multiplier=regularization_strength,
                                         energy_loss_multiplier=energy_importance)
    return dagmm_model, training_model


def calculate_learning_rate(data):
    if use_lr_decay:
        steps_per_epoch = ((len(data) - sequence_length) // batch_size) + 1
        decay_after_steps = decay_after_epochs * steps_per_epoch
        learning_rate = tf.keras.optimizers.schedules.ExponentialDecay(initial_learning_rate,
                                                                       decay_steps=decay_after_steps,
                                                                       decay_rate=decay_factor)
    else:
        learning_rate = initial_learning_rate
    return learning_rate


def create_training_data(data):
    train_data_norm = (data - data.mean()) / data.std()
    train_data_dataset = tf.keras.utils.timeseries_dataset_from_array(train_data_norm,
                                                                      targets=None,
                                                                      sequence_length=sequence_length,
                                                                      batch_size=batch_size)
    train_data_dataset = train_data_dataset.shuffle(buffer_size=len(data))
    return train_data_dataset


def parse_parameters(args):
    if len(args) != 1:
        print(args, flush=True)
        return
    try:
        df = pd.read_csv(os.path.join(args[0], "data.csv"))
        train(args[0], df)
    except Exception:
        remove_training_identifier(args[0])

def remove_training_identifier(path):
    training_file = os.path.join(path, "..", "..", "training")
    if os.path.isfile(training_file):
        os.remove(training_file)

if __name__ == '__main__':
    parse_parameters(sys.argv[1:])
