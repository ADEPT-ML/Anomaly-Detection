import tensorflow as tf
from keras import layers

from OutputFedRNN import OutputFedRNN


def simple_cnn_ae(input_layer, feature_count):
    x = tf.keras.layers.Conv1D(32, 3, strides=2, padding="same", activation="relu")(input_layer)
    x = tf.keras.layers.Conv1D(16, 3, strides=2, padding="same", activation="relu")(x)
    x = tf.keras.layers.Conv1D(8, 3, strides=2, padding="same", activation="relu")(x)
    encoder_state = tf.keras.layers.Conv1D(4, 3, strides=2, padding="same", activation="relu")(x)

    x = tf.keras.layers.Conv1DTranspose(4, 3, strides=2, padding="same", activation="relu")(encoder_state)
    x = tf.keras.layers.Conv1DTranspose(8, 3, strides=2, padding="same", activation="relu")(x)
    x = tf.keras.layers.Conv1DTranspose(16, 3, strides=2, padding="same", activation="relu")(x)
    x = tf.keras.layers.Conv1DTranspose(32, 3, strides=2, padding="same", activation="relu")(x)
    prediction = tf.keras.layers.Conv1D(feature_count, 3, padding="same")(x)

    encoder_state = tf.keras.layers.Flatten()(encoder_state)

    encoder_state = tf.ensure_shape(encoder_state, [None, 24])

    return encoder_state, prediction


def leiden_cnn_ae(input_layer, feature_count, kernel_size=3):
    conv1 = layers.Conv1D(64, kernel_size, dilation_rate=1, padding="same", activation="relu")(input_layer)
    conv1 = layers.Conv1D(16, 1)(conv1)

    conv2 = layers.Conv1D(64, kernel_size, dilation_rate=2, padding="same", activation="relu")(conv1)
    conv2 = layers.Conv1D(16, 1)(conv2)

    conv3 = layers.Conv1D(64, kernel_size, dilation_rate=4, padding="same", activation="relu")(conv2)
    conv3 = layers.Conv1D(16, 1)(conv3)

    conv4 = layers.Conv1D(64, kernel_size, dilation_rate=8, padding="same", activation="relu")(conv3)
    conv4 = layers.Conv1D(16, 1)(conv4)

    conv5 = layers.Conv1D(64, kernel_size, dilation_rate=16, padding="same", activation="relu")(conv4)
    conv5 = layers.Conv1D(16, 1)(conv5)

    conv6 = layers.Conv1D(64, kernel_size, dilation_rate=32, padding="same", activation="relu")(conv5)
    conv6 = layers.Conv1D(16, 1)(conv6)

    concat = layers.Concatenate(axis=-1)([conv1, conv2, conv3, conv4, conv5, conv6])

    hidden_vector = layers.Conv1D(4, 1)(concat)
    hidden_vector = layers.AveragePooling1D(pool_size=32, padding="same")(hidden_vector)

    upsample = layers.UpSampling1D(32)(hidden_vector)

    de_conv1 = layers.Conv1D(64, kernel_size, dilation_rate=32, padding="same", activation="relu")(upsample)
    de_conv1 = layers.Conv1D(16, 1)(de_conv1)

    de_conv2 = layers.Conv1D(64, kernel_size, dilation_rate=16, padding="same", activation="relu")(de_conv1)
    de_conv2 = layers.Conv1D(16, 1)(de_conv2)

    de_conv3 = layers.Conv1D(64, kernel_size, dilation_rate=8, padding="same", activation="relu")(de_conv2)
    de_conv3 = layers.Conv1D(16, 1)(de_conv3)

    de_conv4 = layers.Conv1D(64, kernel_size, dilation_rate=4, padding="same", activation="relu")(de_conv3)
    de_conv4 = layers.Conv1D(16, 1)(de_conv4)

    de_conv5 = layers.Conv1D(64, kernel_size, dilation_rate=2, padding="same", activation="relu")(de_conv4)
    de_conv5 = layers.Conv1D(16, 1)(de_conv5)

    de_conv6 = layers.Conv1D(64, kernel_size, dilation_rate=1, padding="same", activation="relu")(de_conv5)
    de_conv6 = layers.Conv1D(16, 1)(de_conv6)

    de_concat = layers.Concatenate(axis=-1)([de_conv1, de_conv2, de_conv3, de_conv4, de_conv5, de_conv6])

    prediction = layers.Conv1D(feature_count, 1)(de_concat)
    encoder_state = layers.Flatten()(hidden_vector)
    encoder_state = tf.ensure_shape(encoder_state, [None, 12])

    return encoder_state, prediction


def simplified_leiden_cnn_ae(input_layer, feature_count, kernel_size=3):
    filter_count = 32
    pool_size = 16
    latent_filter_count = 2
    encoder_state_size = (96 // pool_size) * latent_filter_count

    conv1 = layers.Conv1D(filter_count, kernel_size, dilation_rate=1, padding="same", activation="relu")(input_layer)
    conv1 = layers.Conv1D(filter_count // 4, 1)(conv1)

    conv2 = layers.Conv1D(filter_count, kernel_size, dilation_rate=2, padding="same", activation="relu")(conv1)
    conv2 = layers.Conv1D(filter_count // 4, 1)(conv2)

    conv3 = layers.Conv1D(filter_count, kernel_size, dilation_rate=4, padding="same", activation="relu")(conv2)
    conv3 = layers.Conv1D(filter_count // 4, 1)(conv3)

    concat = layers.Concatenate(axis=-1)([conv1, conv2, conv3])

    hidden_vector = layers.Conv1D(latent_filter_count, 1)(concat)
    hidden_vector = layers.AveragePooling1D(pool_size=pool_size, padding="same")(hidden_vector)

    upsample = layers.UpSampling1D(pool_size)(hidden_vector)

    de_conv4 = layers.Conv1D(filter_count, kernel_size, dilation_rate=4, padding="same", activation="relu")(upsample)
    de_conv4 = layers.Conv1D(filter_count // 4, 1)(de_conv4)

    de_conv5 = layers.Conv1D(filter_count, kernel_size, dilation_rate=2, padding="same", activation="relu")(de_conv4)
    de_conv5 = layers.Conv1D(filter_count // 4, 1)(de_conv5)

    de_conv6 = layers.Conv1D(filter_count, kernel_size, dilation_rate=1, padding="same", activation="relu")(de_conv5)
    de_conv6 = layers.Conv1D(filter_count // 4, 1)(de_conv6)

    de_concat = layers.Concatenate(axis=-1)([de_conv4, de_conv5, de_conv6])

    prediction = layers.Conv1D(feature_count, 1)(de_concat)
    encoder_state = layers.Flatten()(hidden_vector)
    encoder_state = tf.ensure_shape(encoder_state, [None, encoder_state_size])

    return encoder_state, prediction


def gru_ae(input_layer, recurrent_size, seq_length, feature_count):
    encoded, encoder_state = tf.keras.layers.GRU(recurrent_size, return_state=True)(input_layer)
    decoded = tf.keras.layers.RepeatVector(seq_length)(encoded)
    decoded = tf.keras.layers.GRU(recurrent_size, return_sequences=True)(decoded, initial_state=encoder_state)
    prediction = tf.keras.layers.Dense(feature_count)(decoded)
    return encoder_state, prediction


def output_fed_gru_ae(input_layer, recurrent_size, seq_length, feature_count):
    _, encoder_state = tf.keras.layers.GRU(recurrent_size, return_state=True)(input_layer)
    decoder = OutputFedRNN(tf.keras.layers.GRUCell(recurrent_size), feature_count, seq_length)
    prediction = decoder(encoder_state)
    return encoder_state, prediction


def multilayer_lstm_ae(input_layer, recurrent_size, seq_length, feature_count, layer_count=2, dropout=0):
    x = tf.keras.layers.LSTM(recurrent_size, return_sequences=(layer_count > 1), return_state=(layer_count == 1),
                             dropout=dropout)(input_layer)
    if layer_count > 1:
        for _ in range(layer_count - 2):
            x = tf.keras.layers.LSTM(recurrent_size, return_sequences=True, dropout=dropout)(x)
        x = tf.keras.layers.LSTM(recurrent_size, return_state=True)(x)

    encoded, encoder_memory, encoder_carry = x
    encoder_state = tf.keras.layers.Concatenate()([encoder_memory, encoder_carry])

    repeated_encoder_output = tf.keras.layers.RepeatVector(seq_length)(encoded)

    decoded = tf.keras.layers.LSTM(recurrent_size,
                                   return_sequences=True,
                                   dropout=dropout)(repeated_encoder_output,
                                                    initial_state=[encoder_memory, encoder_carry])
    if layer_count > 1:
        for _ in range(layer_count - 2):
            decoded = tf.keras.layers.LSTM(recurrent_size, return_sequences=True, dropout=dropout)(decoded)
        decoded = tf.keras.layers.LSTM(recurrent_size, return_sequences=True)(decoded)
    prediction = tf.keras.layers.Dense(feature_count)(decoded)
    return encoder_state, prediction


def output_fed_lstm_ae(input_layer, recurrent_size, seq_length, feature_count):
    _, encoder_memory, encoder_carry = tf.keras.layers.LSTM(recurrent_size, return_state=True)(input_layer)
    decoder = OutputFedRNN(tf.keras.layers.LSTMCell(recurrent_size), feature_count, seq_length)
    prediction = decoder([encoder_memory, encoder_carry])
    encoder_state = tf.keras.layers.Concatenate()([encoder_memory, encoder_carry])
    return encoder_state, prediction
