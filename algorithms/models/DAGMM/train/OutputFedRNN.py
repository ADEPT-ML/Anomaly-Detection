import tensorflow as tf
from tensorflow import keras


class OutputFedRNN(keras.layers.Layer):
    def __init__(self, cell, feature_count, seq_length, **kwargs):
        super().__init__(**kwargs)
        self.seq_length = seq_length
        self.feature_count = feature_count
        self.dense = keras.layers.Dense(feature_count)
        self.cell = cell

    @tf.function
    def call(self, inputs, *args, **kwargs):
        batch_size = tf.shape(inputs)[-2]
        state = inputs
        outputs = tf.TensorArray(tf.float32, self.seq_length)
        last_output = tf.zeros((batch_size, self.feature_count))

        for i in tf.range(self.seq_length - 1, -1, -1):
            decoder_output, state = self.cell(last_output, state)
            last_output = self.dense(decoder_output)
            outputs = outputs.write(i, last_output)
        return tf.transpose(outputs.stack(), [1, 0, 2])
