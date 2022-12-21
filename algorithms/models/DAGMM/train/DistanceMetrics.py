import tensorflow as tf


class DistanceMetrics(tf.keras.layers.Layer):
    @tf.function
    def call(self, inputs, *args, **kwargs):
        prediction, label = inputs

        euclidean = tf.math.squared_difference(prediction, label)
        euclidean = tf.reduce_sum(euclidean, axis=[1, 2])

        f_mse = tf.keras.losses.MeanSquaredError(reduction=tf.keras.losses.Reduction.NONE)
        mse = tf.math.reduce_mean(f_mse(prediction, label), axis=1)

        f_cs = tf.keras.losses.CosineSimilarity(axis=1, reduction=tf.keras.losses.Reduction.NONE)
        cs = tf.math.reduce_mean(f_cs(prediction, label), axis=1)

        return tf.stack([euclidean, mse, cs], axis=1)
