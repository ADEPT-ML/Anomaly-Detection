import tensorflow as tf
import tensorflow_probability as tfp

import Autoencoders
from DistanceMetrics import DistanceMetrics


class TrainingDAGMM(tf.keras.Model):

    def __init__(self, model, energy_loss_multiplier=0.1, regularization_loss_multiplier=0.005, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.model = model
        self.energy_loss_multiplier = energy_loss_multiplier
        self.regularization_loss_multiplier = regularization_loss_multiplier

    def call(self, inputs, training=True, mask=None):
        prediction_errors, z, class_prediction = self.model(inputs)
        mixture_probabilities, mu, sigma = estimate_gmm_params(z, class_prediction)
        sample_energies = compute_sample_energy(mu, sigma, z, mixture_probabilities)

        euclidean_distances = tf.gather(prediction_errors, [0], axis=1)
        loss_reconstruction_error = tf.reduce_mean(euclidean_distances)
        loss_energy = self.energy_loss_multiplier * tf.reduce_mean(sample_energies)

        # Regularization by keeping diagonal of sigma large
        # sigma_diag = tf.linalg.diag_part(sigma)
        # diag_reciprocal = tf.math.reciprocal(sigma_diag)
        # sigma_regularization_loss = self.regularization_loss_multiplier * tf.reduce_sum(diag_reciprocal)

        # Regularization by keeping average condition number of sigmas small
        sigma_eigvals = tf.linalg.eigvalsh(sigma)
        sigma_eigvals = tf.math.abs(sigma_eigvals)
        min_eigvals = tf.reduce_min(sigma_eigvals, axis=1)
        max_eigvals = tf.reduce_max(sigma_eigvals, axis=1)
        cond_numbers = max_eigvals / min_eigvals
        sigma_regularization_loss = self.regularization_loss_multiplier * tf.reduce_mean(cond_numbers - 1)

        self.add_loss(loss_reconstruction_error)
        self.add_loss(loss_energy)
        self.add_loss(sigma_regularization_loss)

        self.add_metric(loss_reconstruction_error, name="Reconstruction")
        self.add_metric(loss_energy, name="Energy")
        self.add_metric(sigma_regularization_loss, name="Regularization")


class InferenceDAGMM(tf.keras.Model):

    def __init__(self, model, dataset, normal_dataset, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.normalize_layer = tf.keras.layers.Normalization()
        self.normalize_layer.adapt(dataset)
        self.model = model

        prediction_errors, z, class_prediction = model.predict(normal_dataset)
        mixture_probabilities, mu, sigma = estimate_gmm_params(z, class_prediction)

        self.mixture_probabilities = tf.Variable(mixture_probabilities, trainable=False)
        self.mu = tf.Variable(mu, trainable=False)
        self.sigma = tf.Variable(sigma, trainable=False)

    def call(self, inputs, training=False, mask=None):
        inputs = self.normalize_layer(inputs)
        prediction_errors, z, class_prediction = self.model(inputs)
        return compute_sample_energy(self.mu, self.sigma, z, self.mixture_probabilities)


def build_model(input_layer, encoder_state, prediction, gmm_component_count=5):
    prediction_errors = DistanceMetrics()([prediction, input_layer])
    gmm_input = tf.keras.layers.Concatenate()([encoder_state, prediction_errors])

    class_prediction = tf.keras.layers.Dense(128, activation="tanh")(gmm_input)
    class_prediction = tf.keras.layers.BatchNormalization()(class_prediction)

    class_prediction = tf.keras.layers.Dropout(0.5)(class_prediction)

    class_prediction = tf.keras.layers.Dense(32, activation="tanh")(class_prediction)
    class_prediction = tf.keras.layers.BatchNormalization()(class_prediction)

    class_prediction = tf.keras.layers.Dense(gmm_component_count, activation="softmax")(class_prediction)

    model = tf.keras.Model(input_layer, [prediction_errors, gmm_input, class_prediction])

    return model


@tf.function
def estimate_gmm_params(z, softmax_probabilities):
    # Tensor of mixture probabilities with shape [#GMM components, ]
    mixture_probabilities = tf.math.reduce_mean(softmax_probabilities, axis=0)  # Average over batch

    # Tensor of mean values with shape [#GMM components, #features in z]
    mu = tf.einsum("bi,bk->bik", softmax_probabilities, z)  # Compute tensor of sample wise mu vectors
    mu = tf.reduce_mean(mu, axis=0)  # Average over batch
    mu = mu / tf.expand_dims(mixture_probabilities, axis=-1)  # Divide by corresponding mixture probability

    # Tensor of covariance matrices with shape [#GMM components, #features in z, #features in z]
    sigma = tf.map_fn(lambda m: tf.subtract(z, m), elems=mu)  # Compute mu-z-difference vectors
    sigma = tf.einsum("cbi,cbj->cbij", sigma, sigma)  # Outer product for covariance matrix (c: gmm-component, b: batch)
    sigma = tf.einsum("cbik,bc->cbik", sigma, softmax_probabilities)  # Multiply with corresponding softmax value
    sigma = tf.reduce_mean(sigma, axis=1)  # Average over batch

    # Prepare division by mixture probability by creating matrix with that value of appropiate size
    divisor_shape = tf.shape(sigma)[1:]  # Leave out component count (axis 0)
    divisor = tf.map_fn(lambda x: tf.fill(divisor_shape, x), elems=mixture_probabilities)
    sigma = sigma / divisor

    # Regularization of covariance matrix
    sigma = sigma + 0.001 * tf.eye(tf.shape(sigma)[-1])

    return mixture_probabilities, mu, sigma


@tf.function
def compute_sample_energy(mu, sigma, z, mixture_probabilities):
    normal = tfp.distributions.MultivariateNormalTriL(loc=mu, scale_tril=tf.linalg.cholesky(sigma))
    densities = tf.map_fn(lambda x: normal.prob(x), elems=z)
    energy = densities * mixture_probabilities
    energy = tf.reduce_sum(energy, axis=1)
    energy = -tf.math.log(energy)
    return energy
