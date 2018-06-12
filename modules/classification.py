import tensorflow as tf
import tensorflow.contrib.slim as slim
import module
import math
from tensorflow import flags

FLAGS = flags.FLAGS


class NetFV(module.BaseModule):
    def __init__(self, feature_size, max_frames, cluster_size, add_batch_norm, is_training):
        self.feature_size = feature_size
        self.max_frames = max_frames
        self.is_training = is_training
        self.add_batch_norm = add_batch_norm
        self.cluster_size = cluster_size

    def forward(self, reshaped_input):
        cluster_weights = tf.get_variable("cluster_weights",
                                          [self.feature_size, self.cluster_size],
                                          initializer=tf.random_normal_initializer(
                                              stddev=1 / math.sqrt(self.feature_size)))

        covar_weights = tf.get_variable("covar_weights",
                                        [self.feature_size, self.cluster_size],
                                        initializer=tf.random_normal_initializer(mean=1.0, stddev=1 / math.sqrt(
                                            self.feature_size)))

        covar_weights = tf.square(covar_weights)
        eps = tf.constant([1e-6])
        covar_weights = tf.add(covar_weights, eps)

        tf.summary.histogram("cluster_weights", cluster_weights)
        activation = tf.matmul(reshaped_input, cluster_weights)
        if self.add_batch_norm:
            activation = slim.batch_norm(
                activation,
                center=True,
                scale=True,
                is_training=self.is_training,
                scope="cluster_bn")
        else:
            cluster_biases = tf.get_variable("cluster_biases",
                                             [self.cluster_size],
                                             initializer=tf.random_normal(stddev=1 / math.sqrt(self.feature_size)))
            tf.summary.histogram("cluster_biases", cluster_biases)
            activation += cluster_biases

        activation = tf.nn.softmax(activation)
        tf.summary.histogram("cluster_output", activation)

        activation = tf.reshape(activation, [-1, self.max_frames, self.cluster_size])

        a_sum = tf.reduce_sum(activation, -2, keep_dims=True)

        if not FLAGS.fv_couple_weights:
            cluster_weights2 = tf.get_variable("cluster_weights2",
                                               [1, self.feature_size, self.cluster_size],
                                               initializer=tf.random_normal_initializer(
                                                   stddev=1 / math.sqrt(self.feature_size)))
        else:
            cluster_weights2 = tf.scalar_mul(FLAGS.fv_coupling_factor, cluster_weights)

        a = tf.multiply(a_sum, cluster_weights2)

        activation = tf.transpose(activation, perm=[0, 2, 1])

        reshaped_input = tf.reshape(reshaped_input, [-1, self.max_frames, self.feature_size])
        fv1 = tf.matmul(activation, reshaped_input)

        fv1 = tf.transpose(fv1, perm=[0, 2, 1])

        # computing second order FV
        a2 = tf.multiply(a_sum, tf.square(cluster_weights2))

        b2 = tf.multiply(fv1, cluster_weights2)
        fv2 = tf.matmul(activation, tf.square(reshaped_input))

        fv2 = tf.transpose(fv2, perm=[0, 2, 1])
        fv2 = tf.add_n([a2, fv2, tf.scalar_mul(-2, b2)])

        fv2 = tf.divide(fv2, tf.square(covar_weights))
        fv2 = tf.subtract(fv2, a_sum)

        fv2 = tf.reshape(fv2, [-1, self.cluster_size * self.feature_size])

        fv2 = tf.nn.l2_normalize(fv2, 1)
        fv2 = tf.reshape(fv2, [-1, self.cluster_size * self.feature_size])
        fv2 = tf.nn.l2_normalize(fv2, 1)

        fv1 = tf.subtract(fv1, a)
        fv1 = tf.divide(fv1, covar_weights)

        fv1 = tf.nn.l2_normalize(fv1, 1)
        fv1 = tf.reshape(fv1, [-1, self.cluster_size * self.feature_size])
        fv1 = tf.nn.l2_normalize(fv1, 1)

        return tf.concat([fv1, fv2], 1)