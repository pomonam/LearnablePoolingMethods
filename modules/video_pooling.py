# Copyright 2018 Juhan, Ruijian Inc. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS-IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Modules for pooling frame-level features."""

import tensorflow as tf
import tensorflow.contrib.slim as slim
from tensorflow import flags

import module
import math

FLAGS = flags.FLAGS
flags.DEFINE_integer("lstm_cells", 1024, "Number of LSTM cells.")
flags.DEFINE_integer("lstm_layers", 2, "Number of LSTM layers.")


class BidirectionalLstmModule(module.BaseModule):
    def __init__(self, variable_name, lstm_cells, frame_length, lstm_layers, feature_size, output_size):
        """

        :param feature_size:
        :param output_size:
        """
        self.variable_name = variable_name
        self.lstm_cells = lstm_cells
        self.lstm_layers = lstm_layers
        self.frame_length = frame_length
        self.feature_size = feature_size

    def __call__(self, inputs, state, scope=None):
        """ LSTM layers.
        :param inputs:
        :param state:
        :param scope:
        :return:
        """
        with tf.variable_scope(scope or type(self).__name__):
            forward = tf.contrib.rnn.BasicLSTMCell(self.lstm_cells, forget_bias=1.0)
            backward = tf.contrib.rnn.BasicLSTMCell(self.lstm_cells, forget_bias=1.0)

            forward_stacked = tf.contrib.rnn.MultiRNNCell([forward])
            backward_stacked = tf.contrib.rnn.MultiRNNCell([backward])

            outputs, states = tf.nn.bidirectional_dynamic_rnn(
                forward_stacked,
                backward_stacked,
                inputs,
                sequence_length=self.frame_length,
                dtype=tf.float32
            )
            return states


class NetVlad(module.BaseModule):
    def __init__(self, feature_size, max_frames, cluster_size, add_batch_norm, is_training):
        self.feature_size = feature_size
        self.max_frames = max_frames
        self.is_training = is_training
        self.add_batch_norm = add_batch_norm
        self.cluster_size = cluster_size

    def __call__(self, inputs, state, scope=None):

        cluster_weights = tf.get_variable("cluster_weights",
                                          [self.feature_size, self.cluster_size],
                                          initializer=tf.random_normal_initializer(
                                              stddev=1 / math.sqrt(self.feature_size)))

        tf.summary.histogram("cluster_weights", cluster_weights)
        activation = tf.matmul(inputs, cluster_weights)

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
                                             initializer=tf.random_normal_initializer(
                                                 stddev=1 / math.sqrt(self.feature_size)))
            tf.summary.histogram("cluster_biases", cluster_biases)
            activation += cluster_biases

        activation = tf.nn.softmax(activation)
        tf.summary.histogram("cluster_output", activation)

        activation = tf.reshape(activation, [-1, self.max_frames, self.cluster_size])

        a_sum = tf.reduce_sum(activation, -2, keep_dims=True)

        cluster_weights2 = tf.get_variable("cluster_weights2",
                                           [1, self.feature_size, self.cluster_size],
                                           initializer=tf.random_normal_initializer(
                                               stddev=1 / math.sqrt(self.feature_size)))

        a = tf.multiply(a_sum, cluster_weights2)

        activation = tf.transpose(activation, perm=[0, 2, 1])

        reshaped_input = tf.reshape(inputs, [-1, self.max_frames, self.feature_size])
        vlad = tf.matmul(activation, reshaped_input)
        vlad = tf.transpose(vlad, perm=[0, 2, 1])
        vlad = tf.subtract(vlad, a)

        vlad = tf.nn.l2_normalize(vlad, 1)

        vlad = tf.reshape(vlad, [-1, self.cluster_size * self.feature_size])
        vlad = tf.nn.l2_normalize(vlad, 1)

        return vlad