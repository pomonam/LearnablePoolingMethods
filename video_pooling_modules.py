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
import modules
import math
from tensorflow import flags

FLAGS = flags.FLAGS


###############################################################################
# Context Learning methods ####################################################
###############################################################################
class ClPhdModule(modules.BaseModule):
    def __init__(self,
                 feature_size,
                 max_frames,
                 cluster_size,
                 batch_norm,
                 is_training,
                 scope_id=None):
        self.feature_size = feature_size
        self.max_frames = max_frames
        self.cluster_size = cluster_size
        self.batch_norm = batch_norm
        self.is_training = is_training
        self.scope_id = scope_id

    def forward(self, inputs, **unused_params):
        """
        :param inputs: (batch_size * num_samples) x feature_size
        :return: (batch_size * num_samples) x num_clusters
        """
        cluster_weights = tf.get_variable("cl_cluster_weights",
                                          [self.feature_size, self.cluster_size],
                                          initializer=tf.random_normal_initializer(
                                              stddev=1 / math.sqrt(self.feature_size)))
        tf.summary.histogram("cl_cluster_weights", cluster_weights)
        activation = tf.matmul(inputs, cluster_weights)
        if self.batch_norm:
            activation = slim.batch_norm(
                activation,
                center=True,
                scale=True,
                is_training=self.is_training,
                scope="cl_cluster_bn")
        else:
            cluster_biases = tf.get_variable("cl_cluster_biases",
                                             [self.cluster_size],
                                             initializer=tf.random_normal_initializer(
                                                 stddev=1 / math.sqrt(self.feature_size)))
            tf.summary.histogram("cl_cluster_biases", cluster_biases)
            activation += cluster_biases
        # -> (batch_size * num_samples) x num_clusters
        activation = tf.nn.relu6(activation)
        tf.summary.histogram("cl_cluster_output", activation)
        return activation


class ClLrModule(modules.BaseModule):
    def __init__(self,
                 feature_size,
                 max_frames,
                 cluster_size,
                 batch_norm,
                 is_training,
                 scope_id=None,
                 l2_penalty=1e-8):
        self.feature_size = feature_size
        self.max_frames = max_frames
        self.cluster_size = cluster_size
        self.batch_norm = batch_norm
        self.is_training = is_training
        self.scope_id = scope_id
        self.l2_penality = l2_penalty

    def forward(self, inputs, **unused_params):
        """
        :param inputs: (batch_size * num_samples) x feature_size
        :return: (batch_size * num_samples) x num_clusters
        """
        output = slim.fully_connected(
            inputs,
            self.cluster_size,
            activation_fn=tf.nn.sigmoid,
            weights_regularizer=slim.l2_regularizer(self.l2_penality)
        )
        return output


class ClMoeModel(modules.BaseModule):
    def __init__(self,
                 feature_size,
                 max_frames,
                 cluster_size,
                 batch_norm,
                 is_training,
                 scope_id=None,
                 num_mixtures=2,
                 l2_penalty=1e-8):
        self.feature_size = feature_size
        self.max_frames = max_frames
        self.cluster_size = cluster_size
        self.batch_norm = batch_norm
        self.is_training = is_training
        self.scope_id = scope_id
        self.num_mixtures = num_mixtures
        self.l2_penality = l2_penalty

    def forward(self, inputs, **unused_params):
        """
        :param inputs: (batch_size * num_samples) x feature_size
        :return: (batch_size * num_samples) x num_clusters
        """
        # inputs: (batch_size * num_samples) x feature_size
        # Compute MoE for all frames in batch.
        gate_activations = slim.fully_connected(
            inputs,
            self.cluster_size * (self.num_mixtures + 1),
            activation_fn=None,
            biases_initializer=None,
            weights_initializer=slim.l2_regularizer(self.l2_penality),
            scope="cl_moe_gates")
        # -> (batch_size * num_samples) x (num_clusters * (num_mixtures + 1))
        expert_activations = slim.fully_connected(
            inputs,
            self.cluster_size * self.num_mixtures,
            activation_fn=None,
            weights_regularizer=slim.l2_regularizer(self.l2_penality),
            scope="cl_moe_gates")
        # -> (batch_size * num_samples) x (num_clusters * num_mixtures)

        gating_distribution = tf.nn.softmax(tf.reshape(
            gate_activations, [-1, self.num_mixtures + 1]))
        # -> (batch_size * num_samples * num_cluster) x (num_mixtures + 1)
        expert_distribution = tf.nn.sigmoid(tf.reshape(
            expert_activations, [-1, self.num_mixtures]))
        # -> (batch_size * num_samples * num_cluster) x num_mixtures

        final_probabilities = tf.reduce_sum(
            gating_distribution[:, :self.num_mixtures] * expert_distribution, 1)
        final_probabilities = tf.reshape(final_probabilities,
                                         [-1, self.cluster_size])
        # -> (batch_size * num_samples) x num_clusters
        return final_probabilities


class ClVladModule(modules.BaseModule):
    def __init__(self,
                 feature_size,
                 max_frames,
                 cluster_size,
                 batch_norm,
                 is_training,
                 scope_id=None):
        self.feature_size = feature_size
        self.max_frames = max_frames
        self.is_training = is_training
        self.batch_norm = batch_norm
        self.cluster_size = cluster_size
        self.scope_id = scope_id

    def forward(self, inputs, **unused_params):
        cluster_weights = tf.get_variable((
            "clVladCluster",
            [self.feature_size, self.cluster_size]))
        tf.summary.histogram("clVladCluster",
                             cluster_weights)
        cluster_activation = tf.matmul(inputs, cluster_weights)
        # -> (batch_size * num_samples) x num_clusters

        if self.batch_norm:
            cluster_activation = slim.batch_norm(
                cluster_activation,
                center=True,
                scale=True,
                is_training=self.is_training,
                scope="clVladClusterBn")
        else:
            cluster_biases = tf.get_variable("clVladClusterBias",
                                             [self.cluster_size],
                                             initializer=tf.random_normal_initializer(
                                                 stddev=1 / math.sqrt(self.feature_size)))
            tf.summary.histogram("clVladClusterBias", cluster_biases)
            cluster_activation += cluster_biases
        cluster_activation = tf.nn.softmax(cluster_activation)
        tf.summary.histogram("cluster_activation", cluster_activation)

        cluster_activation = tf.reshape(cluster_activation,
                                        [-1, self.max_frames, self.cluster_size])
        # -> batch_size x num_samples x cluster_size
        activation_sum = tf.reduce_sum(cluster_activation, -2,
                                       keep_dims=True)

        cluster_weights2 = tf.get_variable("cluster_weights2",
                                           [1, self.feature_size, self.cluster_size],
                                           initializer=tf.random_normal_initializer(
                                               stddev=1 / math.sqrt(self.feature_size)))
        vlad_activation = tf.multiply(activation_sum,
                                      cluster_weights2)
        cluster_activation = tf.transpose(cluster_activation,
                                          perm=[0, 2, 1])
        reshaped_input = tf.reshape(inputs,
                                    [-1, self.max_frames, self.feature_size])

        vlad = tf.matmul(cluster_activation, reshaped_input)
        vlad = tf.transpose(vlad, perm=[0, 2, 1])
        vlad = tf.subtract(vlad, vlad_activation)
        vlad = tf.nn.l2_normalize(vlad, 1)
        vlad = tf.reshape(vlad, [-1, self.cluster_size * self.feature_size])
        vlad = tf.nn.l2_normalize(vlad, 1)

        # batch_size x (cluster_size * feature_size)
        return vlad


class ClFisherModule(modules.BaseModule):
    def __init__(self,
                 feature_size,
                 max_frames,
                 cluster_size,
                 batch_norm,
                 is_training,
                 scope_id=None):
        self.feature_size = feature_size
        self.max_frames = max_frames
        self.is_training = is_training
        self.batch_norm = batch_norm
        self.cluster_size = cluster_size
        self.scope_id = scope_id

    def forward(self, inputs, **unused_params):
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
        activation = tf.matmul(inputs, cluster_weights)
        if self.batch_norm:
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

        reshaped_input = tf.reshape(inputs, [-1, self.max_frames, self.feature_size])
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


class ClLstmModule(modules.BaseModule):
    def __init__(self,
                 cluster_size,
                 lstm_size,
                 num_layers,
                 num_frames,
                 scope_id=None):
        self.cluster_size = cluster_size
        self.lstm_size = lstm_size
        self.num_layers = num_layers
        self.num_frames = num_frames
        self.scope_id = scope_id

    def forward(self, inputs, **unused_params):
        """
        :param inputs: batch_size x num_frames x num_features
        :return:
        """

        stacked_lstm = tf.contrib.rnn.MultiRNNCell(
            [
                tf.contrib.rnn.BasicLSTMCell(
                    self.lstm_size, forget_bias=1.0)
                for _ in range(self.num_layers)
            ])

        outputs, state = tf.nn.dynamic_rnn(stacked_lstm, inputs,
                                           sequence_length=self.num_frames,
                                           dtype=tf.float32)
        # batch_size x cluster_size
        return state[-1].h


class ClGruModule(modules.BaseModule):
    def __init__(self,
                 cluster_size,
                 gru_size,
                 num_layers,
                 num_frames,
                 scope_id=None):
        self.cluster_size = cluster_size
        self.gru_size = gru_size
        self.num_layers = num_layers
        self.num_frames = num_frames
        self.scope_id = scope_id

    def forward(self, inputs, **unused_params):
        """
        :param inputs: batch_size x num_frames x num_features
        :return:
        """

        stacked_gru = tf.contrib.rnn.MultiRNNCell(
            [
                tf.contrib.rnn.GRUCell(
                    self.gru_size, forget_bias=1.0)
                for _ in range(self.num_layers)
            ])

        outputs, state = tf.nn.dynamic_rnn(stacked_gru, inputs,
                                           sequence_length=self.num_frames,
                                           dtype=tf.float32)
        # batch_size x cluster_size
        return state[-1].h


###############################################################################
# CNN Type pooling methods ####################################################
###############################################################################
class CnnV1(modules.BaseModule):
    """

    """
    def __init__(self, kernel_width, kernel_height, num_filter, is_training, scope_id=None):
        self.kernel_width = kernel_width
        self.kernel_height = kernel_height
        self.num_filter = num_filter
        self.is_training = is_training
        self.scope_id = scope_id

    def forward(self, inputs, **unused_params):
        cnn_activation = slim.conv2d(inputs=inputs,
                                     num_outputs=self.num_filter,
                                     kernel_size=[self.kernel_height, self.kernel_width],
                                     stride=1,
                                     padding="SAME",
                                     activation_fn=tf.nn.relu,
                                     scope="CnnV1_1")

        state = tf.reshape(cnn_activation, shape=[-1, self.num_filter])
        state = tf.nn.l2_normalize(state, dim=1)
        return state


###############################################################################
# RNN (LSTM, GRU) Type pooling methods ########################################
###############################################################################

class LstmModule(modules.BaseModule):
    def __init__(self, lstm_size, num_layers, max_frame):
        self.lstm_size = lstm_size
        self.num_layers = num_layers
        self.max_frame = max_frame

    def forward(self, inputs, **unused_params):
        stacked_lstm = tf.contrib.rnn.MultiRNNCell(
            [
                tf.contrib.rnn.BasicLSTMCell(
                    self.lstm_size, forget_bias=1.0)
                for _ in range(self.num_layers)
            ])

        loss = 0.0

        outputs, state = tf.nn.dynamic_rnn(stacked_lstm, inputs,
                                           sequence_length=self.max_frame,
                                           dtype=tf.float32)
        return state.h[-1]


class LstmConcatAverage(modules.BaseModule):
    def __init__(self, lstm_size, num_layers, max_frame):
        self.lstm_size = lstm_size
        self.num_layers = num_layers
        self.max_frame = max_frame

    def forward(self, inputs, **unused_params):
        stacked_lstm = tf.contrib.rnn.MultiRNNCell(
            [
                tf.contrib.rnn.BasicLSTMCell(
                    self.lstm_size, forget_bias=1.0, state_is_tuple=False)
                for _ in range(self.num_layers)
            ], state_is_tuple=False)

        loss = 0.0

        outputs, state = tf.nn.dynamic_rnn(stacked_lstm, inputs,
                                           sequence_length=self.max_frame,
                                           dtype=tf.float32)

        context_memory = tf.nn.l2_normalize(tf.reduce_sum(outputs, axis=1), dim=1)
        average_state = tf.nn.l2_normalize(tf.reduce_sum(inputs, axis=1), dim=1)
        # state = tf.concat([state[0], state[1]], 1)

        final_state = tf.concat([context_memory, state, average_state], 1)

        return final_state


###############################################################################
# Distribution-Learning methods                                  ##############
# Please look the copyright notice for each class                ##############
# VLAD implementation are based on WILLOW paper & public code    ##############
###############################################################################
class SparseNetVLAD(modules.BaseModule):
    """
    Sparse representation VLAD module.
    """
    def __init__(self, feature_size, max_frames, cluster_size, projection_size, batch_norm, is_training, scope_id=None):
        self.feature_size = feature_size
        self.max_frames = max_frames
        self.is_training = is_training
        self.projection_size = projection_size
        self.batch_norm = batch_norm
        self.cluster_size = cluster_size
        self.scope_id = scope_id

    def forward(self, inputs, **unused_params):
        projection_weights = tf.get_variable("projection_weights",
                                             [self.feature_size, self.projection_size],
                                             initializer=tf.random_normal_initializer(
                                                 stddev=1 / math.sqrt(self.feature_size)))
        tf.summary.histogram("projection_weights{}".format("" if self.scope_id is None else str(self.scope_id)),
                             projection_weights)
        # frame_size x feature_size, feature_size x projection_size
        # frame_size x projection_size
        project_activation = tf.matmul(inputs, projection_weights)

        cluster_weights = tf.get_variable("cluster_weights{}".format("" if self.scope_id is None
                                                                     else str(self.scope_id)),
                                          [self.projection_size, self.cluster_size],
                                          initializer=tf.random_normal_initializer(
                                              stddev=1 / math.sqrt(self.feature_size)))

        tf.summary.histogram("cluster_weights{}".format("" if self.scope_id is None else str(self.scope_id)),
                             cluster_weights)
        # frame_size x cluster_size
        activation = tf.matmul(project_activation, cluster_weights)

        if self.batch_norm:
            activation = slim.batch_norm(
                activation,
                center=True,
                scale=True,
                is_training=self.is_training,
                scope="cluster_bn")
        else:
            cluster_biases = tf.get_variable("cluster_biases{}".format("" if self.scope_id is None
                                                                       else str(self.scope_id)),
                                             [self.cluster_size],
                                             initializer=tf.random_normal_initializer(
                                                 stddev=1 / math.sqrt(self.projection_size)))
            tf.summary.histogram("cluster_biases", cluster_biases)
            activation += cluster_biases

        activation = tf.nn.softmax(activation)
        tf.summary.histogram("cluster_output", activation)

        # batch_size x frame_size x cluster_size
        activation = tf.reshape(activation, [-1, self.max_frames, self.cluster_size])

        # batch_size x 1 x cluster_size
        a_sum = tf.reduce_sum(activation, -2, keep_dims=True)

        # 1 x feature_size x cluster_size
        cluster_weights2 = tf.get_variable("cluster_weights2",
                                           [1, self.projection_size, self.cluster_size],
                                           initializer=tf.random_normal_initializer(
                                               stddev=1 / math.sqrt(self.projection_size)))

        a = tf.multiply(a_sum, cluster_weights2)

        activation = tf.transpose(activation, perm=[0, 2, 1])

        reshaped_input = tf.reshape(project_activation, [-1, self.max_frames, self.projection_size])
        vlad = tf.matmul(activation, reshaped_input)
        vlad = tf.transpose(vlad, perm=[0, 2, 1])
        vlad = tf.subtract(vlad, a)
        vlad = tf.nn.l2_normalize(vlad, 1)
        vlad = tf.reshape(vlad, [-1, self.cluster_size * self.projection_size])
        vlad = tf.nn.l2_normalize(vlad, 1)

        # batch_size x (cluster_size * feature_size)
        return vlad


class LightVLAD(modules.BaseModule):
    """
    LightVLAD version from public code in WILLOW paper & public code.
    https://github.com/antoine77340/Youtube-8M-WILLOW
    """
    def __init__(self, feature_size, max_frames, cluster_size, batch_norm, is_training, scope_id=None):
        self.feature_size = feature_size
        self.max_frames = max_frames
        self.is_training = is_training
        self.batch_norm = batch_norm
        self.cluster_size = cluster_size
        self.scope_id = scope_id

    def forward(self, inputs, **unused_params):
        cluster_weights = tf.get_variable("cluster_weights{}".format("" if self.scope_id is None
                                                                     else str(self.scope_id)),
                                          [self.feature_size, self.cluster_size],
                                          initializer=tf.random_normal_initializer(
                                              stddev=1 / math.sqrt(self.feature_size)))
        activation = tf.matmul(inputs, cluster_weights)

        if self.batch_norm:
            activation = slim.batch_norm(
                activation,
                center=True,
                scale=True,
                is_training=self.is_training,
                scope="cluster_bn")
        else:
            cluster_biases = tf.get_variable("cluster_biases{}".format("" if self.scope_id is None
                                                                       else str(self.scope_id)),
                                             [self.cluster_size],
                                             initializer=tf.random_normal_initializer(
                                                 stddev=1 / math.sqrt(self.feature_size)))
            tf.summary.histogram("cluster_biases", cluster_biases)
            activation += cluster_biases

        activation = tf.nn.softmax(activation)
        activation = tf.reshape(activation, [-1, self.max_frames, self.cluster_size])
        activation = tf.transpose(activation, perm=[0, 2, 1])
        reshaped_input = tf.reshape(inputs, [-1, self.max_frames, self.feature_size])

        vlad = tf.matmul(activation, reshaped_input)
        vlad = tf.transpose(vlad, perm=[0, 2, 1])
        vlad = tf.nn.l2_normalize(vlad, 1)
        vlad = tf.reshape(vlad, [-1, self.cluster_size * self.feature_size])
        vlad = tf.nn.l2_normalize(vlad, 1)

        # batch_size x (cluster_size * feature_size)
        return vlad


class NetVLAD(modules.BaseModule):
    """
    NetVLAD version from public code in WILLOW paper & public code.
    https://github.com/antoine77340/Youtube-8M-WILLOW
    """
    def __init__(self, feature_size, max_frames, cluster_size, batch_norm, is_training, scope_id=None):
        self.feature_size = feature_size
        self.max_frames = max_frames
        self.is_training = is_training
        self.batch_norm = batch_norm
        self.cluster_size = cluster_size
        self.scope_id = scope_id

    def forward(self, inputs, **unused_params):
        cluster_weights = tf.get_variable("cluster_weights{}".format("" if self.scope_id is None
                                                                     else str(self.scope_id)),
                                          [self.feature_size, self.cluster_size],
                                          initializer=tf.random_normal_initializer(
                                              stddev=1 / math.sqrt(self.feature_size)))

        tf.summary.histogram("cluster_weights{}".format("" if self.scope_id is None else str(self.scope_id)),
                             cluster_weights)
        activation = tf.matmul(inputs, cluster_weights)

        if self.batch_norm:
            activation = slim.batch_norm(
                activation,
                center=True,
                scale=True,
                is_training=self.is_training,
                scope="cluster_bn")
        else:
            cluster_biases = tf.get_variable("cluster_biases{}".format("" if self.scope_id is None
                                                                       else str(self.scope_id)),
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

        # batch_size x (cluster_size * feature_size)
        return vlad

