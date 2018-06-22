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

from tensorflow import flags
import tensorflow as tf
import tensorflow.contrib.slim as slim
import modules
import math


###############################################################################
# Necessary FLAGS #############################################################
###############################################################################
FLAGS = flags.FLAGS


###############################################################################
# Prototype ###################################################################
###############################################################################
class TembeddingModule(modules.BaseModule):
    """

    """
    def __init__(self, feature_size, max_frames, cluster_size, batch_norm, is_training,
                 det_reg=1e-6, scope_id=None):
        self.feature_size = feature_size
        self.max_frames = max_frames
        self.is_training = is_training
        self.batch_norm = batch_norm
        self.cluster_size = cluster_size
        self.det_reg = det_reg
        self.scope_id = scope_id

    def forward(self, inputs, **unused_params):
        cluster_weights = tf.get_variable("cluster_weights{}".format("" if self.scope_id is None
                                                                     else str(self.scope_id)),
                                          [self.feature_size, self.cluster_size],
                                          initializer=tf.random_normal_initializer(
                                              stddev=1 / math.sqrt(self.feature_size)))
        tf.summary.histogram("cluster_weights{}".format("" if self.scope_id is None else str(self.scope_id)),
                             cluster_weights)

        # inputs: (batch_size * max_frames) x feature_size
        activation = tf.matmul(inputs, cluster_weights)
        # -> (batch_size * max_frames) x cluster_size
        activation = slim.batch_norm(
            activation,
            center=True,
            scale=True,
            is_training=self.is_training,
            scope="cluster_bn")
        activation = tf.nn.softmax(activation)
        # -> (batch_size * max_frames) x cluster_size
        tf.summary.histogram("cluster_output", activation)

        softmax_weight = tf.multiply(cluster_weights, activation)
        # -> (batch_size * max_frames) x cluster_size

        temb = tf.subtract(inputs, softmax_weight)
        temb = tf.nn.l2_normalize(temb, 1)
        temb = tf.reshape(temb, [-1, self.max_frames, self.cluster_size * self.feature_size])
        avg_temb = tf.reduce_mean(temb, 1)
        avg_temb = tf.reshape(avg_temb, [-1, self.cluster_size * self.feature_size])
        # batch_size x (cluster_size * feature_size)
        return avg_temb


class NetVLADetReg(modules.BaseModule):
    """
    NetVLAD version from public code in WILLOW paper & public code.
    https://github.com/antoine77340/Youtube-8M-WILLOW
    """
    def __init__(self, feature_size, max_frames, cluster_size, batch_norm, is_training,
                 det_reg=1e-6, scope_id=None):
        self.feature_size = feature_size
        self.max_frames = max_frames
        self.is_training = is_training
        self.batch_norm = batch_norm
        self.cluster_size = cluster_size
        self.det_reg = det_reg
        self.scope_id = scope_id

    def forward(self, inputs, **unused_params):
        cluster_weights = tf.get_variable("cluster_weights{}".format("" if self.scope_id is None
                                                                     else str(self.scope_id)),
                                          [self.feature_size, self.cluster_size],
                                          initializer=tf.random_normal_initializer(
                                              stddev=1 / math.sqrt(self.feature_size)))
        cluster_weights = tf.get_variable("cluster_weights{}",[self.feature_size, self.cluster_size],
                                          initializer=tf.random_normal_initializer(
                                              stddev=1 / math.sqrt(self.feature_size)))

        tf.summary.histogram("cluster_weights{}".format("" if self.scope_id is None else str(self.scope_id)),
                             cluster_weights)

        # Normalize the columns of cluster weights.
        norm_cluster_weights = tf.nn.l2_normalize(cluster_weights, axis=1)

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


class NetVLADNccReg(modules.BaseModule):
    """
    NetVLAD version from public code in WILLOW paper & public code.
    https://github.com/antoine77340/Youtube-8M-WILLOW
    """
    def __init__(self, feature_size, max_frames, cluster_size, batch_norm, is_training, cor_reg=1e-6, scope_id=None):
        self.feature_size = feature_size
        self.max_frames = max_frames
        self.is_training = is_training
        self.batch_norm = batch_norm
        self.cluster_size = cluster_size
        self.cor_reg = cor_reg
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


class NetVLADBowWeight(modules.BaseModule):
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

        # input: (batch_size * max_frames) x feature_size
        activation = tf.matmul(inputs, cluster_weights)
        # -> (batch_size * max_frames) x cluster_size

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
        # -> (batch_size * max_frames) x cluster_size
        tf.summary.histogram("cluster_output", activation)

        activation = tf.reshape(activation, [-1, self.max_frames, self.cluster_size])
        # -> batch_size x max_frames x cluster_size

        a_sum = tf.reduce_sum(activation, -2, keep_dims=True)
        # -> batch_size x 1 x cluster_size

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


###############################################################################
# State-of-Art Image Retrieval Pooling ########################################
###############################################################################
class MaxPoolingModule(modules.BaseModule):
    def __init__(self, feature_size, max_frames):
        self.feature_size = feature_size
        self.max_frames = max_frames

    def forward(self, inputs, **unused_params):
        return tf.reduce_max(inputs, 1)


class SpocPoolingModule(modules.BaseModule):
    def __init__(self, feature_size, max_frames):
        self.feature_size = feature_size
        self.max_frames = max_frames

    def forward(self, inputs, **unused_params):
        return tf.reduce_mean(inputs, 1)


class GemPoolingModule(modules.BaseModule):
    def __init__(self, feature_size, max_frames, eps=1e-6):
        """ Initialize class GemPoolingModule.
        GeM
        :param feature_size:
        :param max_frames:
        """
        self.feature_size = feature_size
        self.max_frames = max_frames
        self.eps = eps

    def forward(self, inputs, **unused_params):
        """

        :param inputs: batch_size x max_frames x num_features
        :return: batch_size x feature_size
        """
        p = tf.get_variable("p",
                            shape=[1])
        # Clip some values.
        frames = tf.clip_by_value(inputs, clip_value_min=self.eps, clip_value_max=None)
        frames = tf.pow(frames, p)
        frames = tf.reduce_mean(frames, 1)
        frames = tf.pow(frames, 1. / p)
        return frames


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
        self.l2_penalty = l2_penalty

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
            weights_regularizer=slim.l2_regularizer(self.l2_penalty))
        # -> (batch_size * num_samples) x (num_clusters * (num_mixtures + 1))
        expert_activations = slim.fully_connected(
            inputs,
            self.cluster_size * self.num_mixtures,
            activation_fn=None,
            weights_regularizer=slim.l2_regularizer(self.l2_penalty))
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
        print(inputs)
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
# RNN (LSTM, GRU) Type pooling methods ########################################
###############################################################################

class LstmModule(modules.BaseModule):
    def __init__(self, lstm_size, num_layers, num_frames):
        self.lstm_size = lstm_size
        self.num_layers = num_layers
        self.num_frames = num_frames

    def forward(self, inputs, **unused_params):
        stacked_lstm = tf.contrib.rnn.MultiRNNCell(
            [
                tf.contrib.rnn.BasicLSTMCell(
                    self.lstm_size, forget_bias=1.0)
                for _ in range(self.num_layers)
            ])

        loss = 0.0

        outputs, state = tf.nn.dynamic_rnn(stacked_lstm, inputs,
                                           sequence_length=self.num_frames,
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
