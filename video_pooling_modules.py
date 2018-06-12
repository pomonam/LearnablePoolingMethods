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
# Distribution-Learning methods                                  ##############
# Please look the copyright notice for each class                ##############
# VLAD implementation are based on WILLOW paper & public code    ##############
###############################################################################
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

