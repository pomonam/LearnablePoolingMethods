# Copyright 2018 Deep Topology Inc. All Rights Reserved.
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

"""Modules for aggregating features."""

from tensorflow import flags
import tensorflow as tf
import tensorflow.contrib.slim as slim
import modules
import math


###############################################################################
# NN Based Aggregation ########################################################
###############################################################################
class NnAggregationModule(modules.BaseModule):
    def __init__(self, feature_size, max_frames, scope_id):
        self.feature_size = feature_size
        self.max_frames = max_frames
        self.scope_id = scope_id

    def forward(self, inputs, **unused_params):
        """ Neural Network based aggregation.
        :param inputs: batch_size x num_frames x feature_size
        :return: batch_size x feature_size
        """
        reshaped_inputs = tf.reshape(inputs, [-1, self.feature_size])
        reshaped_inputs_dim = reshaped_inputs.get_shape().as_list()[0]

        aggregation_weights_1 = tf.get_variable("aggregation_weights_1{}".format(self.scope_id),
                                              [self.feature_size, reshaped_inputs_dim],
                                              initializer=tf.random_normal_initializer(
                                                  stddev=1 / math.sqrt(reshaped_inputs_dim)))
        activation = tf.matmul(reshaped_inputs, aggregation_weights_1)
        activation = tf.nn.relu(activation)

        aggregation_weights_2 = tf.get_variable("aggregation_weights_2{}".format(self.scope_id),
                                              [reshaped_inputs_dim, reshaped_inputs_dim],
                                              initializer=tf.random_normal_initializer(
                                                  stddev=1 / math.sqrt(reshaped_inputs_dim)))
        activation = tf.matmul(activation, aggregation_weights_2)
        activation = tf.nn.relu(activation)

        return tf.reduce_mean(inputs * activation, 1)


###############################################################################
# State-of-Art Image Retrieval Aggregation ####################################
###############################################################################
class MaxMeanPoolingModule(modules.BaseModule):
    def __init__(self, feature_size, max_frames, l2_normalize=True):
        self.feature_size = feature_size
        self.max_frames = max_frames
        self.l2_normalize = l2_normalize

    def forward(self, inputs, **unused_params):
        """ Forward method for mean & max pooling.
        :param inputs: batch_size x max_frames x num_features
        :return: batch_size x feature_size
        """
        max_pooled = tf.reduce_max(inputs, 1)
        avg_pooled = tf.reduce_mean(inputs, 1)

        if self.l2_normalize:
            max_pooled = tf.nn.l2_normalize(max_pooled, 1)
            avg_pooled = tf.nn.l2_normalize(avg_pooled, 1)

        concat = tf.concat([max_pooled, avg_pooled], 1)
        return concat


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
