# Copyright 2016 Google Inc. All Rights Reserved.
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

import tensorflow as tf
import math


class ContextGateV1:
    """
    Given the weight W, calculate sigmoid(WX + b) o X. o is an element-wise
    multiplication.

    Citation: Learnable pooling with Context Gating for video classification.
    """
    def __init__(self, feature_size, cluster_size):
        """ Initialize a class ContextGateV1.
        The idea and implementation is adopted from WILLOW.
        :param feature_size:
        :param cluster_size:
        """
        self.feature_size = feature_size
        self.cluster_size = cluster_size

    def forward(self, reshaped_input):
        gate_weights = tf.get_variable("gate_weights",
                                       [1, self.cluster_size, self.feature_size],
                                       initializer=tf.random_normal_initializer(
                                           stddev=1 / math.sqrt(self.feature_size)))
        gate_weights = tf.sigmoid(gate_weights)
        multiplied_weights = tf.multiply(reshaped_input, gate_weights)
        return multiplied_weights


class DeepAttentionV1:
    def __init__(self, feature_size, cluster_size):
        self.feature_size = feature_size
        self.cluster_size = cluster_size

    def forward(self, reshaped_input):
        dav1_weights = tf.get_variable("dav1_weights",
                                       [1, self.cluster_size, self.feature_size],
                                       initializer=tf.random_normal_initializer(
                                           stddev=1 / math.sqrt(self.feature_size)))


class SparseAttentionV1:
    def __init__(self, variable_name, feature_size, projection_size, num_layers):
        self.variable_name = variable_name
        self.feature_size = feature_size
        self.projection_size = projection_size
        self.num_layers = num_layers

    def forward(self, features):
        """

        :param features: 'batch_size' x 'max_samples' x 'feature_size'
        :return: 'batch_size' x 'max_samples' x 'feature_size'
        """

        # 1. Project features into higher dimension


