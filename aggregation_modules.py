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

""" Modules for feature pooling and aggregation. """

import tensorflow as tf
import modules


class IndirectClusterMeanPoolModule(modules.BaseModule):
    """
    Mean pooling method. Mean is computed from weighted average
    inspired from self-attention mechanism (indirect clustering)
    """
    def __init__(self, l2_normalize):
        """ Initialize IndirectClusterMaxMeanPoolModule
        :param l2_normalize: bool
        """
        self.l2_normalize = l2_normalize

    def forward(self, t_inputs, c_inputs, **unused_params):
        """ Forward method for max & mean pooling with indirect clustering (self-attention).
        :param t_inputs: batch_size x max_frames x num_features
        :param c_inputs: batch_size x max_frames x num_features
        :return: batch_size x feature_size
        """
        attention = tf.matmul(t_inputs, tf.transpose(t_inputs, perm=[0, 2, 1]))
        # -> batch_size x max_frames x max_frames
        attention = tf.expand_dims(attention, -1)
        attention = tf.nn.relu(attention)

        attention = tf.reduce_sum(attention, axis=2)
        # -> batch_size x max_frames x 1
        attention = tf.nn.softmax(attention, axis=1)

        mean_pool = tf.reduce_mean(tf.multiply(c_inputs, attention), axis=1)
        # -> batch_size x num_features

        if self.l2_normalize:
            mean_pool = tf.nn.l2_normalize(mean_pool, 1)

        return mean_pool



class IndirectClusterMaxMeanPoolModule(modules.BaseModule):
    """
    Max-Mean pooling method. Mean is computed from weighted average
    inspired from self-attention mechanism (indirect clustering)
    """
    def __init__(self, l2_normalize):
        """ Initialize IndirectClusterMaxMeanPoolModule
        :param l2_normalize: bool
        """
        self.l2_normalize = l2_normalize

    def forward(self, inputs, **unused_params):
        """ Forward method for max & mean pooling with indirect clustering (self-attention).
            Where
        :param inputs: batch_size x max_frames x num_features
        :return: batch_size x feature_size
        """
        attention   = tf.matmul(inputs, tf.transpose(inputs, perm=[0, 2, 1]))
        # -> batch_size x max_frames x max_frames
        attention = tf.expand_dims(attention, -1)
        attention = tf.nn.relu(attention)

        attention = tf.reduce_sum(attention, axis=2)
        # -> batch_size x max_frames x 1
        attention = tf.nn.softmax(attention, axis=1)

        mean_pool = tf.reduce_mean(tf.multiply(inputs, attention), axis=1)
        max_pool = tf.reduce_max(inputs, axis=1)
        # -> batch_size x num_features

        if self.l2_normalize:
            mean_pool = tf.nn.l2_normalize(mean_pool, 1)
            max_pool = tf.nn.l2_normalize(max_pool, 1)

        concat_pool = tf.concat([mean_pool, max_pool], 1)
        return concat_pool


class MaxMeanPoolingModule(modules.BaseModule):
    """ Max-Mean pooling method. """
    def __init__(self, l2_normalize=True):
        """ Initialize MaxMeanPoolingModule.
        :param l2_normalize: bool
        """
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
        # -> batch_size x num_features

        concat = tf.concat([max_pooled, avg_pooled], 1)
        return concat


class MaxPoolingModule(modules.BaseModule):
    """ Max pooling method. """
    def __init__(self, l2_normalize=False):
        """ Initialize MaxPoolingModule.
        :param l2_normalize: bool
        """
        self.l2_normalize = l2_normalize

    def forward(self, inputs, **unused_params):
        """ Forward method for max pooling.
        :param inputs: batch_size x max_frames x num_features
        :return: batch_size x feature_size
        """
        return tf.reduce_max(inputs, 1)

      
class MeanPooling(modules.BaseModule):
    """ Average pooling method. """
    def __init__(self, l2_normalize=False):
        """ Initialize MeanPooling.
        :param l2_normalize: bool
        """
        self.l2_normalize = l2_normalize

    def forward(self, inputs, **unused_params):
        """ Forward method for mean pooling.
        :param inputs: batch_size x max_frames x num_features
        :return: batch_size x feature_size
        """
        return tf.reduce_mean(inputs, 1)

      
class GemPoolingModule(modules.BaseModule):
    """ Generalized Mean Pooling. """
    def __init__(self, l2_normalize=False, eps=1e-6):
        """ Initialize GemPoolingModule.
        :param l2_normalize: bool
        """
        self.l2_normalize = l2_normalize
        self.eps = eps

    # TODO: Implementation is incorrect / incomplete.
    def forward(self, inputs, **unused_params):
        """ Forward method for GeM pooling
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
