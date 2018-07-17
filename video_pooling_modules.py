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

"""Modules for pooling frame-level features."""

from tensorflow import flags
import tensorflow as tf
import tensorflow.contrib.slim as slim
import module_utils
import modules
import loupe_modules
import math
import attention_modules
import numbers
from tensorflow.python.platform import tf_logging as logging
from tensorflow.python.ops import standard_ops
from tensorflow.python.framework import ops

###############################################################################
# Necessary FLAGS #############################################################
###############################################################################
FLAGS = flags.FLAGS


###############################################################################
# Triangulation Embedding Methods #############################################
###############################################################################
class TriangulationV6Module(modules.BaseModule):
    """ CNN-integrated Triangulation Embedding Module with attention.
    """
    def __init__(self,
                 feature_size,
                 max_frames,
                 anchor_size,
                 self_attention,
                 hidden_layer_size,
                 kernel_size,
                 output_dim,
                 cluster_size,
                 add_relu,
                 batch_norm,
                 is_training,
                 scope_id=None):
        """ Initialize class TriangulationNsCnnIndirectAttentionModule.
        :param feature_size: int
        :param max_frames: max_frames x 1
        :param anchor_size: int
        :param self_attention: bool
        :param hidden_layer_size: int
        :param kernel_size: int
        :param output_dim: int
        :param add_relu: bool
        :param batch_norm: bool
        :param is_training: bool
        :param scope_id: Object
        """
        self.feature_size = feature_size
        self.max_frames = max_frames
        self.anchor_size = anchor_size
        self.self_attention = self_attention
        self.hidden_layer_size = hidden_layer_size
        self.kernel_size = kernel_size
        self.output_dim = output_dim
        self.add_relu = add_relu
        self.cluster_size = cluster_size
        self.batch_norm = batch_norm
        self.is_training = is_training
        self.scope_id = scope_id

    def forward(self, inputs, **unused_params):
        """ Forward method for TriangulationNsCnnIndirectAttentionModule.
        :param inputs: (batch_size * max_frames) x feature_size
        :return: batch_size x output_dim
        """
        anchor_weights = tf.get_variable("anchor_weights{}".format("" if self.scope_id is
                                                                         None else str(self.scope_id)),
                                         [self.feature_size, self.anchor_size],
                                         initializer=tf.contrib.layers.xavier_initializer())
        tf.summary.histogram("anchor_weights{}".format("" if self.scope_id is None else str(self.scope_id)),
                             anchor_weights)

        # Transpose weights for proper subtraction.
        anchor_weights = tf.transpose(anchor_weights)
        anchor_weights = tf.reshape(anchor_weights, [1, self.feature_size * self.anchor_size])

        # Tile inputs to subtract them with all anchors.
        tiled_inputs = tf.tile(inputs, [1, self.anchor_size])
        spatial = tf.subtract(tiled_inputs, anchor_weights)

        spatial = tf.reshape(spatial, [-1, self.anchor_size, self.feature_size])

        # Normalize the inputs for each frame; Obtain normalized residual vectors.
        spatial = tf.nn.l2_normalize(spatial, 2)
        spatial = tf.reshape(spatial, [-1, self.feature_size * self.anchor_size])
        spatial = tf.nn.l2_normalize(spatial, 1)

        if self.batch_norm:
            spatial = slim.batch_norm(
                spatial,
                center=True,
                scale=True,
                is_training=self.is_training,
                scope="spatial_bn")

        att = \
            attention_modules.OneFcAttention(self.feature_size * self.anchor_size, self.max_frames,
                                             self.cluster_size, do_shift=True)
        activation = att.forward(spatial)

        hidden_weight = tf.get_variable("hidden_weight",
                                        [self.feature_size * self.cluster_size * self.anchor_size,
                                         self.output_dim],
                                        initializer=tf.contrib.layers.xavier_initializer())

        spatial_activation = tf.matmul(activation, hidden_weight)

        if self.batch_norm:
            spatial_activation = slim.batch_norm(
                spatial_activation,
                center=True,
                scale=True,
                is_training=self.is_training,
                scope="spatial_pool2_bn")

        spatial_activation = tf.nn.relu(spatial_activation)

        return spatial_activation



class TriangulationV5Module(modules.BaseModule):
    """ CNN-integrated Triangulation Embedding Module with non-sharing CNN.
    """
    def __init__(self,
                 feature_size,
                 max_frames,
                 anchor_size,
                 self_attention,
                 hidden_layer_size,
                 kernel_size,
                 output_dim,
                 add_relu,
                 batch_norm,
                 is_training,
                 scope_id=None):
        """ Initialize class TriangulationNsCnnIndirectAttentionModule.
        :param feature_size: int
        :param max_frames: max_frames x 1
        :param anchor_size: int
        :param self_attention: bool
        :param hidden_layer_size: int
        :param kernel_size: int
        :param output_dim: int
        :param add_relu: bool
        :param batch_norm: bool
        :param is_training: bool
        :param scope_id: Object
        """
        self.feature_size = feature_size
        self.max_frames = max_frames
        self.anchor_size = anchor_size
        self.self_attention = self_attention
        self.hidden_layer_size = hidden_layer_size
        self.kernel_size = kernel_size
        self.output_dim = output_dim
        self.add_relu = add_relu
        self.batch_norm = batch_norm
        self.is_training = is_training
        self.scope_id = scope_id

    def forward(self, inputs, **unused_params):
        """ Forward method for TriangulationNsCnnIndirectAttentionModule.
        :param inputs: (batch_size * max_frames) x feature_size
        :return: batch_size x output_dim
        """
        ####################################################################################
        # Get spatial features with t-embedding ############################################
        ####################################################################################
        anchor_weights = tf.get_variable("anchor_weights{}".format("" if self.scope_id is
                                                                         None else str(self.scope_id)),
                                         [self.feature_size, self.anchor_size],
                                         initializer=tf.contrib.layers.xavier_initializer())
        tf.summary.histogram("anchor_weights{}".format("" if self.scope_id is None else str(self.scope_id)),
                             anchor_weights)

        # Transpose weights for proper subtraction.
        anchor_weights = tf.transpose(anchor_weights)
        anchor_weights = tf.reshape(anchor_weights, [1, self.feature_size * self.anchor_size])

        # Tile inputs to subtract them with all anchors.
        tiled_inputs = tf.tile(inputs, [1, self.anchor_size])
        spatial = tf.subtract(tiled_inputs, anchor_weights)

        spatial = tf.reshape(spatial, [-1, self.anchor_size, self.feature_size])
        spatial_norm = tf.norm(spatial, ord=2, axis=2, keepdims=False)
        # Normalize the inputs for each frame; Obtain normalized residual vectors.
        spatial = tf.nn.l2_normalize(spatial, 2)
        spatial = tf.reshape(spatial, [-1, self.feature_size * self.anchor_size])
        ####################################################################################

        ####################################################################################
        # Get temporal features from frame-difference in t-embedded feature. ###############
        ####################################################################################
        # Shift the input to the right (to subtract frame T-1 from frame T):
        cloned_spatial = tf.manip.roll(spatial, shift=1, axis=1)
        temporal = tf.subtract(spatial, cloned_spatial)
        temporal = tf.reshape(temporal, [-1, self.max_frames, self.feature_size * self.anchor_size])
        # Eliminate the first row.
        stacks = tf.unstack(temporal, axis=1)
        del stacks[0]
        temporal = tf.stack(stacks, 1)
        temporal = tf.reshape(temporal, [-1, self.anchor_size, self.feature_size])
        temporal_norm = tf.norm(temporal, ord=2, axis=2, keep_dims=False)
        temporal = tf.nn.l2_normalize(temporal, 2)

        # Both spatial, temporal have shape (batch_size * max_frames) x anchor_size x feature_size
        spatial = tf.reshape(spatial, [-1, self.anchor_size, self.feature_size])
        temporal = tf.reshape(temporal, [-1, self.anchor_size, self.feature_size])

        spatial_norm = tf.reshape(spatial_norm, [-1, self.max_frames, self.anchor_size])
        temporal_norm = tf.reshape(temporal_norm, [-1, self.max_frames - 1, self.anchor_size])
        ####################################################################################

        ####################################################################################
        # Reduce the number of parameters with non-share CNN ###############################
        ####################################################################################
        spatial_cnn_weights = tf.get_variable("spatial_cnn_weights{}".format(""
                                                                             if self.scope_id is None
                                                                             else str(self.scope_id)),
                                              [self.anchor_size, self.kernel_size, self.feature_size],
                                              initializer=tf.contrib.layers.xavier_initializer())
        temporal_cnn_weights = tf.get_variable("temporal_cnn_weights{}".format(""
                                                                               if self.scope_id is None
                                                                               else str(self.scope_id)),
                                               [self.anchor_size, self.kernel_size, self.feature_size],
                                               initializer=tf.contrib.layers.xavier_initializer())

        tp_spatial_cnn_weights = tf.transpose(spatial_cnn_weights, perm=[0, 2, 1])
        tp_temporal_cnn_weights = tf.transpose(temporal_cnn_weights, perm=[0, 2, 1])

        tp_spatial = tf.transpose(spatial, perm=[1, 0, 2])
        tp_temporal = tf.transpose(temporal, perm=[1, 0, 2])

        spatial_output = tf.matmul(tp_spatial, tp_spatial_cnn_weights)
        temporal_output = tf.matmul(tp_temporal, tp_temporal_cnn_weights)

        tp_spatial_output = tf.transpose(spatial_output, perm=[1, 0, 2])
        tp_temporal_output = tf.transpose(temporal_output, perm=[1, 0, 2])

        spatial_output = tf.reshape(tp_spatial_output, [-1, self.max_frames,
                                                        self.kernel_size * self.anchor_size])
        temporal_output = tf.reshape(tp_temporal_output, [-1, self.max_frames - 1,
                                                          self.kernel_size * self.anchor_size])

        spatial_output = tf.concat([spatial_output, spatial_norm], 2)
        temporal_output = tf.concat([temporal_output, temporal_norm], 2)

        spatial_mean = tf.reduce_mean(spatial_output, 1)
        temporal_mean = tf.reduce_mean(temporal_output, 1)

        spatial_variance = module_utils.reduce_var(spatial_output, 1)
        temporal_variance = module_utils.reduce_var(temporal_output, 1)

        spatial_pool = tf.concat([spatial_mean, spatial_variance], 1)
        temporal_pool = tf.concat([temporal_mean, temporal_variance], 1)

        if self.batch_norm:
            spatial_pool = slim.batch_norm(
                spatial_pool,
                center=True,
                scale=True,
                is_training=self.is_training,
                scope="spatial_pool_bn")
            temporal_pool = slim.batch_norm(
                temporal_pool,
                center=True,
                scale=True,
                is_training=self.is_training,
                scope="temporal_pool_bn")

        spatial_dim = spatial_pool.get_shape().as_list()[1]
        spatial_weights = tf.get_variable("spatial_hidden",
                                          [spatial_dim, self.hidden_layer_size],
                                          initializer=tf.contrib.layers.xavier_initializer())

        temporal_dim = temporal_pool.get_shape().as_list()[1]
        temporal_weights = tf.get_variable("temporal_hidden",
                                           [temporal_dim, self.hidden_layer_size],
                                           initializer=tf.contrib.layers.xavier_initializer())

        spatial_activation = tf.matmul(spatial_pool, spatial_weights)
        temporal_activation = tf.matmul(temporal_pool, temporal_weights)

        if self.batch_norm:
            spatial_activation = slim.batch_norm(
                spatial_activation,
                center=True,
                scale=True,
                is_training=self.is_training,
                scope="spatial_activation_bn")

            temporal_activation = slim.batch_norm(
                temporal_activation,
                center=True,
                scale=True,
                is_training=self.is_training,
                scope="temporal_activation_bn")

        spatial_activation = tf.nn.relu(spatial_activation)
        temporal_activation = tf.nn.relu(temporal_activation)

        spatial_weights2 = tf.get_variable("spatial_hidden2",
                                           [self.hidden_layer_size, self.hidden_layer_size],
                                           initializer=tf.contrib.layers.xavier_initializer())

        temporal_weights2 = tf.get_variable("temporal_hidden2",
                                            [self.hidden_layer_size, self.hidden_layer_size],
                                            initializer=tf.contrib.layers.xavier_initializer())

        spatial_activation = tf.matmul(spatial_activation, spatial_weights2)
        temporal_activation = tf.matmul(temporal_activation, temporal_weights2)

        if self.batch_norm:
            spatial_activation = slim.batch_norm(
                spatial_activation,
                center=True,
                scale=True,
                is_training=self.is_training,
                scope="spatial_pool2_bn")
            temporal_activation = slim.batch_norm(
                temporal_activation,
                center=True,
                scale=True,
                is_training=self.is_training,
                scope="temporal_pool2_bn")

        spatial_activation = tf.nn.relu(spatial_activation)
        temporal_activation = tf.nn.relu(temporal_activation)

        ####################################################################################
        # Fuse spatial & temporal features #################################################
        ####################################################################################
        spatial_temporal_concat = tf.concat([spatial_activation, temporal_activation], 1)

        sp_dim = spatial_temporal_concat.get_shape().as_list()[1]
        sp_weights = tf.get_variable("spa_temp_fusion",
                                     [sp_dim, self.output_dim],
                                     initializer=tf.contrib.layers.xavier_initializer())
        activation = tf.matmul(spatial_temporal_concat, sp_weights)

        if self.batch_norm:
            activation = slim.batch_norm(
                activation,
                center=True,
                scale=True,
                is_training=self.is_training,
                scope="st_fuse_activation_bn")

        activation = tf.nn.relu(activation)
        ####################################################################################

        return activation


class TriangulationEmbedding(modules.BaseModule):
    """ Triangulation embedding for each frame.

    References:
    Triangulation embedding and democratic aggregation for image search.
    """
    def __init__(self,
                 feature_size,
                 max_frames,
                 anchor_size,
                 batch_norm,
                 is_training,
                 scope_id=None):
        self.feature_size = feature_size
        self.max_frames = max_frames
        self.batch_norm = batch_norm
        self.anchor_size = anchor_size
        self.is_training = is_training
        self.scope_id = scope_id

    def forward(self, inputs, **unused_params):
        """ Forward method for Triangulation Embedding.
        :param inputs: (batch_size * max_frames) x feature_size
        :return: (batch_size * max_frames) x (feature_size * anchor_size)
        """
        anchor_weights = tf.get_variable("anchor_weights{}".format("" if self.scope_id is None else str(self.scope_id)),
                                         [self.feature_size, self.anchor_size],
                                         initializer=tf.random_normal_initializer(
                                              stddev=1 / math.sqrt(self.anchor_size)),
                                         dtype=tf.float32)
        tf.summary.histogram("anchor_weights{}".format("" if self.scope_id is None else str(self.scope_id)),
                             anchor_weights)

        # Normalize each columns.
        anchor_weights = tf.nn.l2_normalize(anchor_weights, axis=0)

        # Transpose weights for proper subtraction; See investigation.
        anchor_weights = tf.transpose(anchor_weights)
        anchor_weights = tf.reshape(anchor_weights, [1, self.feature_size * self.anchor_size])

        # Tile inputs to subtract them with all anchors.
        tiled_inputs = tf.tile(inputs, [1, self.anchor_size])
        # -> (batch_size * max_frames) x (feature_size * anchor_size)
        t_emb = tf.subtract(tiled_inputs, anchor_weights)
        # -> (batch_size * max_frames) x (feature_size * anchor_size)

        t_emb = tf.reshape(t_emb, [-1, self.anchor_size, self.feature_size])
        # Normalize the inputs for each frame.
        t_emb = tf.nn.l2_normalize(t_emb, 2)
        t_emb = tf.reshape(t_emb, [-1, self.feature_size * self.anchor_size])
        # -> (batch_size * max_frames) x (feature_size * anchor_size)

        return t_emb


class TriangulationCnnIndirectAttentionModule(modules.BaseModule):
    """ CNN-integrated Triangulation Embedding Module
    """
    def __init__(self,
                 feature_size,
                 max_frames,
                 anchor_size,
                 self_attention,
                 hidden_layer_size,
                 output_dim,
                 add_relu,
                 batch_norm,
                 is_training,
                 scope_id=None):
        """ Initialize class TriangulationIndirectAttentionModule.
        :param feature_size: int
        :param max_frames: max_frames x 1
        :param anchor_size: int
        :param self_attention: bool
        :param hidden_layer_size: int
        :param output_dim: int
        :param add_relu: bool
        :param batch_norm: bool
        :param is_training: bool
        :param scope_id: Object
        """
        self.feature_size = feature_size
        self.max_frames = max_frames
        self.anchor_size = anchor_size
        self.self_attention = self_attention
        self.hidden_layer_size = hidden_layer_size
        self.output_dim = output_dim
        self.add_relu = add_relu
        self.batch_norm = batch_norm
        self.is_training = is_training
        self.scope_id = scope_id

    def forward(self, inputs, **unused_params):
        """ Forward method for TriangulationCnnModule.
        :param inputs: (batch_size * max_frames) x feature_size
        :return: batch_size x output_dim
        """
        ####################################################################################
        # Get spatial features with t-embedding ############################################
        ####################################################################################
        anchor_weights = tf.get_variable("anchor_weights{}".format("" if self.scope_id is None else str(self.scope_id)),
                                         [self.feature_size, self.anchor_size],
                                         initializer=tf.random_normal_initializer(
                                             stddev=1 / math.sqrt(self.anchor_size)),
                                         dtype=tf.float32)
        tf.summary.histogram("anchor_weights{}".format("" if self.scope_id is None else str(self.scope_id)),
                             anchor_weights)

        # Transpose weights for proper subtraction.
        anchor_weights = tf.transpose(anchor_weights)
        anchor_weights = tf.reshape(anchor_weights, [1, self.feature_size * self.anchor_size])

        # Tile inputs to subtract them with all anchors.
        tiled_inputs = tf.tile(inputs, [1, self.anchor_size])
        # -> (batch_size * max_frames) x (feature_size * anchor_size)
        spatial = tf.subtract(tiled_inputs, anchor_weights)
        # -> (batch_size * max_frames) x (feature_size * anchor_size)

        spatial = tf.reshape(spatial, [-1, self.anchor_size, self.feature_size])
        # Normalize the inputs for each frame; Obtain normalized residual vectors.
        spatial = tf.nn.l2_normalize(spatial, 2)
        spatial = tf.reshape(spatial, [-1, self.feature_size * self.anchor_size])
        # -> (batch_size * max_frames) x (feature_size * anchor_size)
        ####################################################################################

        ####################################################################################
        # Get temporal features from frame-difference in t-embedded feature. ###############
        ####################################################################################
        cloned_spatial = tf.identity(spatial)
        # Shift the input to the right (to subtract frame T-1 from frame T):
        cloned_spatial = tf.manip.roll(cloned_spatial, shift=1, axis=1)
        temporal = tf.subtract(spatial, cloned_spatial)
        temporal = tf.reshape(temporal, [-1, self.max_frames, self.feature_size * self.anchor_size])
        # Eliminate the first row.
        stacks = tf.unstack(temporal, axis=1)
        del stacks[0]
        temporal = tf.stack(stacks, 1)
        temporal = tf.reshape(temporal, [-1, self.feature_size * self.anchor_size])

        # It becomes redundant & gives too much weight to main direction.
        # -> Whiten the representation.
        if self.batch_norm:
            spatial = slim.batch_norm(
                spatial,
                center=True,
                scale=True,
                is_training=self.is_training,
                scope="spatial_bn")

            temporal = slim.batch_norm(
                temporal,
                center=True,
                scale=True,
                is_training=self.is_training,
                scope="temporal_bn")

        # Both spatial, temporal have shape (batch_size * max_frames) x (feature_size * anchor_size)
        spatial = tf.reshape(spatial, [-1, self.max_frames, self.feature_size * self.anchor_size])
        temporal = tf.reshape(temporal, [-1, self.max_frames - 1, self.feature_size * self.anchor_size])

        ####################################################################################
        # Calculate weights for soft attention mechanism ###################################
        ####################################################################################
        spatial_attention = tf.matmul(spatial, tf.transpose(spatial, perm=[0, 2, 1]))
        temporal_attention = tf.matmul(temporal, tf.transpose(temporal, perm=[0, 2, 1]))

        spatial_attention = tf.expand_dims(spatial_attention, -1)
        temporal_attention = tf.expand_dims(temporal_attention, -1)

        # Zero-out negative weight.
        spatial_attention = tf.nn.relu(spatial_attention)
        temporal_attention = tf.nn.relu(temporal_attention)

        spatial_attention = tf.reduce_sum(spatial_attention, axis=2)
        temporal_attention = tf.reduce_sum(temporal_attention, axis=2)
        # -> batch_size x max_frames x 1

        spatial_attention_weight = tf.nn.softmax(spatial_attention, axis=1)
        temporal_attention_weight = tf.nn.softmax(temporal_attention, axis=1)
        ####################################################################################

        ####################################################################################
        # Reduce the number of parameters ##################################################
        ####################################################################################
        if self.self_attention:
            spatial_mean = tf.reduce_mean(tf.multiply(spatial, spatial_attention_weight), 1)
            temporal_mean = tf.reduce_mean(tf.multiply(temporal, temporal_attention_weight), 1)
        else:
            spatial_mean = tf.reduce_mean(spatial, 1)
            temporal_mean = tf.reduce_mean(temporal, 1)

        spatial_variance = module_utils.reduce_var(spatial, 1)
        temporal_variance = module_utils.reduce_var(temporal, 1)

        spatial_pool = tf.concat([spatial_mean, spatial_variance], 1)
        temporal_pool = tf.concat([temporal_mean, temporal_variance], 1)

        spatial_dim = spatial_pool.get_shape().as_list()[1]
        spatial_weights = tf.get_variable("spatial_hidden",
                                          [spatial_dim, self.hidden_layer_size],
                                          initializer=tf.random_normal_initializer(
                                              stddev=1 / math.sqrt(self.hidden_layer_size)))

        temporal_dim = temporal_pool.get_shape().as_list()[1]
        temporal_weights = tf.get_variable("temporal_hidden",
                                           [temporal_dim, self.hidden_layer_size],
                                           initializer=tf.random_normal_initializer(
                                              stddev=1 / math.sqrt(self.hidden_layer_size)))

        spatial_activation = tf.matmul(spatial_pool, spatial_weights)
        temporal_activation = tf.matmul(temporal_pool, temporal_weights)

        if self.batch_norm:
            spatial_activation = slim.batch_norm(
                spatial_activation,
                center=True,
                scale=True,
                is_training=self.is_training,
                scope="spatial_activation_bn")

            temporal_activation = slim.batch_norm(
                temporal_activation,
                center=True,
                scale=True,
                is_training=self.is_training,
                scope="temporal_activation_bn")

        if self.add_relu:
            spatial_activation = tf.nn.relu(spatial_activation)
            temporal_activation = tf.nn.relu(temporal_activation)

        ####################################################################################
        # Fuse spatial & temporal features #################################################
        ####################################################################################
        spatial_temporal_concat = tf.concat([spatial_activation, temporal_activation], 1)

        sp_dim = spatial_temporal_concat.get_shape().as_list()[1]
        sp_weights = tf.get_variable("spa_temp_fusion",
                                     [sp_dim, self.output_dim],
                                     initializer=tf.random_normal_initializer(
                                         stddev=1 / math.sqrt(self.hidden_layer_size)))
        activation = tf.matmul(spatial_temporal_concat, sp_weights)

        if self.batch_norm:
            activation = slim.batch_norm(
                activation,
                center=True,
                scale=True,
                is_training=self.is_training,
                scope="activation_bn")

        if self.add_relu:
            activation = tf.nn.relu(activation)
        ####################################################################################

        return activation


class TriangulationMagnitudeNsCnnIndirectAttentionModule(modules.BaseModule):
    """ CNN-integrated Triangulation Embedding Module with non-sharing CNN / magnitude preserving.
    """
    def __init__(self,
                 feature_size,
                 max_frames,
                 anchor_size,
                 self_attention,
                 hidden_layer_size,
                 kernel_size,
                 output_dim,
                 add_relu,
                 add_norm,
                 batch_norm,
                 is_training,
                 scope_id=None):
        """ Initialize class TriangulationMagnitudeNsCnnIndirectAttentionModule.
        :param feature_size: int
        :param max_frames: max_frames x 1
        :param anchor_size: int
        :param self_attention: bool
        :param hidden_layer_size: int
        :param kernel_size: int
        :param output_dim: int
        :param add_relu: bool
        :param batch_norm: bool
        :param is_training: bool
        :param scope_id: Object
        """
        self.feature_size = feature_size
        self.max_frames = max_frames
        self.anchor_size = anchor_size
        self.self_attention = self_attention
        self.hidden_layer_size = hidden_layer_size
        self.kernel_size = kernel_size
        self.output_dim = output_dim
        self.add_relu = add_relu
        self.add_norm = add_norm
        self.batch_norm = batch_norm
        self.is_training = is_training
        self.scope_id = scope_id

    def forward(self, inputs, **unused_params):
        """ Forward method for TriangulationNsCnnIndirectAttentionModule.
        :param inputs: (batch_size * max_frames) x feature_size
        :return: batch_size x output_dim
        """
        ####################################################################################
        # Get spatial features with t-embedding ############################################
        ####################################################################################
        anchor_weights = tf.get_variable("anchor_weights{}".format("" if self.scope_id is None else str(self.scope_id)),
                                         [self.feature_size, self.anchor_size],
                                         initializer=tf.orthogonal_initializer(),
                                         dtype=tf.float32)
        tf.summary.histogram("anchor_weights{}".format("" if self.scope_id is None else str(self.scope_id)),
                             anchor_weights)

        # Transpose weights for proper subtraction.
        anchor_weights = tf.transpose(anchor_weights)
        anchor_weights = tf.reshape(anchor_weights, [1, self.feature_size * self.anchor_size])

        # Tile inputs to subtract them with all anchors.
        tiled_inputs = tf.tile(inputs, [1, self.anchor_size])
        # -> (batch_size * max_frames) x (feature_size * anchor_size)
        spatial = tf.subtract(tiled_inputs, anchor_weights)
        # -> (batch_size * max_frames) x (feature_size * anchor_size)

        spatial = tf.reshape(spatial, [-1, self.anchor_size, self.feature_size])
        spatial_norm = tf.norm(spatial, ord=2, axis=2, keepdims=False)
        # -> (batch_size * max_frames) x anchor_size

        # Normalize the inputs for each frame; Obtain normalized residual vectors.
        spatial = tf.nn.l2_normalize(spatial, 2)
        spatial = tf.reshape(spatial, [-1, self.feature_size * self.anchor_size])
        # -> (batch_size * max_frames) x (feature_size * anchor_size)
        spatial_norm = tf.reshape(spatial_norm, [-1, self.max_frames, self.anchor_size])
        # -> batch_size x max_frames x anchor_size
        ####################################################################################

        ####################################################################################
        # Get temporal features from frame-difference in t-embedded feature. ###############
        ####################################################################################
        cloned_spatial = tf.identity(spatial)
        # Shift the input to the right (to subtract frame T-1 from frame T):
        cloned_spatial = tf.manip.roll(cloned_spatial, shift=1, axis=1)
        temporal = tf.subtract(spatial, cloned_spatial)
        temporal = tf.reshape(temporal, [-1, self.max_frames, self.feature_size * self.anchor_size])
        # Eliminate the first row.
        stacks = tf.unstack(temporal, axis=1)
        del stacks[0]
        temporal = tf.stack(stacks, 1)
        temporal = tf.reshape(temporal, [-1, self.anchor_size, self.feature_size])
        temporal_norm = tf.norm(temporal, ord=2, axis=2, keep_dims=False)
        # -> (batch_size * max_frames) x anchor_size
        temporal_norm = tf.reshape(temporal_norm, [-1, self.max_frames - 1, self.anchor_size])
        # -> batch_size x (max_frames - 1) x anchor_size

        # Normalize the inputs for each frame; Obtain normalized residual vectors.
        temporal = tf.nn.l2_normalize(temporal, 2)

        # Both spatial, temporal have shape (batch_size * max_frames) x anchor_size x feature_size
        spatial = tf.reshape(spatial, [-1, self.anchor_size, self.feature_size])
        temporal = tf.reshape(temporal, [-1, self.anchor_size, self.feature_size])

        ####################################################################################
        # Calculate weights for soft attention mechanism ###################################
        ####################################################################################
        spatial_attention = tf.matmul(spatial, tf.transpose(spatial, perm=[0, 2, 1]))
        temporal_attention = tf.matmul(temporal, tf.transpose(temporal, perm=[0, 2, 1]))

        spatial_attention = tf.expand_dims(spatial_attention, -1)
        temporal_attention = tf.expand_dims(temporal_attention, -1)

        # Zero-out negative weight.
        spatial_attention = tf.nn.relu(spatial_attention)
        temporal_attention = tf.nn.relu(temporal_attention)

        spatial_attention = tf.reduce_sum(spatial_attention, axis=2)
        temporal_attention = tf.reduce_sum(temporal_attention, axis=2)
        # -> batch_size x max_frames x 1

        spatial_attention_weight = tf.nn.softmax(spatial_attention, axis=1)
        temporal_attention_weight = tf.nn.softmax(temporal_attention, axis=1)
        ####################################################################################

        ####################################################################################
        # Reduce the number of parameters with non-share CNN ###############################
        ####################################################################################
        spatial_cnn_weights = tf.get_variable("spatial_cnn_weights{}".format(""
                                                                             if self.scope_id is None
                                                                             else str(self.scope_id)),
                                              [self.anchor_size, self.kernel_size, self.feature_size],
                                              initializer=tf.random_normal_initializer(
                                                  stddev=1 / math.sqrt(self.kernel_size * self.feature_size)),
                                              dtype=tf.float32)
        temporal_cnn_weights = tf.get_variable("temporal_cnn_weights{}".format(""
                                                                               if self.scope_id is None
                                                                               else str(self.scope_id)),
                                               [self.anchor_size, self.kernel_size, self.feature_size],
                                               initializer=tf.random_normal_initializer(
                                                  stddev=1 / math.sqrt(self.kernel_size * self.feature_size)),
                                               dtype=tf.float32)

        tp_spatial_cnn_weights = tf.transpose(spatial_cnn_weights, perm=[0, 2, 1])
        tp_temporal_cnn_weights = tf.transpose(temporal_cnn_weights, perm=[0, 2, 1])

        tp_spatial = tf.transpose(spatial, perm=[1, 0, 2])
        tp_temporal = tf.transpose(temporal, perm=[1, 0, 2])

        spatial_output = tf.matmul(tp_spatial, tp_spatial_cnn_weights)
        temporal_output = tf.matmul(tp_temporal, tp_temporal_cnn_weights)

        tp_spatial_output = tf.transpose(spatial_output, perm=[1, 0, 2])
        tp_temporal_output = tf.transpose(temporal_output, perm=[1, 0, 2])
        # -> (batch_size * max_frames) x anchor_size x kernel_size

        # Apply rectified activation.
        spatial_output = tf.reshape(tp_spatial_output, [-1, self.kernel_size * self.anchor_size])
        temporal_output = tf.reshape(tp_temporal_output, [-1, self.kernel_size * self.anchor_size])
        spatial_output = tf.nn.relu(spatial_output)
        temporal_output = tf.nn.relu(temporal_output)

        spatial_output = tf.reshape(spatial_output, [-1, self.max_frames, self.kernel_size * self.anchor_size])
        temporal_output = tf.reshape(temporal_output, [-1, self.max_frames - 1, self.kernel_size * self.anchor_size])

        if self.add_norm:
            spatial_output = tf.concat([spatial_output, spatial_norm], 2)
            temporal_output = tf.concat([temporal_output, temporal_norm], 2)

        if self.self_attention:
            spatial_mean = tf.reduce_mean(tf.multiply(spatial_output, spatial_attention_weight), 1)
            temporal_mean = tf.reduce_mean(tf.multiply(temporal_output, temporal_attention_weight), 1)
        else:
            spatial_mean = tf.reduce_mean(spatial_output, 1)
            temporal_mean = tf.reduce_mean(temporal_output, 1)

        spatial_variance = module_utils.reduce_var(spatial_output, 1)
        temporal_variance = module_utils.reduce_var(temporal_output, 1)

        spatial_pool = tf.concat([spatial_mean, spatial_variance], 1)
        temporal_pool = tf.concat([temporal_mean, temporal_variance], 1)

        if self.batch_norm:
            spatial_pool = slim.batch_norm(
                spatial_pool,
                center=True,
                scale=True,
                is_training=self.is_training,
                scope="spatial_pool_bn")
            temporal_pool = slim.batch_norm(
                temporal_pool,
                center=True,
                scale=True,
                is_training=self.is_training,
                scope="temporal_pool_bn")

        spatial_dim = spatial_pool.get_shape().as_list()[1]
        spatial_weights = tf.get_variable("spatial_hidden",
                                          [spatial_dim, self.hidden_layer_size],
                                          initializer=tf.random_normal_initializer(
                                              stddev=1 / math.sqrt(self.hidden_layer_size)))

        temporal_dim = temporal_pool.get_shape().as_list()[1]
        temporal_weights = tf.get_variable("temporal_hidden",
                                           [temporal_dim, self.hidden_layer_size],
                                           initializer=tf.random_normal_initializer(
                                              stddev=1 / math.sqrt(self.hidden_layer_size)))

        spatial_activation = tf.matmul(spatial_pool, spatial_weights)
        temporal_activation = tf.matmul(temporal_pool, temporal_weights)

        if self.batch_norm:
            spatial_activation = slim.batch_norm(
                spatial_activation,
                center=True,
                scale=True,
                is_training=self.is_training,
                scope="spatial_activation_bn")

            temporal_activation = slim.batch_norm(
                temporal_activation,
                center=True,
                scale=True,
                is_training=self.is_training,
                scope="temporal_activation_bn")

        if self.add_relu:
            spatial_activation = tf.nn.relu(spatial_activation)
            temporal_activation = tf.nn.relu(temporal_activation)

        ####################################################################################
        # Fuse spatial & temporal features #################################################
        ####################################################################################
        spatial_temporal_concat = tf.concat([spatial_activation, temporal_activation], 1)

        sp_dim = spatial_temporal_concat.get_shape().as_list()[1]
        sp_weights = tf.get_variable("spa_temp_fusion",
                                     [sp_dim, self.output_dim],
                                     initializer=tf.random_normal_initializer(
                                         stddev=1 / math.sqrt(self.output_dim)))
        activation = tf.matmul(spatial_temporal_concat, sp_weights)

        if self.batch_norm:
            activation = slim.batch_norm(
                activation,
                center=True,
                scale=True,
                is_training=self.is_training,
                scope="activation_bn")

        if self.add_relu:
            activation = tf.nn.relu(activation)
        ####################################################################################

        return activation


class TriangulationMagnitudeNsCnnNetVladModule(modules.BaseModule):
    """ CNN-integrated Triangulation Embedding Module with non-sharing CNN / magnitude preserving.
    """
    def __init__(self,
                 feature_size,
                 max_frames,
                 anchor_size,
                 self_attention,
                 hidden_layer_size,
                 kernel_size,
                 output_dim,
                 add_relu,
                 add_norm,
                 batch_norm,
                 is_training,
                 scope_id=None):
        """ Initialize class TriangulationMagnitudeNsCnnIndirectAttentionModule.
        :param feature_size: int
        :param max_frames: max_frames x 1
        :param anchor_size: int
        :param self_attention: bool
        :param hidden_layer_size: int
        :param kernel_size: int
        :param output_dim: int
        :param add_relu: bool
        :param batch_norm: bool
        :param is_training: bool
        :param scope_id: Object
        """
        self.feature_size = feature_size
        self.max_frames = max_frames
        self.anchor_size = anchor_size
        self.self_attention = self_attention
        self.hidden_layer_size = hidden_layer_size
        self.kernel_size = kernel_size
        self.output_dim = output_dim
        self.add_relu = add_relu
        self.add_norm = add_norm
        self.batch_norm = batch_norm
        self.is_training = is_training
        self.scope_id = scope_id

    def forward(self, inputs, **unused_params):
        """ Forward method for TriangulationNsCnnIndirectAttentionModule.
        :param inputs: (batch_size * max_frames) x feature_size
        :return: batch_size x output_dim
        """
        ####################################################################################
        # Get spatial features with t-embedding ############################################
        ####################################################################################
        anchor_weights = tf.get_variable("anchor_weights{}".format("" if self.scope_id is None else str(self.scope_id)),
                                         [self.feature_size, self.anchor_size],
                                         initializer=tf.orthogonal_initializer(),
                                         dtype=tf.float32)
        tf.summary.histogram("anchor_weights{}".format("" if self.scope_id is None else str(self.scope_id)),
                             anchor_weights)

        # Transpose weights for proper subtraction.
        anchor_weights = tf.transpose(anchor_weights)
        anchor_weights = tf.reshape(anchor_weights, [1, self.feature_size * self.anchor_size])

        # Tile inputs to subtract them with all anchors.
        tiled_inputs = tf.tile(inputs, [1, self.anchor_size])
        # -> (batch_size * max_frames) x (feature_size * anchor_size)
        spatial = tf.subtract(tiled_inputs, anchor_weights)
        # -> (batch_size * max_frames) x (feature_size * anchor_size)

        spatial = tf.reshape(spatial, [-1, self.anchor_size, self.feature_size])
        spatial_norm = tf.norm(spatial, ord=2, axis=2, keepdims=False)
        # -> (batch_size * max_frames) x anchor_size

        # Normalize the inputs for each frame; Obtain normalized residual vectors.
        spatial = tf.nn.l2_normalize(spatial, 2)
        spatial = tf.reshape(spatial, [-1, self.feature_size * self.anchor_size])
        # -> (batch_size * max_frames) x (feature_size * anchor_size)
        spatial_norm = tf.reshape(spatial_norm, [-1, self.max_frames, self.anchor_size])
        # -> batch_size x max_frames x anchor_size
        ####################################################################################

        ####################################################################################
        # Get temporal features from frame-difference in t-embedded feature. ###############
        ####################################################################################
        cloned_spatial = tf.identity(spatial)
        # Shift the input to the right (to subtract frame T-1 from frame T):
        cloned_spatial = tf.manip.roll(cloned_spatial, shift=1, axis=1)
        temporal = tf.subtract(spatial, cloned_spatial)
        temporal = tf.reshape(temporal, [-1, self.max_frames, self.feature_size * self.anchor_size])
        # Eliminate the first row.
        stacks = tf.unstack(temporal, axis=1)
        del stacks[0]
        temporal = tf.stack(stacks, 1)
        temporal = tf.reshape(temporal, [-1, self.anchor_size, self.feature_size])
        temporal_norm = tf.norm(temporal, ord=2, axis=2, keep_dims=False)
        # -> (batch_size * max_frames) x anchor_size
        temporal_norm = tf.reshape(temporal_norm, [-1, self.max_frames - 1, self.anchor_size])
        # -> batch_size x (max_frames - 1) x anchor_size

        # Normalize the inputs for each frame; Obtain normalized residual vectors.
        temporal = tf.nn.l2_normalize(temporal, 2)

        # Both spatial, temporal have shape (batch_size * max_frames) x anchor_size x feature_size
        spatial = tf.reshape(spatial, [-1, self.anchor_size, self.feature_size])
        temporal = tf.reshape(temporal, [-1, self.anchor_size, self.feature_size])

        ####################################################################################
        # Reduce the number of parameters with non-share CNN ###############################
        ####################################################################################
        spatial_cnn_weights = tf.get_variable("spatial_cnn_weights{}".format(""
                                                                             if self.scope_id is None
                                                                             else str(self.scope_id)),
                                              [self.anchor_size, self.kernel_size, self.feature_size],
                                              initializer=tf.random_normal_initializer(
                                                  stddev=1 / math.sqrt(self.kernel_size * self.feature_size)),
                                              dtype=tf.float32)
        temporal_cnn_weights = tf.get_variable("temporal_cnn_weights{}".format(""
                                                                               if self.scope_id is None
                                                                               else str(self.scope_id)),
                                               [self.anchor_size, self.kernel_size, self.feature_size],
                                               initializer=tf.random_normal_initializer(
                                                  stddev=1 / math.sqrt(self.kernel_size * self.feature_size)),
                                               dtype=tf.float32)

        tp_spatial_cnn_weights = tf.transpose(spatial_cnn_weights, perm=[0, 2, 1])
        tp_temporal_cnn_weights = tf.transpose(temporal_cnn_weights, perm=[0, 2, 1])

        tp_spatial = tf.transpose(spatial, perm=[1, 0, 2])
        tp_temporal = tf.transpose(temporal, perm=[1, 0, 2])

        spatial_output = tf.matmul(tp_spatial, tp_spatial_cnn_weights)
        temporal_output = tf.matmul(tp_temporal, tp_temporal_cnn_weights)

        tp_spatial_output = tf.transpose(spatial_output, perm=[1, 0, 2])
        tp_temporal_output = tf.transpose(temporal_output, perm=[1, 0, 2])
        # -> (batch_size * max_frames) x anchor_size x kernel_size

        # Apply rectified activation.
        spatial_output = tf.reshape(tp_spatial_output, [-1, self.kernel_size * self.anchor_size])
        temporal_output = tf.reshape(tp_temporal_output, [-1, self.kernel_size * self.anchor_size])
        spatial_output = tf.nn.relu(spatial_output)
        temporal_output = tf.nn.relu(temporal_output)

        spatial_output = tf.reshape(spatial_output, [-1, self.max_frames, self.kernel_size * self.anchor_size])
        temporal_output = tf.reshape(temporal_output, [-1, self.max_frames - 1, self.kernel_size * self.anchor_size])

        spatial_output = tf.concat([spatial_output, spatial_norm], 2)
        temporal_output = tf.concat([temporal_output, temporal_norm], 2)

        spatial_output = tf.reshape(spatial_output, [-1, self.kernel_size * self.anchor_size + self.anchor_size])
        temporal_output = tf.reshape(temporal_output, [-1, self.kernel_size * self.anchor_size + self.anchor_size])
        spatial_dim = spatial_output.get_shape().as_list()[1]
        spatial_vlad = loupe_modules.NetVLAD(feature_size=spatial_dim,
                                             max_samples=self.max_frames,
                                             cluster_size=256,
                                             output_dim=self.hidden_layer_size,
                                             gating=False,
                                             add_batch_norm=self.batch_norm,
                                             is_training=self.is_training)
        temporal_dim = temporal_output.get_shape().as_list()[1]
        temporal_vlad = loupe_modules.NetVLAD(feature_size=temporal_dim,
                                              max_samples=self.max_frames - 1,
                                              cluster_size=256,
                                              output_dim=self.hidden_layer_size,
                                              gating=False,
                                              add_batch_norm=self.batch_norm,
                                              is_training=self.is_training)
        with tf.variable_scope("spatial_vlad"):
            spatial_agg = spatial_vlad.forward(spatial_output)

        with tf.variable_scope('temporal_vlad'):
            temporal_agg = temporal_vlad.forward(temporal_output)

        if self.batch_norm:
            spatial_agg = slim.batch_norm(
                spatial_agg,
                center=True,
                scale=True,
                is_training=self.is_training,
                scope="spatial_activation_bn")

            temporal_agg = slim.batch_norm(
                temporal_agg,
                center=True,
                scale=True,
                is_training=self.is_training,
                scope="temporal_activation_bn")

        if self.add_relu:
            spatial_agg = tf.nn.relu(spatial_agg)
            temporal_agg = tf.nn.relu(temporal_agg)

        ####################################################################################
        # Fuse spatial & temporal features #################################################
        ####################################################################################
        spatial_temporal_concat = tf.concat([spatial_agg, temporal_agg], 1)

        sp_dim = spatial_temporal_concat.get_shape().as_list()[1]
        sp_weights = tf.get_variable("spa_temp_fusion",
                                     [sp_dim, self.output_dim],
                                     initializer=tf.random_normal_initializer(
                                         stddev=1 / math.sqrt(self.output_dim)))
        activation = tf.matmul(spatial_temporal_concat, sp_weights)

        if self.batch_norm:
            activation = slim.batch_norm(
                activation,
                center=True,
                scale=True,
                is_training=self.is_training,
                scope="activation_bn")

        if self.add_relu:
            activation = tf.nn.relu(activation)
        ####################################################################################

        return activation


class TriangulationNsCnnIndirectAttentionModule(modules.BaseModule):
    """ CNN-integrated Triangulation Embedding Module with non-sharing CNN.
    """
    def __init__(self,
                 feature_size,
                 max_frames,
                 anchor_size,
                 self_attention,
                 hidden_layer_size,
                 kernel_size,
                 output_dim,
                 add_relu,
                 batch_norm,
                 is_training,
                 scope_id=None):
        """ Initialize class TriangulationNsCnnIndirectAttentionModule.
        :param feature_size: int
        :param max_frames: max_frames x 1
        :param anchor_size: int
        :param self_attention: bool
        :param hidden_layer_size: int
        :param kernel_size: int
        :param output_dim: int
        :param add_relu: bool
        :param batch_norm: bool
        :param is_training: bool
        :param scope_id: Object
        """
        self.feature_size = feature_size
        self.max_frames = max_frames
        self.anchor_size = anchor_size
        self.self_attention = self_attention
        self.hidden_layer_size = hidden_layer_size
        self.kernel_size = kernel_size
        self.output_dim = output_dim
        self.add_relu = add_relu
        self.batch_norm = batch_norm
        self.is_training = is_training
        self.scope_id = scope_id

    def forward(self, inputs, **unused_params):
        """ Forward method for TriangulationNsCnnIndirectAttentionModule.
        :param inputs: (batch_size * max_frames) x feature_size
        :return: batch_size x output_dim
        """
        ####################################################################################
        # Get spatial features with t-embedding ############################################
        ####################################################################################
        anchor_weights = tf.get_variable("anchor_weights{}".format("" if self.scope_id is None else str(self.scope_id)),
                                         [self.feature_size, self.anchor_size],
                                         initializer=tf.orthogonal_initializer(),
                                         dtype=tf.float32)
        tf.summary.histogram("anchor_weights{}".format("" if self.scope_id is None else str(self.scope_id)),
                             anchor_weights)

        # Transpose weights for proper subtraction.
        anchor_weights = tf.transpose(anchor_weights)
        anchor_weights = tf.reshape(anchor_weights, [1, self.feature_size * self.anchor_size])

        # Tile inputs to subtract them with all anchors.
        tiled_inputs = tf.tile(inputs, [1, self.anchor_size])
        # -> (batch_size * max_frames) x (feature_size * anchor_size)
        spatial = tf.subtract(tiled_inputs, anchor_weights)
        # -> (batch_size * max_frames) x (feature_size * anchor_size)

        spatial = tf.reshape(spatial, [-1, self.anchor_size, self.feature_size])
        # Normalize the inputs for each frame; Obtain normalized residual vectors.
        spatial = tf.nn.l2_normalize(spatial, 2)
        spatial = tf.reshape(spatial, [-1, self.feature_size * self.anchor_size])
        # -> (batch_size * max_frames) x (feature_size * anchor_size)
        ####################################################################################

        ####################################################################################
        # Get temporal features from frame-difference in t-embedded feature. ###############
        ####################################################################################
        cloned_spatial = tf.identity(spatial)
        # Shift the input to the right (to subtract frame T-1 from frame T):
        cloned_spatial = tf.manip.roll(cloned_spatial, shift=1, axis=1)
        temporal = tf.subtract(spatial, cloned_spatial)
        temporal = tf.reshape(temporal, [-1, self.max_frames, self.feature_size * self.anchor_size])
        # Eliminate the first row.
        stacks = tf.unstack(temporal, axis=1)
        del stacks[0]
        temporal = tf.stack(stacks, 1)
        temporal = tf.reshape(temporal, [-1, self.feature_size * self.anchor_size])

        # Both spatial, temporal have shape (batch_size * max_frames) x anchor_size x feature_size
        spatial = tf.reshape(spatial, [-1, self.anchor_size, self.feature_size])
        temporal = tf.reshape(temporal, [-1, self.anchor_size, self.feature_size])

        ####################################################################################
        # Calculate weights for soft attention mechanism ###################################
        ####################################################################################
        spatial_attention = tf.matmul(spatial, tf.transpose(spatial, perm=[0, 2, 1]))
        temporal_attention = tf.matmul(temporal, tf.transpose(temporal, perm=[0, 2, 1]))

        spatial_attention = tf.expand_dims(spatial_attention, -1)
        temporal_attention = tf.expand_dims(temporal_attention, -1)

        # Zero-out negative weight.
        spatial_attention = tf.nn.relu(spatial_attention)
        temporal_attention = tf.nn.relu(temporal_attention)

        spatial_attention = tf.reduce_sum(spatial_attention, axis=2)
        temporal_attention = tf.reduce_sum(temporal_attention, axis=2)
        # -> batch_size x max_frames x 1

        spatial_attention_weight = tf.nn.softmax(spatial_attention, axis=1)
        temporal_attention_weight = tf.nn.softmax(temporal_attention, axis=1)
        ####################################################################################

        ####################################################################################
        # Reduce the number of parameters with non-share CNN ###############################
        ####################################################################################
        spatial_cnn_weights = tf.get_variable("spatial_cnn_weights{}".format(""
                                                                             if self.scope_id is None
                                                                             else str(self.scope_id)),
                                              [self.anchor_size, self.kernel_size, self.feature_size],
                                              initializer=tf.random_normal_initializer(
                                                  stddev=1 / math.sqrt(self.kernel_size * self.feature_size)),
                                              dtype=tf.float32)
        temporal_cnn_weights = tf.get_variable("temporal_cnn_weights{}".format(""
                                                                               if self.scope_id is None
                                                                               else str(self.scope_id)),
                                               [self.anchor_size, self.kernel_size, self.feature_size],
                                               initializer=tf.random_normal_initializer(
                                                  stddev=1 / math.sqrt(self.kernel_size * self.feature_size)),
                                               dtype=tf.float32)

        tp_spatial_cnn_weights = tf.transpose(spatial_cnn_weights, perm=[0, 2, 1])
        tp_temporal_cnn_weights = tf.transpose(temporal_cnn_weights, perm=[0, 2, 1])


        tp_spatial = tf.transpose(spatial, perm=[1, 0, 2])
        tp_temporal = tf.transpose(temporal, perm=[1, 0, 2])


        spatial_output = tf.matmul(tp_spatial, tp_spatial_cnn_weights)
        temporal_output = tf.matmul(tp_temporal, tp_temporal_cnn_weights)


        tp_spatial_output = tf.transpose(spatial_output, perm=[1, 0, 2])
        tp_temporal_output = tf.transpose(temporal_output, perm=[1, 0, 2])


        spatial_output = tf.reshape(tp_spatial_output, [-1, self.max_frames, self.kernel_size * self.anchor_size])
        temporal_output = tf.reshape(tp_temporal_output, [-1, self.max_frames - 1, self.kernel_size * self.anchor_size])


        if self.self_attention:
            spatial_mean = tf.reduce_mean(tf.multiply(spatial_output, spatial_attention_weight), 1)
            temporal_mean = tf.reduce_mean(tf.multiply(temporal_output, temporal_attention_weight), 1)
        else:
            spatial_mean = tf.reduce_mean(spatial_output, 1)
            temporal_mean = tf.reduce_mean(temporal_output, 1)

        spatial_variance = module_utils.reduce_var(spatial_output, 1)
        temporal_variance = module_utils.reduce_var(temporal_output, 1)

        spatial_pool = tf.concat([spatial_mean, spatial_variance], 1)
        temporal_pool = tf.concat([temporal_mean, temporal_variance], 1)

        if self.batch_norm:
            spatial_pool = slim.batch_norm(
                spatial_pool,
                center=True,
                scale=True,
                is_training=self.is_training,
                scope="spatial_pool_bn")
            temporal_pool = slim.batch_norm(
                temporal_pool,
                center=True,
                scale=True,
                is_training=self.is_training,
                scope="temporal_pool_bn")

        spatial_dim = spatial_pool.get_shape().as_list()[1]
        spatial_weights = tf.get_variable("spatial_hidden",
                                          [spatial_dim, self.hidden_layer_size],
                                          initializer=tf.random_normal_initializer(
                                              stddev=1 / math.sqrt(self.hidden_layer_size)))

        temporal_dim = temporal_pool.get_shape().as_list()[1]
        temporal_weights = tf.get_variable("temporal_hidden",
                                           [temporal_dim, self.hidden_layer_size],
                                           initializer=tf.random_normal_initializer(
                                              stddev=1 / math.sqrt(self.hidden_layer_size)))

        spatial_activation = tf.matmul(spatial_pool, spatial_weights)
        temporal_activation = tf.matmul(temporal_pool, temporal_weights)

        if self.batch_norm:
            spatial_activation = slim.batch_norm(
                spatial_activation,
                center=True,
                scale=True,
                is_training=self.is_training,
                scope="spatial_activation_bn")

            temporal_activation = slim.batch_norm(
                temporal_activation,
                center=True,
                scale=True,
                is_training=self.is_training,
                scope="temporal_activation_bn")

        if self.add_relu:
            spatial_activation = tf.nn.relu(spatial_activation)
            temporal_activation = tf.nn.relu(temporal_activation)

        ####################################################################################
        # Fuse spatial & temporal features #################################################
        ####################################################################################
        spatial_temporal_concat = tf.concat([spatial_activation, temporal_activation], 1)

        sp_dim = spatial_temporal_concat.get_shape().as_list()[1]
        sp_weights = tf.get_variable("spa_temp_fusion",
                                     [sp_dim, self.output_dim],
                                     initializer=tf.random_normal_initializer(
                                         stddev=1 / math.sqrt(self.output_dim)))
        activation = tf.matmul(spatial_temporal_concat, sp_weights)

        if self.batch_norm:
            activation = slim.batch_norm(
                activation,
                center=True,
                scale=True,
                is_training=self.is_training,
                scope="activation_bn")

        if self.add_relu:
            activation = tf.nn.relu(activation)
        ####################################################################################

        return activation


class TriangulationCnnModule(modules.BaseModule):
    """ Compute CNN after triangulation embedding.
    """
    def __init__(self,
                 feature_size,
                 max_frames,
                 num_filters,
                 anchor_size,
                 batch_norm,
                 is_training,
                 scope_id=None):
        """ Initialize CNN module after T-Emb.
        :param feature_size: int
        :param max_frames: max_frame x 1
        :param num_filters: int
        :param batch_norm: bool
        :param is_training: bool
        :param scope_id: Object
        """
        self.feature_size = feature_size
        self.max_frames = max_frames
        self.batch_norm = batch_norm
        self.num_filters = num_filters
        self.anchor_size = anchor_size
        self.is_training = is_training
        self.scope_id = scope_id

    def forward(self, inputs, **unused_params):
        """ Forward method for TriangulationCnnModule.
        :param inputs: (batch_size * max_frames) x (feature_size * anchor_size)
        :return: batch_size x max_frames x (anchor_size x num_filters)
        """
        # -> (batch_size * max_frames) x feature_size x anchor_size
        cnn_weights = tf.get_variable("cnn_weights",
                                      [self.anchor_size, self.num_filters, self.feature_size],
                                      initializer=tf.random_normal_initializer(
                                                 stddev=1 / math.sqrt(self.num_filters * self.feature_size)))

        cnn_weights = tf.transpose(cnn_weights, perm=[0, 2, 1])
        # -> anchor_size x feature_size x num_filters

        reshaped_inputs = tf.reshape(inputs, [-1, self.anchor_size, self.feature_size])
        reshaped_inputs = tf.transpose(reshaped_inputs, perm=[1, 0, 2])
        output = tf.matmul(reshaped_inputs, cnn_weights)
        output = tf.transpose(output, perm=[1, 0, 2])
        output = tf.reshape(output, [-1, self.max_frames, self.anchor_size * self.num_filters])
        # -> batch_size x max_frames x (anchor * num_filters)
        return output


class WeightedTriangulationEmbedding(modules.BaseModule):
    """ Weighted Triangulation embedding for each frame.

    References:
    Triangulation embedding and democratic aggregation for image search.
    """
    def __init__(self,
                 feature_size,
                 max_frames,
                 anchor_size,
                 batch_norm,
                 is_training,
                 scope_id=None):
        self.feature_size = feature_size
        self.max_frames = max_frames
        self.batch_norm = batch_norm
        self.anchor_size = anchor_size
        self.is_training = is_training
        self.det_reg = True
        self.det_reg_lambda = 1e-5
        self.scope_id = scope_id

    def forward(self, inputs, **unused_params):
        """ Forward method for Triangulation Embedding.
        :param inputs: (batch_size * max_frames) x feature_size
        :return: (batch_size * max_frames) x (feature_size * anchor_size)
        """
        # Anchor weights calculation.
        anchor_weights = tf.get_variable("anchor_weights{}".format("" if self.scope_id is None else str(self.scope_id)),
                                         [self.feature_size, self.anchor_size],
                                         initializer=tf.random_normal_initializer(
                                              stddev=1 / math.sqrt(self.anchor_size)))
        tf.summary.histogram("anchor_weights{}".format("" if self.scope_id is None
                                                       else str(self.scope_id)), anchor_weights)

        # Calculate orthogonal regularization.
        if self.det_reg:
            anchor_weights_t = tf.transpose(anchor_weights)
            det_reg = tf.matmul(anchor_weights_t, anchor_weights)
            identity = tf.identity(det_reg)
            det_reg = tf.subtract(det_reg, identity)
            det_reg = tf.reduce_sum(tf.abs(det_reg))
            det_reg = det_reg * self.det_reg_lambda
        else:
            det_reg = None

        # Transpose weights for proper subtraction; See investigation report.
        anchor_weights = tf.transpose(anchor_weights)
        anchor_weights = tf.reshape(anchor_weights, [1, self.feature_size * self.anchor_size])

        # Tile inputs to subtract them with all anchors.
        tiled_inputs = tf.tile(inputs, [1, self.anchor_size])
        # -> (batch_size * max_frames) x (feature_size * anchor_size)
        t_emb = tf.subtract(tiled_inputs, anchor_weights)
        # -> (batch_size * max_frames) x (feature_size * anchor_size)

        t_emb = tf.reshape(t_emb, [-1, self.anchor_size, self.feature_size])
        # Normalize the inputs for each frame.
        t_emb = tf.nn.l2_normalize(t_emb, 2)
        t_emb = tf.reshape(t_emb, [-1, self.feature_size * self.anchor_size])
        t_emb = tf.nn.l2_normalize(t_emb, 1)
        # -> (batch_size * max_frames) x (feature_size * anchor_size)
        t_emb = tf.reshape(t_emb, [-1, self.max_frames, self.feature_size * self.anchor_size])
        # t_emb = tf.multiply(assignment_activation, t_emb)
        return t_emb, det_reg


class TriangulationTemporalEmbedding(modules.BaseModule):
    """ Triangulation embedding which calculates the difference in between frames. """
    def __init__(self,
                 feature_size,
                 max_frames,
                 anchor_size,
                 batch_norm,
                 is_training,
                 scope_id=None):
        self.feature_size = feature_size
        self.max_frames = max_frames
        self.batch_norm = batch_norm
        self.anchor_size = anchor_size
        self.is_training = is_training
        self.scope_id = scope_id

    def forward(self, inputs, **unused_params):
        """ Forward method for TriangulationTemporalEmbedding.
        :param inputs: (batch_size * max_frames) x (feature_size * anchor_size)
        :return: batch_size x (max_frames - 1) x (feature_size * anchor_size)
        """
        cloned_inputs = tf.identity(inputs)
        # Shift the input to the right (to subtract frame T-1 from frame T):
        cloned_inputs = tf.manip.roll(cloned_inputs, shift=1, axis=1)
        temp_info = tf.subtract(inputs, cloned_inputs)

        temp_info_reshaped = tf.reshape(temp_info, [-1, self.anchor_size, self.feature_size])
        # No normalization
        temp_info_reshaped = tf.nn.l2_normalize(temp_info_reshaped, 2)
        temp_info = tf.reshape(temp_info_reshaped, [-1, self.max_frames,
                                                    self.feature_size * self.anchor_size])

        stacks = tf.unstack(temp_info, axis=1)
        del stacks[0]
        temp_info = tf.stack(stacks, 1)
        return temp_info


###############################################################################
# NetVLAD Prototype ###########################################################
###############################################################################
class NetVladOrthoReg(modules.BaseModule):
    """ NetVLAD from WILLOW's model with orthogonal regularization. """
    def __init__(self, feature_size, max_frames, cluster_size, batch_norm, is_training,
                 det_reg=None, scope_id=None):
        """ Initialize NetVLAD with orthogonal regularization.
        :param feature_size: int
        :param max_frames: max_frames x 1
        :param cluster_size: int
        :param batch_norm: bool
        :param is_training: bool
        :param det_reg: float
        :param scope_id: Object
        """
        self.feature_size = feature_size
        self.max_frames = max_frames
        self.is_training = is_training
        self.batch_norm = batch_norm
        self.cluster_size = cluster_size
        self.det_reg = det_reg
        self.scope_id = scope_id

    def forward(self, inputs, **unused_params):
        """ Forward method for NetVladOrthoReg.
        :param inputs: (batch_size * max_frames) x feature_size
        :return: (batch_size * max_frames) x (feature_size * cluster_size)
        """
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

        if self.det_reg is None:
            cluster_weights2 = tf.get_variable("cluster_weights2",
                                               [self.feature_size, self.cluster_size],
                                               initializer=tf.random_normal_initializer(
                                                   stddev=1 / math.sqrt(self.feature_size)))
        else:
            cluster_weights2 = tf.get_variable("cluster_weights2",
                                               [self.feature_size, self.cluster_size],
                                               initializer=tf.random_normal_initializer(
                                                   stddev=1 / math.sqrt(self.feature_size)),
                                               regularizer=module_utils.orthogonal_regularizer(self.det_reg,
                                                                                               self.scope_id))

        cluster_weights2 = tf.expand_dims(cluster_weights2, axis=0)
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
