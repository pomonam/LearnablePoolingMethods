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
import math
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
                                              stddev=1 / math.sqrt(self.anchor_size)))
        tf.summary.histogram("anchor_weights{}".format("" if self.scope_id is None else str(self.scope_id)),
                             anchor_weights)
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
        # -> (batch_size * max_frames) x (feature_size * cluster_size)

        return t_emb

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
        # Assignment soft-max weights calculation.
        # assignment_weights = tf.get_variable("assignment_weights".format("" if self.scope_id is None
        #                                                                  else str(self.scope_id)),
        #                                      [self.feature_size, self.anchor_size],
        #                                      initializer=tf.random_normal_initializer(
        #                                          stddev=1 / math.sqrt(self.anchor_size)))
        # tf.summary.histogram("assignment_weights{}".format("" if self.scope_id is None
        #                                                    else str(self.scope_id)),
        #                      assignment_weights)
        #
        # assignment_activation = tf.matmul(inputs, assignment_weights)
        #
        # if self.batch_norm:
        #     assignment_activation = slim.batch_norm(
        #         assignment_weights,
        #         center=True,
        #         scale=True,
        #         is_training=self.is_training,
        #         scope="assignment_bn")
        # else:
        #     assignment_bias = tf.get_variable("assignment_bias{}".format("" if self.scope_id is None
        #                                                                  else str(self.scope_id)),
        #                                      [self.anchor_size],
        #                                       initializer=tf.random_normal_initializer(
        #                                          stddev=1 / math.sqrt(self.anchor_size)))
        #     tf.summary.histogram("assignment_bias{}".format("" if self.scope_id is None
        #                                                     else str(self.scope_id)), assignment_bias)
        #     assignment_activation += assignment_bias
        #
        # assignment_activation = tf.nn.sigmoid(assignment_activation)
        # tf.summary.histogram("assignment_activation", assignment_activation)
        # # -> (batch_size * max_frames) x anchor_size
        #
        # assignment_activation = tf.reshape(assignment_activation, [-1, self.max_frames, self.anchor_size])
        # assignment_activation = tf.reduce_sum(assignment_activation, 2)
        # # -> batch_size x max_frames

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
        :param inputs: batch_size x max_frames x (feature_size * anchor_size)
        :return: batch_size x (max_frames -1) x (feature_size * anchor_size)
        """
        cloned_inputs = tf.identity(inputs)
        # Shift the input to the right.
        cloned_inputs = tf.manip.roll(cloned_inputs, shift=1, axis=1)
        temp_info = tf.subtract(inputs, cloned_inputs)

        temp_info_reshaped = tf.reshape(temp_info, [-1, self.anchor_size, self.feature_size])
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
