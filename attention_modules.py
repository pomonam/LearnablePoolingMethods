# Copyright 2018 Deep Topology All Rights Reserved.
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

from tensorflow.python.ops import nn
import tensorflow as tf
import tensorflow.contrib.slim as slim
import math
import modules


class OneFcAttention(modules.BaseModule):
    def __init__(self, num_features, num_frames, num_cluster, do_shift=True):
        self.num_feature = num_features
        self.num_frames = num_frames
        self.num_cluster = num_cluster
        self.do_shift = do_shift

    def forward(self, inputs, **unused_params):
        attention_weights = \
            tf.get_variable("one_fc_attention_weight",
                            [self.num_feature, self.num_cluster],
                            initializer=tf.contrib.layers.xavier_initializer())
        attention = tf.matmul(inputs, attention_weights)
        attention = tf.reshape(attention, [-1, self.num_frames, self.num_cluster])
        attention = tf.scalar_mul(1 / math.sqrt(self.num_feature), attention)
        attention = tf.nn.softmax(attention, dim=1)

        reshaped_inputs = tf.reshape(inputs, [-1, self.num_frames, self.num_feature])
        activation = tf.transpose(attention, perm=[0, 2, 1])
        activation = tf.matmul(activation, reshaped_inputs)
        # -> batch_size x num_cluster x feature_size

        reshaped_activation = tf.reshape(activation, [-1, self.num_feature])

        if self.do_shift:
            alpha = \
                tf.get_variable("alpha",
                                [1],
                                initializer=tf.constant_initializer(1))
            beta = \
                tf.get_variable("beta",
                                [1],
                                initializer=tf.constant_initializer(0.01))

            reshaped_activation = alpha * reshaped_activation
            reshaped_activation = reshaped_activation + beta
            reshaped_activation = tf.nn.l2_normalize(reshaped_activation, 1)
            reshaped_activation = tf.scalar_mul(1 / math.sqrt(self.num_cluster), reshaped_activation)

        activation = tf.reshape(reshaped_activation, [-1, self.num_cluster * self.num_feature])

        return activation


class MultiHeadAttention(modules.BaseModule):
    def __init__(self, num_heads, num_units, max_frames, block_id):
        """

        :param num_heads: Number of self-attention modules
        :param num_units: last dimension of Q, K, V
        """
        self.num_heads = num_heads
        self.num_units = num_units
        self.max_frames = max_frames
        self.block_id = block_id

    def self_attention(self, inputs, scope_id):
        """

        :param Q: batch_size x max_frames x num_units
        :param K: batch_size x max_frames x num_units
        :param V: batch_size x max_frames x num_units
        :return:
        """
        with tf.variable_scope("Block{}Layer{}".format(self.block_id, scope_id), reuse=tf.AUTO_REUSE):
            # Calculate query, key, value pair
            Q = tf.layers.dense(inputs, self.num_units, activation=tf.nn.relu)
            K = tf.layers.dense(inputs, self.num_units, activation=tf.nn.relu)
            V = tf.layers.dense(inputs, self.num_units, activation=tf.nn.relu)
            # Q, K, V: -> (batch_size * max_frames) x num_units

            # Reshape for self-attention calculation
            Q = tf.reshape(Q, [-1, self.max_frames, self.num_units])
            K = tf.reshape(K, [-1, self.max_frames, self.num_units])
            V = tf.reshape(V, [-1, self.max_frames, self.num_units])
            # Q, K, V: -> batch_size x max_frames x num_units

            # Self-attention
            attention = tf.matmul(Q, tf.transpose(K, perm=[0, 2, 1]))
            # attention: -> batch_size x max_frames x max_frames
            float_cpy = tf.cast(self.num_units, dtype=tf.float32)
            attention = tf.nn.softmax(tf.divide(attention, tf.sqrt(float_cpy)))

            output = tf.matmul(attention, V)
            # output: -> batch_size x max_frames x num_units
            return output

    def forward(self, inputs, **unused_params):
        result = self.self_attention(inputs, scope_id=0)
        for i in range(1, self.num_heads):
            result = tf.identity(result)
            output = self.self_attention(inputs, scope_id=i)
            result = tf.concat([result, output], 2)
        # result: -> batch_size x max_frames x (num_units * num_heads)
        return result


class TransformerEncoderBlock(modules.BaseModule):
    def __init__(self, is_training, num_units, max_frames, feature_size, num_heads, block_id):
        """

        :param is_training:
        :param num_units: Number of hidden units of fully connected layers
        """
        self.is_training = is_training
        self.num_units = num_units
        self.max_frames = max_frames
        self.feature_size = feature_size
        self.num_heads = num_heads
        self.block_id = block_id

    def forward(self, inputs, **unused_params):
        """
        One block of encoder containing one self-attention layer and one fully connected layer.
        :param inputs: (batch_size * max_frames) x feature_size
        :param unused_params:
        :return:
        """
        multi_head_layer = MultiHeadAttention(self.num_heads, self.num_units, self.max_frames, self.block_id)

        attention_output = multi_head_layer.forward(inputs)
        # output: -> batch_size x max_frames x (num_units * num_heads)

        attention_output = tf.reshape(attention_output, [-1, self.num_units * self.num_heads])
        # output: -> (batch_size * max_frames) x (num_units * num_heads)

        attention_output = tf.layers.dense(attention_output, self.feature_size, activation=tf.nn.relu)
        # output: -> (batch_size * max_frames) x feature_size

        # Residual connection & Layer normalization
        attention_output += inputs
        attention_output = tf.contrib.layers.layer_norm(attention_output)

        # 2 layers of 1 x 1 convolution
        output = tf.reshape(attention_output, [-1, self.max_frames, self.feature_size])
        output = tf.layers.conv1d(output, filters=4 * self.num_units, kernel_size=1, activation=tf.nn.relu,
                                  use_bias=True)
        output = tf.layers.conv1d(output, filters=self.num_units, kernel_size=1, activation=None, use_bias=True)

        # Residual connection & Layer normalization
        output = tf.contrib.layers.layer_norm(output)
        output = tf.reshape(output, [-1, self.feature_size])

        return output


class PnGateModule(modules.BaseModule):
    def __init__(self, vocab_size, is_training, scope_id=None):
        """ Initialize class PnGateModule.
        :param vocab_size: int
            Size of the classes.
        :param is_training: bool
            True iff the model is being trained.
        :param scope_id: Object
        """
        self.vocab_size = vocab_size
        self.scope_id = scope_id
        self.is_training = is_training

    def forward(self, inputs, **unused_params):
        """ PN Gate for correlation learning.
        vocabularies -> P gate -> N gate -> output
        :param inputs: batch_size x vocab_size
        :return: batch_size x vocab_size
        """
        p_gating_weights = \
            tf.get_variable("p_pn_gate",
                            [self.vocab_size, self.vocab_size],
                            initializer=tf.random_normal_initializer(stddev=1 / math.sqrt(self.vocab_size)))
        n_gating_weights = \
            tf.get_variable("n_pn_gate",
                            [self.vocab_size, self.vocab_size],
                            initializer=tf.random_normal_initializer(stddev=1 / math.sqrt(self.vocab_size)))

        # batch_size x vocab_size, vocab_size x vocab_size --> batch_size x vocab_size
        p_activation = tf.matmul(inputs, p_gating_weights)
        p_activation = tf.nn.relu6(p_activation)
        p_gate = inputs + p_activation

        # batch_size x vocab_size, vocab_size x vocab_size --> batch_size x vocab_size
        n_activation = tf.matmul(p_gate, n_gating_weights)
        n_activation = -1 * n_activation
        n_activation = tf.nn.relu6(n_activation)
        n_gate = p_gate + (-1 * n_activation)

        output = tf.nn.softmax(n_gate)
        return output


class NpGateModule(modules.BaseModule):
    def __init__(self, vocab_size, is_training, scope_id=None):
        """ Initialize class NpGateModule.
        :param vocab_size: int
            Size of the classes.
        :param is_training: bool
            True iff the model is being trained.
        :param scope_id: Object
        """
        self.vocab_size = vocab_size
        self.scope_id = scope_id
        self.is_training = is_training

    def forward(self, inputs, **unused_params):
        """ PN Gate for correlation learning.
        vocabularies -> N gate -> P gate -> output
        :param inputs: batch_size x vocab_size
        :return: batch_size x vocab_size
        """
        p_gating_weights = \
            tf.get_variable("p_np_gate",
                            [self.vocab_size, self.vocab_size],
                            initializer=tf.random_normal_initializer(stddev=1 / math.sqrt(self.vocab_size)))
        n_gating_weights = \
            tf.get_variable("n_np_gate",
                            [self.vocab_size, self.vocab_size],
                            initializer=tf.random_normal_initializer(stddev=1 / math.sqrt(self.vocab_size)))

        # batch_size x vocab_size, vocab_size x vocab_size --> batch_size x vocab_size
        n_activation = tf.matmul(inputs, n_gating_weights)
        n_activation = -1 * n_activation
        n_activation = tf.nn.relu6(n_activation)
        n_gate = inputs + (-1 * n_activation)

        # batch_size x vocab_size, vocab_size x vocab_size --> batch_size x vocab_size
        p_activation = tf.matmul(n_gate, p_gating_weights)
        p_activation = tf.nn.relu6(p_activation)
        p_gate = n_gate + p_activation
        output = tf.nn.softmax(p_gate)

        return output


class PGateModule(modules.BaseModule):
    def __init__(self, vocab_size, is_training, scope_id=None):
        """ Initialize class PGateModule.
        :param vocab_size: int
            Size of the classes.
        :param is_training: bool
            True iff the model is being trained.
        :param scope_id: Object
        """
        self.vocab_size = vocab_size
        self.scope_id = scope_id
        self.is_training = is_training

    def forward(self, inputs, **unused_params):
        """ PN Gate for correlation learning.
        vocabularies -> P gate -> output
        :param inputs: batch_size x vocab_size
        :return: batch_size x vocab_size
        """
        p_gating_weights = \
            tf.get_variable("p_p_gate",
                            [self.vocab_size, self.vocab_size],
                            initializer=tf.random_normal_initializer(stddev=1 / math.sqrt(self.vocab_size)))

        # batch_size x vocab_size, vocab_size x vocab_size --> batch_size x vocab_size
        p_activation = tf.matmul(inputs, p_gating_weights)
        p_activation = tf.nn.relu6(p_activation)
        p_gate = inputs + p_activation
        output = tf.nn.softmax(p_gate)

        return output


class CorNNGateModule(modules.BaseModule):
    def __init__(self, vocab_size, is_training, batch_norm=True, scope_id=None):
        """ Initialize a class CorNNGateModule.
        :param vocab_size: int
            Size of the classes.
        :param is_training: bool
        :param batch_norm: bool
        :param scope_id: int
        """
        self.vocab_size = vocab_size
        self.is_training = is_training
        self.batch_norm = batch_norm
        self.scope_id = scope_id

    def forward(self, inputs, **unused_params):
        """ Forward function of CorNNGateModule.
        :param inputs: batch_size x vocab_size
        :return: batch_size x vocab_size
        """
        fc1_out = slim.fully_connected(
            inputs=inputs,
            num_outputs=self.vocab_size,
            activation_fn=nn.relu,
            scope="vocab_gate1_v1{}".format("" if self.scope_id is None else str(self.scope_id))
        )

        fc2_out = slim.fully_connected(
            inputs=fc1_out,
            num_outputs=self.vocab_size,
            activation_fn=nn.relu,
            scope="vocab_gate2_v1{}".format("" if self.scope_id is None else str(self.scope_id))
        )

        fc3_out = slim.fully_connected(
            inputs=fc2_out,
            num_outputs=self.vocab_size,
            activation_fn=nn.sigmoid,
            scope="vocab_gate3_v1{}".format("" if self.scope_id is None else str(self.scope_id))
        )

        return fc3_out


class ContextGateV1(modules.BaseModule):
    """
    Given the weight W, calculate sigmoid(WX + b) o X. o is an element-wise
    multiplication.
    """
    def __init__(self, vocab_size, is_training, batch_norm=True, scope_id=None):
        """ Initialize a class ContextGateV1. The idea and implementation is adopted from WILLOW.
        :param vocab_size: int
            Size of the classes.
        :param is_training: bool
        :param batch_norm: bool
        :param scope_id: int
        """
        self.vocab_size = vocab_size
        self.is_training = is_training
        self.batch_norm = batch_norm
        self.scope_id = scope_id

    def forward(self, inputs, **unused_params):
        """ Forward function of ContextGateV1
        :param inputs: batch_size x vocab_size
        :return: batch_size x vocab_size
        """
        gating_weights = tf.get_variable("vocab_gate_v1{}".format("" if self.scope_id is None else str(self.scope_id)),
                                         [self.vocab_size, self.vocab_size])

        # batch_size x vocab_size, vocab_size x vocab_size --> batch_size x vocab_size
        gates = tf.matmul(inputs, gating_weights)

        if self.batch_norm:
            gates = slim.batch_norm(
                gates,
                center=True,
                scale=True,
                is_training=self.is_training,
                scope="vocab_gate_bn_v1{}".format("" if self.scope_id is None else str(self.scope_id)))

        gates = tf.sigmoid(gates)

        # batch_size x vocab_size, batch_size x vocab_size -> batch_size x vocab_size
        updated_inputs = tf.multiply(inputs, gates)

        # batch_size x vocab_size
        return updated_inputs
