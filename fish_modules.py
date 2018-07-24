import tensorflow.contrib.slim as slim
import tensorflow as tf
import modules
import math


class LuckyFishModule(modules.BaseModule):
    """ Attention cluster. """
    def __init__(self, feature_size, max_frames, cluster_size, add_batch_norm, shift_operation, is_training):
        self.feature_size = feature_size
        self.max_frames = max_frames
        self.is_training = is_training
        self.add_batch_norm = add_batch_norm
        self.shift_operation = shift_operation
        self.cluster_size = cluster_size

    def forward(self, inputs, **unused_params):
        reshaped_inputs = tf.reshape(inputs, [-1, self.feature_size])
        inputs = tf.reshape(inputs, [-1, self.max_frames, self.feature_size])
        attention_weights = tf.layers.dense(reshaped_inputs, self.cluster_size, use_bias=False, activation=None)
        # -> (batch_size * max_frames) x cluster_size
        attention_weights = tf.layers.batch_normalization(attention_weights, training=self.is_training)
        attention_weights = tf.nn.softmax(attention_weights)

        reshaped_attention = tf.reshape(attention_weights, [-1, self.max_frames, self.cluster_size])
        transposed_attention = tf.transpose(reshaped_attention, perm=[0, 2, 1])
        # -> transposed_attention: batch_size x cluster_size x max_frames
        activation = tf.matmul(transposed_attention, inputs)
        # -> activation: batch_size x cluster_size x feature_size

        if self.shift_operation:
            alpha = tf.get_variable("alpha",
                                    [1, self.cluster_size, 1],
                                    initializer=tf.constant_initializer(1.0))
            beta = tf.get_variable("beta",
                                   [1, self.cluster_size, 1],
                                   initializer=tf.constant_initializer(0.0))
            activation = tf.multiply(activation, alpha)
            activation = tf.add(activation, beta)
            float_cpy = tf.cast(self.cluster_size, dtype=tf.float32)
            activation = tf.divide(activation, tf.sqrt(float_cpy))

        normalized_activation = tf.nn.l2_normalize(activation, 2)
        reshaped_normalized_activation = tf.reshape(normalized_activation, [-1, self.cluster_size * self.feature_size])
        final_activation = tf.nn.l2_normalize(reshaped_normalized_activation)
        reshaped_final_activation = tf.reshape(final_activation, [-1, self.cluster_size, self.feature_size])

        return reshaped_final_activation


class BadFishModule(modules.BaseModule):
    """ Attention cluster. """
    def __init__(self, feature_size, max_frames, cluster_size, add_batch_norm, shift_operation, is_training):
        self.feature_size = feature_size
        self.max_frames = max_frames
        self.is_training = is_training
        self.add_batch_norm = add_batch_norm
        self.shift_operation = shift_operation
        self.cluster_size = cluster_size

    def forward(self, inputs, **unused_params):
        reshaped_inputs = tf.reshape(inputs, [-1, self.feature_size])
        inputs = tf.reshape(inputs, [-1, self.max_frames, self.feature_size])

        attention_weights = tf.matmul(inputs, tf.transpose(inputs, perm=[0, 2, 1]))
        attention_weights = tf.layers.batch_normalization(attention_weights, training=self.is_training)
        reshaped_attention = tf.reshape(attention_weights, [-1, self.max_frames])
        attention_weights = tf.layers.dense(reshaped_attention, self.cluster_size, use_bias=False,
                                            activation=None)
        attention_weights = tf.layers.batch_normalization(attention_weights, training=self.is_training)
        attention_weights = tf.nn.softmax(attention_weights)

        reshaped_attention = tf.reshape(attention_weights, [-1, self.max_frames, self.cluster_size])
        transposed_attention = tf.transpose(reshaped_attention, perm=[0, 2, 1])
        # -> transposed_attention: batch_size x cluster_size x max_frames
        activation = tf.matmul(transposed_attention, inputs)
        # -> activation: batch_size x cluster_size x feature_size

        if self.shift_operation:
            alpha = tf.get_variable("alpha",
                                    [1, self.cluster_size, 1],
                                    initializer=tf.constant_initializer(1.0))
            beta = tf.get_variable("beta",
                                   [1, self.cluster_size, 1],
                                   initializer=tf.constant_initializer(0.0))
            activation = tf.multiply(activation, alpha)
            activation = tf.add(activation, beta)
            float_cpy = tf.cast(self.cluster_size, dtype=tf.float32)
            activation = tf.divide(activation, tf.sqrt(float_cpy))

        normalized_activation = tf.nn.l2_normalize(activation, 2)
        reshaped_normalized_activation = tf.reshape(normalized_activation, [-1, self.cluster_size * self.feature_size])
        final_activation = tf.nn.l2_normalize(reshaped_normalized_activation)
        reshaped_final_activation = tf.reshape(final_activation, [-1, self.cluster_size, self.feature_size])

        return reshaped_final_activation


class FishMultiHead(modules.BaseModule):
    def __init__(self, feature_size, filter_size, num_units, num_heads, cluster_size, is_training):
        self.feature_size = feature_size
        self.filter_size = filter_size
        self.num_units = num_units
        self.num_heads = num_heads
        self.cluster_size = cluster_size
        self.is_training = is_training

    def self_attention(self, inputs, head_id):
        with tf.variable_scope("head{}".format(head_id)):
            reshaped_inputs = tf.reshape(inputs, [-1, self.feature_size])
            q = tf.layers.dense(reshaped_inputs, self.num_units, use_bias=False, activation=None)
            k = tf.layers.dense(reshaped_inputs, self.num_units, use_bias=False, activation=None)
            v = tf.layers.dense(reshaped_inputs, self.num_units, use_bias=False, activation=None)

            q = tf.reshape(q, [-1, self.cluster_size, self.num_units])
            k = tf.reshape(k, [-1, self.cluster_size, self.num_units])
            v = tf.reshape(v, [-1, self.cluster_size, self.num_units])

            attention = tf.matmul(q, tf.transpose(k, perm=[0, 2, 1]))
            # -> batch_size x max_frames x max_frames
            attention = tf.layers.batch_normalization(attention, training=self.is_training)
            attention = tf.nn.softmax(attention)
            activation = tf.matmul(attention, v)
            # output: -> batch_size x max_frames x num_units

            activation = tf.nn.l2_normalize(activation, 2)
            reshaped_activation = tf.reshape(activation, [-1, self.cluster_size * self.num_units])
            activation = tf.nn.l2_normalize(reshaped_activation)
            reshaped_final_activation = tf.reshape(activation, [-1, self.cluster_size, self.num_units])
            return reshaped_final_activation

    def forward(self, inputs, **unused_params):
        result = self.self_attention(inputs, head_id=0)
        for i in range(1, self.num_heads):
            output = self.self_attention(inputs, head_id=i)
            result = tf.concat([result, output], 2)
        result = tf.reshape(result, [-1, self.num_units * self.num_heads])
        output = tf.layers.dense(result, self.feature_size, use_bias=False, activation=None)
        # -> batch_size x cluster_size x feature_size
        output = tf.layers.batch_normalization(output, training=self.is_training)
        output = tf.reshape(output, [-1, self.cluster_size, self.feature_size])
        return output


class FishMultiHeadFF(modules.BaseModule):
    def __init__(self, feature_size, filter_size, num_units, num_heads, cluster_size, is_training):
        self.feature_size = feature_size
        self.filter_size = filter_size
        self.num_units = num_units
        self.num_heads = num_heads
        self.cluster_size = cluster_size
        self.is_training = is_training

    def self_attention(self, inputs, head_id):
        with tf.variable_scope("head{}".format(head_id)):
            reshaped_inputs = tf.reshape(inputs, [-1, self.feature_size])
            q = tf.layers.dense(reshaped_inputs, self.num_units, use_bias=False, activation=None)
            k = tf.layers.dense(reshaped_inputs, self.num_units, use_bias=False, activation=None)
            v = tf.layers.dense(reshaped_inputs, self.num_units, use_bias=False, activation=None)

            q = tf.reshape(q, [-1, self.cluster_size, self.num_units])
            k = tf.reshape(k, [-1, self.cluster_size, self.num_units])
            v = tf.reshape(v, [-1, self.cluster_size, self.num_units])

            attention = tf.matmul(q, tf.transpose(k, perm=[0, 2, 1]))
            # -> batch_size x max_frames x max_frames
            attention = tf.layers.batch_normalization(attention, training=self.is_training)
            attention = tf.nn.softmax(attention)
            activation = tf.matmul(attention, v)
            # output: -> batch_size x max_frames x num_units

            activation = tf.nn.l2_normalize(activation, 2)
            reshaped_activation = tf.reshape(activation, [-1, self.cluster_size * self.num_units])
            activation = tf.nn.l2_normalize(reshaped_activation)
            reshaped_final_activation = tf.reshape(activation, [-1, self.cluster_size, self.num_units])
            return reshaped_final_activation

    def forward(self, inputs, should_add, **unused_params):
        result = self.self_attention(inputs, head_id=0)
        for i in range(1, self.num_heads):
            output = self.self_attention(inputs, head_id=i)
            result = tf.concat([result, output], 2)
        result = tf.reshape(result, [-1, self.num_units * self.num_heads])
        output = tf.layers.dense(result, self.feature_size, use_bias=False, activation=None)
        # -> batch_size x cluster_size x feature_size
        if should_add:
            output = output + inputs
        output = tf.layers.batch_normalization(output, training=self.is_training)
        output = tf.reshape(output, [-1, self.cluster_size, self.feature_size])
        return output


class FishGate(modules.BaseModule):
    def __init__(self, hidden_size, is_training):
        self.hidden_size = hidden_size
        self.is_training = is_training

    def forward(self, inputs, **unused_params):
        gating_weights = tf.get_variable("gating_weights_2",
                                         [self.hidden_size, self.hidden_size],
                                         initializer=tf.random_normal_initializer(
                                             stddev=1 / math.sqrt(self.hidden_size)))

        gates = tf.matmul(inputs, gating_weights)
        gates = tf.layers.batch_normalization(gates, training=self.is_training)
        gates = tf.sigmoid(gates)
        activation = tf.multiply(inputs, gates)
        return activation
