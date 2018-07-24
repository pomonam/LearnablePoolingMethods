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
        # float_cpy = tf.cast(self.feature_size, dtype=tf.float32)
        # attention_weights = tf.divide(attention_weights, tf.sqrt(float_cpy))
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


class LuckyFishFastForward(modules.BaseModule):
    """ Feed Forward Network. """
    def __init__(self, feature_size, max_frames, filter_size, relu_dropout,
                 is_train, scope_id):
        """ Initialize class FeedForwardNetwork.
        :param hidden_size: int
        :param filter_size: int
        :param relu_dropout: int
        :param is_train: bool
        :param scope_id: String
        """
        self.feature_size = feature_size
        self.max_frames = max_frames
        self.filter_size = filter_size
        self.relu_dropout = relu_dropout
        self.is_train = is_train
        self.scope_id = scope_id

    def forward(self, inputs, **unused_params):
        """ Forward method for FeedForwardNetwork.
        :param inputs: 3D Tensor with size 'batch_size x num_feature x feature_size'
        :return: 3D Tensor with size 'batch_size x num_feature x hidden_size'
        """
        reshaped_inputs = tf.reshape(inputs, [-1, self.feature_size])
        filter_output = tf.layers.dense(reshaped_inputs, self.filter_size,
                                        use_bias=True,
                                        activation=tf.nn.relu,
                                        name="filter_output{}".format(self.scope_id))
        # -> (batch_size * max_frames) x filter_size

        output = tf.layers.dense(filter_output, self.feature_size,
                                 use_bias=True,
                                 activation=tf.nn.relu,
                                 name="ff_output{}".format(self.scope_id))
        # -> (batch_size * max_frames) x feature_size

        output = output + reshaped_inputs
        output = tf.contrib.layers.layer_norm(output)
        activation = tf.reshape(output, [-1, self.max_frames, self.feature_size])

        return activation


class FishMultiHead(modules.BaseModule):
    def __init__(self, feature_size, filter_size, num_units, num_heads, max_frames, is_training):
        self.feature_size = feature_size
        self.filter_size = filter_size
        self.num_units = num_units
        self.num_heads = num_heads
        self.max_frames = max_frames
        self.is_training = is_training

    def self_attention(self, inputs, head_id):
        with tf.variable_scope("head{}".format(head_id)):
            reshaped_inputs = tf.reshape(inputs, [-1, self.feature_size])
            q = tf.layers.dense(reshaped_inputs, self.num_units, use_bias=False, activation=None)
            k = tf.layers.dense(reshaped_inputs, self.num_units, use_bias=False, activation=None)
            v = tf.layers.dense(reshaped_inputs, self.num_units, use_bias=False, activation=None)

            q = tf.reshape(q, [-1, self.max_frames, self.num_units])
            k = tf.reshape(k, [-1, self.max_frames, self.num_units])
            v = tf.reshape(v, [-1, self.max_frames, self.num_units])

            attention = tf.matmul(q, tf.transpose(k, perm=[0, 2, 1]))
            # float_cpy = tf.cast(self.feature_size, dtype=tf.float32)
            # attention = tf.divide(attention, tf.sqrt(float_cpy))
            attention = tf.layers.batch_normalization(attention, training=self.is_training)
            attention = tf.nn.softmax(attention)
            activation = tf.matmul(attention, v)
            # output: -> batch_size x max_frames x num_units

            activation = tf.nn.l2_normalize(activation, 2)
            reshaped_activation = tf.reshape(activation, [-1, self.max_frames * self.num_units])
            activation = tf.nn.l2_normalize(reshaped_activation)
            reshaped_final_activation = tf.reshape(activation, [-1, self.max_frames, self.num_units])
            return reshaped_final_activation

    def forward(self, inputs, **unused_params):
        result = self.self_attention(inputs, head_id=0)
        for i in range(1, self.num_heads):
            output = self.self_attention(inputs, head_id=i)
            result = tf.concat([result, output], 2)
        result = tf.reshape(result, [-1, self.num_units * self.num_heads])
        output = tf.layers.dense(result, self.feature_size, use_bias=False, activation=None)
        output = tf.layers.batch_normalization(output, training=self.is_training)

        filter1 = tf.layers.dense(output, self.filter_size,
                                  use_bias=False,
                                  activation=tf.nn.leaky_relu,
                                  name="filter_output")
        filter1 = tf.layers.batch_normalization(filter1, training=self.is_training)

        filter2 = tf.layers.dense(filter1, self.feature_size,
                                  use_bias=True,
                                  activation=None,
                                  name="ff_output")

        output = filter2 + output
        output = tf.nn.leaky_relu(output)
        output = tf.layers.batch_normalization(output, training=self.is_training)
        output = tf.reshape(output, [-1, self.max_frames, self.feature_size])

        return output
