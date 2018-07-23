import tensorflow as tf
import modules
import math


class CrazyCluster(modules.BaseModule):
    def __init__(self, feature_size, hidden_size, max_frames, last_layer, num_cluster, do_shift=True):
        self.feature_size = feature_size
        self.max_frames = max_frames
        self.hidden_size = hidden_size
        self.num_cluster = num_cluster
        self.last_layer = last_layer
        self.do_shift = do_shift

    def forward(self, inputs, **unused_params):
        """
        :param inputs: batch_size x num_frames x feature_size
        :param cluster_id:
        :return:
        """
        reshaped_inputs = tf.reshape(inputs, [-1, self.feature_size])
        attention = tf.layers.dense(reshaped_inputs, self.num_cluster, activation=None)
        reshaped_attention = tf.reshape(attention, [-1, self.max_frames, self.num_cluster])
        float_cpy = tf.cast(self.feature_size, dtype=tf.float32)
        reshaped_attention = tf.divide(reshaped_attention, tf.sqrt(float_cpy))
        attention = tf.nn.softmax(reshaped_attention)
        # -> batch_size x max_frame x cluster_size

        transposed_attention = tf.transpose(attention, perm=[0, 2, 1])
        # -> batch_size x cluster_size x feature_size
        activation = tf.matmul(transposed_attention, inputs)
        # -> batch_size x num_cluster x feature_size

        alpha = tf.get_variable("alpha",
                                [self.num_cluster, 1],
                                initializer=tf.constant_initializer(1.0))
        beta = tf.get_variable("beta",
                               [self.num_cluster, 1],
                               initializer=tf.constant_initializer(0.0))
        activation = activation * alpha
        activation = activation + beta
        activation = tf.reshape(activation, [-1, self.feature_size])
        activation = tf.nn.l2_normalize(activation)
        float_cpy = tf.cast(self.num_cluster, dtype=tf.float32)
        activation = tf.divide(activation, tf.sqrt(float_cpy))
        activation = tf.reshape(activation, [-1, self.num_cluster, self.feature_size])

        return activation


class CrazyMultiHeadV2(modules.BaseModule):
    def __init__(self, feature_size, num_units, num_heads, max_frames, is_training):
        self.feature_size = feature_size
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
            # q, k, v: -> batch_size x max_frames x num_units

            attention = tf.matmul(q, tf.transpose(k, perm=[0, 2, 1]))
            float_cpy = tf.cast(self.num_units, dtype=tf.float32)
            attention = tf.divide(attention, tf.sqrt(float_cpy))
            # -> batch_size x max_frames x max_frames
            attention = tf.nn.softmax(attention)
            activation = tf.matmul(attention, v)
            # output: -> batch_size x max_frames x num_units

            alpha = tf.get_variable("alpha",
                                    [1],
                                    initializer=tf.constant_initializer(1.0))
            beta = tf.get_variable("beta",
                                   [1],
                                   initializer=tf.constant_initializer(0.0))
            reshaped_activation = tf.reshape(activation, [-1, self.num_units])
            reshaped_activation = reshaped_activation * alpha
            reshaped_activation = reshaped_activation + beta
            reshaped_activation = tf.nn.l2_normalize(reshaped_activation)
            float_cpy = tf.cast(self.num_heads, dtype=tf.float32)
            reshaped_activation = tf.divide(reshaped_activation, tf.sqrt(float_cpy))
            activation = tf.reshape(reshaped_activation, [-1, self.max_frames, self.num_units])

            return activation

    def forward(self, inputs, **unused_params):
        result = self.self_attention(inputs, head_id=0)
        for i in range(1, self.num_heads):
            output = self.self_attention(inputs, head_id=i)
            result = tf.concat([result, output], 2)
        reshaped_result = tf.reshape(result, [-1, self.num_units * self.num_heads])

        activation = tf.layers.dense(reshaped_result, self.feature_size, use_bias=False, activation=None)
        reshaped_inputs = tf.reshape(inputs, [-1, self.feature_size])
        reshaped_activation = tf.reshape(activation, [-1, self.feature_size])
        activation = reshaped_activation + reshaped_inputs
        activation = tf.contrib.layers.layer_norm(activation)
        activation = tf.reshape(activation, [-1, self.max_frames, self.feature_size])
        return activation


class CrazyFeedForwardV2(modules.BaseModule):
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
