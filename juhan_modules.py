import tensorflow as tf
import modules


class EncoderStack:
    """ Transformer encoder stack.
    The encoder stack is made up of N identical layers. Each layer is composed of the sublayers:
    1. Self-attention layer
    2. Feed Forward network (which is 2 fully-connected layers)
    """
    def __init__(self, num_heads, num_units, max_frames, block_id):
        """
        :param num_heads: Number of self-attention modules
        :param num_units: last dimension of Q, K, V
        """
        self.num_heads = num_heads
        self.num_units = num_units
        self.max_frames = max_frames
        self.block_id = block_id

    def scaled_dot_product(self, queries, keys, dropout_rate, is_training, head_id):
        """
        :param Q: batch_size x max_frames x num_units
        :param K: batch_size x max_frames x num_units
        :param V: batch_size x max_frames x num_units
        :return:
        """
        with tf.variable_scope("multi_head_head{}".format(str(head_id))):
            # Linear projections
            q = tf.layers.dense(queries, self.num_units, activation=None)
            k = tf.layers.dense(keys, self.num_units, activation=None)
            v = tf.layers.dense(keys, self.num_units, activation=None)
            # -> batch_size x num_frames x num_units

            outputs = tf.matmul(q, tf.transpose(k, [0, 2, 1]))
            # -> batch_size x num_frames x num_frames
            outputs = tf.divide(outputs, (k.get_shape().as_list()[-1] ** 0.5))
            outputs = tf.nn.softmax(outputs)
            outputs = tf.layers.dropout(outputs, rate=dropout_rate, training=tf.convert_to_tensor(is_training))
            outputs = tf.matmul(outputs, v)

            return outputs

    def forward(self, queries, keys, dropout_rate, is_training):
        with tf.variable_scope("multi_head_block{}".format(str(self.block_id))):
            result = self.scaled_dot_product(queries, keys, dropout_rate, is_training, head_id=0)
            for i in range(1, self.num_heads):
                result = tf.identity(result)
                output = self.scaled_dot_product(queries, keys, dropout_rate, is_training, head_id=i)
                result = tf.concat([result, output], 2)
            # result: -> batch_size x max_frames x (num_units * num_heads)

            # 2 layers of 1 x 1 convolution
            output = tf.reshape(result, [-1, self.max_frames, self.feature_size])
            output = tf.layers.conv1d(output, filters=4 * self.num_units, kernel_size=1, activation=tf.nn.relu,
                                      use_bias=True)
            output = tf.layers.conv1d(output, filters=self.num_units, kernel_size=1, activation=None, use_bias=True)

            # Residual connection & Layer normalization
            output = tf.contrib.layers.layer_norm(output)
            output = tf.reshape(output, [-1, self.feature_size])

            # Fully connected layer.
            reshaped_result = tf.reshape(result, [-1, self.num_units * self.num_heads])
            reshaped_result = tf.layers.dense(reshaped_result, self.num_units, activation=tf.nn.relu)

            activation = reshaped_result + queries
            activation = tf.contrib.layers.layer_norm(activation)

            dense1 = tf.layers.dense(activation, self.num_units, activation=tf.nn.relu)
            dense2 = tf.layers.dense(dense1, self.num_units, activation=tf.nn.relu)

            final_activation = activation + dense2
            final_activation = tf.contrib.layers.layer_norm(final_activation)

            return final_activation


class JuhanTransformerNetVladDecoder:
    def __init__(self, num_heads, num_units, max_frames, block_id):
        """
        :param num_heads: Number of self-attention modules
        :param num_units: last dimension of Q, K, V
        """
        self.num_heads = num_heads
        self.num_units = num_units
        self.max_frames = max_frames
        self.block_id = block_id

    def scaled_dot_product(self, queries, keys, dropout_rate, is_training, head_id):
        """
        :param Q: batch_size x max_frames x num_units
        :param K: batch_size x max_frames x num_units
        :param V: batch_size x max_frames x num_units
        :return:
        """
        with tf.variable_scope("multi_head_head{}".format(str(head_id))):
            # Linear projections
            q = tf.layers.dense(queries, self.num_units, activation=None)
            k = tf.layers.dense(keys, self.num_units, activation=None)
            v = tf.layers.dense(keys, self.num_units, activation=None)
            # -> batch_size x num_frames x num_units

            outputs = tf.matmul(q, tf.transpose(k, [0, 2, 1]))
            # -> batch_size x num_frames x num_frames
            outputs = tf.divide(outputs, (k.get_shape().as_list()[-1] ** 0.5))
            outputs = tf.nn.softmax(outputs)
            outputs = tf.layers.dropout(outputs, rate=dropout_rate, training=tf.convert_to_tensor(is_training))
            outputs = tf.matmul(outputs, v)

            return outputs

    def forward(self, queries, keys, dropout_rate, is_training):
        with tf.variable_scope("multi_head_block{}".format(str(self.block_id))):
            result = self.scaled_dot_product(queries, keys, dropout_rate, is_training, head_id=0)
            for i in range(1, self.num_heads):
                result = tf.identity(result)
                output = self.scaled_dot_product(queries, keys, dropout_rate, is_training, head_id=i)
                result = tf.concat([result, output], 2)
            # result: -> batch_size x max_frames x (num_units * num_heads)

            # Fully connected layer.
            reshaped_result = tf.reshape(result, [-1, self.num_units * self.num_heads])
            reshaped_result = tf.layers.dense(reshaped_result, self.num_units, activation=tf.nn.relu)

            activation = reshaped_result + queries
            activation = tf.contrib.layers.layer_norm(activation)

            dense1 = tf.layers.dense(activation, self.num_units, activation=tf.nn.relu)
            dense2 = tf.layers.dense(dense1, self.num_units, activation=tf.nn.relu)

            final_activation = activation + dense2
            final_activation = tf.contrib.layers.layer_norm(final_activation)

            return final_activation