import tensorflow as tf
import modules


class TransformerEncoder(modules.BaseModule):
    def __init__(self, feature_size, hidden_size, num_heads, attention_dropout,
                 ff_filter_size, ff_relu_dropout,
                 is_train, scope_id):
        self.feature_size = feature_size
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.attention_dropout = attention_dropout
        self.ff_filter_size = ff_filter_size
        self.ff_relu_dropout = ff_relu_dropout
        self.is_train = is_train
        self.scope_id = scope_id

        self.multi_head_attention = MultiHeadAttention(feature_size,
                                                       hidden_size,
                                                       num_heads,
                                                       attention_dropout,
                                                       is_train)

        self.ff_network = FeedForwardNetwork(feature_size,
                                             ff_filter_size,
                                             ff_relu_dropout,
                                             is_train,
                                             self.scope_id)

    def forward(self, inputs, **unused_params):
        """
        :param inputs: [batch_size, input_length, hidden_size]
        :param unused_params:
        :return:
        """
        attention = self.multi_head_attention.forward(inputs, inputs)
        attention = attention + inputs
        attention = tf.contrib.layers.layer_norm(attention)

        ff_output = self.ff_network.forward(attention)
        ff_output = ff_output + attention
        ff_output = tf.contrib.layers.layer_norm(ff_output)

        return ff_output


class TransformerDecoder(modules.BaseModule):
    def __init__(self, feature_size, hidden_size, num_heads, attention_dropout,
                 ff_filter_size, ff_relu_dropout,
                 is_train, scope_id):
        self.feature_size = feature_size
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.attention_dropout = attention_dropout
        self.ff_filter_size = ff_filter_size
        self.ff_relu_dropout = ff_relu_dropout
        self.is_train = is_train
        self.scope_id = scope_id

        self.multi_head_attention1 = MultiHeadAttention(feature_size,
                                                        hidden_size,
                                                        num_heads,
                                                        attention_dropout,
                                                        is_train)

        self.multi_head_attention2 = MultiHeadAttention(feature_size,
                                                        hidden_size,
                                                        num_heads,
                                                        attention_dropout,
                                                        is_train)

        self.ff_network = FeedForwardNetwork(feature_size,
                                             ff_filter_size,
                                             ff_relu_dropout,
                                             is_train,
                                             scope_id)

    def forward(self, inputs, encoder_inputs, **unused_params):
        with tf.variable_scope("first_mha"):
            attention1 = self.multi_head_attention1.forward(inputs, inputs)
            attention1 = attention1 + inputs
            attention1 = tf.contrib.layers.layer_norm(attention1)

        with tf.variable_scope("second_mha"):
            attention2 = self.multi_head_attention2.forward(attention1, encoder_inputs)
            attention2 = attention2 + attention1
            attention2 = tf.contrib.layers.layer_norm(attention2)

        ff_output = self.ff_network.forward(attention2)
        ff_output = ff_output + attention2
        ff_output = tf.contrib.layers.layer_norm(ff_output)
        return ff_output


class MultiHeadAttention(modules.BaseModule):
    def __init__(self, feature_size, hidden_size, num_heads, attention_dropout, is_train):
        """ Initialize class MultiHeadAttention.
        :param hidden_size: int
        :param num_heads: int
        :param attention_dropout: float
        :param is_train: bool
        """
        self.feature_size = feature_size
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.attention_dropout = attention_dropout
        self.is_train = is_train

    def split_heads(self, inputs):
        """ Split x into different heads, and transpose the resulting value.
        The tensor is transposed to insure the inner dimensions hold the correct
        values during the matrix multiplication.
        :param inputs: 3D Tensor with shape 'batch_size x length x hidden_size'
        :return:
        """
        with tf.name_scope("split_heads"):
            batch_size = tf.shape(inputs)[0]
            length = tf.shape(inputs)[1]

            # Calculate depth of last dimension after it has been split.
            depth = (self.hidden_size // self.num_heads)

            # Split the last dimension
            x = tf.reshape(inputs, [batch_size, length, self.num_heads, depth])

            # Transpose the result
            return tf.transpose(x, [0, 2, 1, 3])

    def combine_heads(self, inputs):
        """ Combine tensor that has been split.
        :param inputs: 4D Tensor with shape 'batch_size x num_heads, num_feature, hidden_size/num_heads'
        :return: 3D Tensor with shape 'batch_size x length x hidden_size'
        """
        with tf.name_scope("combine_heads"):
            batch_size = tf.shape(inputs)[0]
            length = tf.shape(inputs)[2]
            x = tf.transpose(inputs, [0, 2, 1, 3])  # --> [batch, length, num_heads, depth]
            return tf.reshape(x, [batch_size, length, self.hidden_size])

    def forward(self, queries, keys):
        """ Forward method for MultiHeadAttention
        :param queries: 3D Tensor with shape 'batch_size x length x hidden_size'
        :param keys: 3D Tensor with shape 'batch_size x length x hidden_size'
        :return:
        """
        # Layers for linearly projecting the queries, keys, and values.
        q = tf.layers.dense(queries, self.hidden_size, use_bias=False, name="q")
        k = tf.layers.dense(keys, self.hidden_size, use_bias=False, name="k")
        v = tf.layers.dense(keys, self.hidden_size, use_bias=False, name="v")

        # Split q, k, v into heads.
        q = self.split_heads(q)
        k = self.split_heads(k)
        v = self.split_heads(v)
        # -> [batch_size, num_heads, length, hidden_size/num_heads]

        # Scale q to prevent the dot product between q and k from growing too large.
        depth = (self.hidden_size // self.num_heads)
        q *= depth ** -0.5

        logits = tf.matmul(q, k, transpose_b=True)
        weights = tf.nn.softmax(logits, name="attention_weights")

        if self.is_train:
            weights = tf.nn.dropout(weights, 1.0 - self.attention_dropout)
        attention_output = tf.matmul(weights, v)

        # -> batch_size x length x hidden_size]
        attention_output = self.combine_heads(attention_output)

        attention_output = tf.layers.dense(attention_output,
                                           self.feature_size,
                                           use_bias=True, name="output_transform")
        return attention_output


class FeedForwardNetwork(modules.BaseModule):
    """ Feed Forward Network. """
    def __init__(self, feature_size, filter_size, relu_dropout,
                 is_train, scope_id):
        """ Initialize class FeedForwardNetwork.
        :param hidden_size: int
        :param filter_size: int
        :param relu_dropout: int
        :param is_train: bool
        :param scope_id: String
        """
        self.feature_size = feature_size
        self.filter_size = filter_size
        self.relu_dropout = relu_dropout
        self.is_train = is_train
        self.scope_id = scope_id

    def forward(self, inputs, **unused_params):
        """ Forward method for FeedForwardNetwork.
        :param inputs: 3D Tensor with size 'batch_size x num_feature x feature_size'
        :return: 3D Tensor with size 'batch_size x num_feature x hidden_size'
        """
        filter_output = tf.layers.dense(inputs, self.filter_size,
                                        use_bias=True,
                                        activation=tf.nn.relu,
                                        name="filter_output{}".format(self.scope_id))
        if self.is_train:
            filter_output = tf.nn.dropout(filter_output, 1.0 - self.relu_dropout)

        output = tf.layers.dense(filter_output, self.feature_size,
                                 use_bias=True,
                                 activation=tf.nn.relu,
                                 name="ff_output".format(self.scope_id))
        return output
