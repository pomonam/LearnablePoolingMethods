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
        inputs = tf.reshape(inputs, [-1, self.max_frames, self.feature_size])
        reshaped_inputs = tf.reshape(inputs, [-1, self.feature_size])
        # -> reshaped_inputs: (batch_size * max_frames) x feature_size
        attention_weights = tf.layers.dense(reshaped_inputs, self.cluster_size, use_bias=False, activation=None)
        float_cpy = tf.cast(self.feature_size, dtype=tf.float32)
        attention_weights = tf.divide(attention_weights, tf.sqrt(float_cpy))
        attention_weights = tf.layers.batch_normalization(attention_weights, training=self.is_training)
        if self.is_training:
            attention_weights = tf.nn.dropout(attention_weights, 0.8)
        attention_weights = tf.nn.softmax(attention_weights)

        reshaped_attention = tf.reshape(attention_weights, [-1, self.max_frames, self.cluster_size])
        transposed_attention = tf.transpose(reshaped_attention, perm=[0, 2, 1])
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


class LuckyFishModuleV2(modules.BaseModule):
    """ Attention cluster. """
    def __init__(self, feature_size, max_frames, cluster_size, add_batch_norm, shift_operation, is_training):
        self.feature_size = feature_size
        self.max_frames = max_frames
        self.is_training = is_training
        self.add_batch_norm = add_batch_norm
        self.shift_operation = shift_operation
        self.cluster_size = cluster_size

    def forward(self, inputs, **unused_params):
        inputs = tf.reshape(inputs, [-1, self.feature_size])
        reshaped_inputs = tf.reshape(inputs, [-1, self.max_frames, self.feature_size])

        attention_weights = tf.layers.dense(inputs, self.cluster_size, use_bias=False, activation=None)
        float_cpy = tf.cast(self.feature_size, dtype=tf.float32)
        attention_weights = tf.divide(attention_weights, tf.sqrt(float_cpy))
        attention_weights = tf.layers.batch_normalization(attention_weights, training=self.is_training)
        if self.is_training:
            attention_weights = tf.nn.dropout(attention_weights, 0.7)
        attention_weights = tf.nn.softmax(attention_weights)

        reshaped_attention = tf.reshape(attention_weights, [-1, self.max_frames, self.cluster_size])
        transposed_attention = tf.transpose(reshaped_attention, perm=[0, 2, 1])
        # -> transposed_attention: batch_size x cluster_size x max_frames
        activation = tf.matmul(transposed_attention, reshaped_inputs)
        # -> activation: batch_size x cluster_size x feature_size
        transformed_activation = tf.transpose(activation, perm=[0, 2, 1])
        # -> transformed_activation: batch_size x feature_size x cluster_size

        if self.shift_operation:
            alpha = tf.get_variable("alpha",
                                    [self.cluster_size],
                                    initializer=tf.constant_initializer(1.0))
            beta = tf.get_variable("beta",
                                   [self.cluster_size],
                                   initializer=tf.constant_initializer(0.0))
            transformed_activation = tf.multiply(transformed_activation, alpha)
            transformed_activation = tf.add(transformed_activation, beta)
            float_cpy = tf.cast(self.cluster_size, dtype=tf.float32)
            transformed_activation = tf.divide(transformed_activation, tf.sqrt(float_cpy))

        normalized_activation = tf.nn.l2_normalize(transformed_activation, 2)
        normalized_activation = tf.reshape(normalized_activation, [-1, self.cluster_size * self.feature_size])
        normalized_activation = tf.nn.l2_normalize(normalized_activation, 1)

        return normalized_activation


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


class FishGate2(modules.BaseModule):
    def __init__(self, hidden_size, is_training):
        self.hidden_size = hidden_size
        self.is_training = is_training

    def forward(self, inputs, **unused_params):

        weight1 = tf.layers.dense(inputs, self.hidden_size, use_bias=False, activation=tf.nn.relu)
        weight1 = tf.layers.batch_normalization(weight1, training=self.is_training)

        weight2 = tf.layers.dense(weight1, self.hidden_size, use_bias=False, activation=tf.nn.relu)
        weight2 = tf.layers.batch_normalization(weight2, training=self.is_training)

        weight3 = tf.layers.dense(weight2, self.hidden_size, use_bias=False, activation=None)

        output = inputs + weight3
        output = tf.nn.relu(output)
        output = tf.layers.batch_normalization(output, training=self.is_training)

        return output


class FishEncoderStack(modules.BaseModule):
    """Transformer encoder stack.
    The encoder stack is made up of N identical layers. Each layer is composed
    of the sublayers:
    1. Self-attention layer
    2. Feedforward network (which is 2 fully-connected layers)
    """
    def __init__(self, num_layers, max_frames, hidden_size, num_heads, filter_size, relu_dropout, attention_dropout, is_training):
        self.layers = []
        for _ in range(num_layers):
            # Create sublayers for each layer.
            self_attention_layer = FishSelfAttention(hidden_size, num_heads, attention_dropout, is_training)
            feed_forward_network = FishFowardNetwork(hidden_size, max_frames, filter_size, relu_dropout, is_training, False)
            self.layers.append([
                PrePostProcessingWrapper(self_attention_layer, 0, is_training),
                PrePostProcessingWrapper(feed_forward_network, 0, is_training)])

    def forward(self, encoder_inputs, attention_bias, inputs_padding):
        """Return the output of the encoder layer stacks.
        Args:
          encoder_inputs: tensor with shape [batch_size, input_length, hidden_size]
          attention_bias: bias for the encoder self-attention layer.
            [batch_size, 1, 1, input_length]
          inputs_padding: P
        Returns:
          Output of encoder layer stack.
          float32 tensor with shape [batch_size, input_length, hidden_size]
        """
        for n, layer in enumerate(self.layers):
            # Run inputs through the sublayers.
            self_attention_layer = layer[0]
            feed_forward_network = layer[1]

            with tf.variable_scope("layer_%d" % n):
                with tf.variable_scope("self_attention"):
                    encoder_inputs = self_attention_layer.forward(encoder_inputs, encoder_inputs, 0.0)
                with tf.variable_scope("ffn"):
                    encoder_inputs = feed_forward_network.forward(encoder_inputs, None)

        return tf.contrib.layers.layer_norm(encoder_inputs)


class FishSelfAttention(modules.BaseModule):
    """Multi-headed attention layer."""

    def __init__(self, hidden_size, num_heads, attention_dropout, is_training):
        if hidden_size % num_heads != 0:
            raise ValueError("Hidden size must be evenly divisible by the number of "
                             "heads.")
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.attention_dropout = attention_dropout
        self.is_training = is_training

        # Layers for linearly projecting the queries, keys, and values.
        self.q_dense_layer = tf.layers.Dense(hidden_size, use_bias=False, name="q")
        self.k_dense_layer = tf.layers.Dense(hidden_size, use_bias=False, name="k")
        self.v_dense_layer = tf.layers.Dense(hidden_size, use_bias=False, name="v")

        self.output_dense_layer = tf.layers.Dense(hidden_size, use_bias=False,
                                                  name="output_transform")

    def split_heads(self, x):
        """Split x into different heads, and transpose the resulting value.
        The tensor is transposed to insure the inner dimensions hold the correct
        values during the matrix multiplication.
        Args:
          x: A tensor with shape [batch_size, length, hidden_size]
        Returns:
          A tensor with shape [batch_size, num_heads, length, hidden_size/num_heads]
        """
        with tf.name_scope("split_heads"):
            batch_size = tf.shape(x)[0]
            length = tf.shape(x)[1]

            # Calculate depth of last dimension after it has been split.
            depth = (self.hidden_size // self.num_heads)

            # Split the last dimension
            x = tf.reshape(x, [batch_size, length, self.num_heads, depth])

            # Transpose the result
            return tf.transpose(x, [0, 2, 1, 3])

    def combine_heads(self, x):
        """Combine tensor that has been split.
        Args:
          x: A tensor [batch_size, num_heads, length, hidden_size/num_heads]
        Returns:
          A tensor with shape [batch_size, length, hidden_size]
        """
        with tf.name_scope("combine_heads"):
            batch_size = tf.shape(x)[0]
            length = tf.shape(x)[2]
            x = tf.transpose(x, [0, 2, 1, 3])  # --> [batch, length, num_heads, depth]
            return tf.reshape(x, [batch_size, length, self.hidden_size])

    def forward(self, x, y, bias=0.0, cache=None):
        """Apply attention mechanism to x and y.
        Args:
          x: a tensor with shape [batch_size, length_x, hidden_size]
          y: a tensor with shape [batch_size, length_y, hidden_size]
          bias: attention bias that will be added to the result of the dot product.
          cache: (Used during prediction) dictionary with tensors containing results
            of previous attentions. The dictionary must have the items:
                {"k": tensor with shape [batch_size, i, key_channels],
                 "v": tensor with shape [batch_size, i, value_channels]}
            where i is the current decoded length.
        Returns:
          Attention layer output with shape [batch_size, length_x, hidden_size]
        """
        # Linearly project the query (q), key (k) and value (v) using different
        # learned projections. This is in preparation of splitting them into
        # multiple heads. Multi-head attention uses multiple queries, keys, and
        # values rather than regular attention (which uses a single q, k, v).
        q = self.q_dense_layer(x)
        q = tf.layers.batch_normalization(q, training=self.is_training)

        k = self.k_dense_layer(y)
        k = tf.layers.batch_normalization(k, training=self.is_training)

        v = self.v_dense_layer(y)
        v = tf.layers.batch_normalization(v, training=self.is_training)

        # Split q, k, v into heads.
        q = self.split_heads(q)
        k = self.split_heads(k)
        v = self.split_heads(v)

        # Scale q to prevent the dot product between q and k from growing too large.
        # depth = (self.hidden_size // self.num_heads)
        # q *= depth ** -0.5

        # Calculate dot product attention
        logits = tf.matmul(q, k, transpose_b=True)
        weights = tf.nn.softmax(logits, name="attention_weights")
        # if self.train:
        #     weights = tf.nn.dropout(weights, 1.0 - self.attention_dropout)
        attention_output = tf.matmul(weights, v)

        # Recombine heads --> [batch_size, length, hidden_size]
        attention_output = self.combine_heads(attention_output)

        # Run the combined outputs through another linear projection layer.
        attention_output = self.output_dense_layer(attention_output)
        return attention_output


class FishFowardNetwork(modules.BaseModule):
    """Fully connected feedforward network."""

    def __init__(self, hidden_size, max_frames, filter_size, relu_dropout, train, allow_pad):
        self.hidden_size = hidden_size
        self.filter_size = filter_size
        self.max_frames = max_frames
        self.relu_dropout = relu_dropout
        self.train = train
        self.allow_pad = allow_pad

        self.filter_dense_layer = tf.layers.Dense(
            filter_size, use_bias=True, activation=tf.nn.relu, name="filter_layer")
        self.output_dense_layer = tf.layers.Dense(
            hidden_size, use_bias=True, name="output_layer")

    def forward(self, x, padding=None):
        """Return outputs of the feedforward network.
        Args:
          x: tensor with shape [batch_size, length, hidden_size]
        Returns:
          Output of the feedforward network.
          tensor with shape [batch_size, length, hidden_size]
        """
        # output = self.filter_dense_layer(x)
        # output = self.output_dense_layer(output)
        reshaped_inputs = tf.reshape(x, [-1, self.hidden_size])
        gating_weights = tf.layers.dense(reshaped_inputs, self.hidden_size, use_bias=False, activation=None)
        gates = tf.layers.batch_normalization(gating_weights, training=self.train)
        gates = tf.sigmoid(gates)
        activation = tf.multiply(reshaped_inputs, gates)
        reshaped_activation = tf.reshape(activation, [-1, self.max_frames, self.hidden_size])

        return reshaped_activation


class PrePostProcessingWrapper(modules.BaseModule):
    """Wrapper class that applies layer pre-processing and post-processing."""

    def __init__(self, layer, process_dropout, is_training):
        self.layer = layer
        self.postprocess_dropout = process_dropout
        self.is_training = is_training

    def forward(self, x, *args, **kwargs):
        # Preprocessing: apply layer normalization
        y = tf.contrib.layers.layer_norm(x)

        # Get layer output
        y = self.layer.forward(y, *args, **kwargs)

        return x + y