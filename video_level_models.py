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

"""Contains model definitions."""
# noinspection PyUnresolvedReferences
import pathmagic
from tensorflow import flags
import attention_modules
import fish_modules
import tensorflow as tf
import tensorflow.contrib.slim as slim
import models
import math

FLAGS = flags.FLAGS
flags.DEFINE_integer(
    "moe_num_mixtures", 2,
    "The number of mixtures (excluding the dummy 'expert') used for MoeModel.")


###############################################################################
# Baseline (Benchmark) models #################################################
###############################################################################
flags.DEFINE_float(
    "moe_l2", 1e-8,
    "L2 penalty for MoeModel.")
flags.DEFINE_integer(
    "moe_low_rank_gating", -1,
    "Low rank gating for MoeModel.")
flags.DEFINE_bool(
    "moe_prob_gating", False,
    "Prob gating for MoeModel.")
flags.DEFINE_string(
    "moe_prob_gating_input", "prob",
    "input Prob gating for MoeModel.")


class MoeModel(models.BaseModel):
    """A softmax over a mixture of logistic models (with L2 regularization)."""

    def create_model(self,
                     model_input,
                     vocab_size,
                     is_training,
                     num_mixtures=None,
                     l2_penalty=1e-8,
                     **unused_params):
        """Creates a Mixture of (Logistic) Experts model.
         It also includes the possibility of gating the probabilities
         The model consists of a per-class softmax distribution over a
         configurable number of logistic classifiers. One of the classifiers in the
         mixture is not trained, and always predicts 0.
        Args:
          model_input: 'batch_size' x 'num_features' matrix of input features.
          vocab_size: The number of classes in the dataset.
          is_training: Is this the training phase ?
          num_mixtures: The number of mixtures (excluding a dummy 'expert' that
            always predicts the non-existence of an entity).
          l2_penalty: How much to penalize the squared magnitudes of parameter
            values.
        Returns:
          A dictionary with a tensor containing the probability predictions of the
          model in the 'predictions' key. The dimensions of the tensor are
          batch_size x num_classes.
        """
        num_mixtures = num_mixtures or FLAGS.moe_num_mixtures
        low_rank_gating = FLAGS.moe_low_rank_gating
        l2_penalty = FLAGS.moe_l2
        gating_probabilities = FLAGS.moe_prob_gating
        gating_input = FLAGS.moe_prob_gating_input

        input_size = model_input.get_shape().as_list()[1]
        remove_diag = FLAGS.gating_remove_diag

        if low_rank_gating == -1:
            gate_activations = slim.fully_connected(
                model_input,
                vocab_size * (num_mixtures + 1),
                activation_fn=None,
                biases_initializer=None,
                weights_regularizer=slim.l2_regularizer(l2_penalty),
                scope="gates")
        else:
            gate_activations1 = slim.fully_connected(
                model_input,
                low_rank_gating,
                activation_fn=None,
                biases_initializer=None,
                weights_regularizer=slim.l2_regularizer(l2_penalty),
                scope="gates1")
            gate_activations = slim.fully_connected(
                gate_activations1,
                vocab_size * (num_mixtures + 1),
                activation_fn=None,
                biases_initializer=None,
                weights_regularizer=slim.l2_regularizer(l2_penalty),
                scope="gates2")

        expert_activations = slim.fully_connected(
            model_input,
            vocab_size * num_mixtures,
            activation_fn=None,
            weights_regularizer=slim.l2_regularizer(l2_penalty),
            scope="experts")

        gating_distribution = tf.nn.softmax(tf.reshape(
            gate_activations,
            [-1, num_mixtures + 1]))  # (Batch * #Labels) x (num_mixtures + 1)
        expert_distribution = tf.nn.sigmoid(tf.reshape(
            expert_activations,
            [-1, num_mixtures]))  # (Batch * #Labels) x num_mixtures

        probabilities_by_class_and_batch = tf.reduce_sum(
            gating_distribution[:, :num_mixtures] * expert_distribution, 1)
        probabilities = tf.reshape(probabilities_by_class_and_batch,
                                   [-1, vocab_size])

        if gating_probabilities:
            if gating_input == 'prob':
                gating_weights = tf.get_variable("gating_prob_weights",
                                                 [vocab_size, vocab_size],
                                                 initializer=tf.random_normal_initializer(
                                                     stddev=1 / math.sqrt(vocab_size)))
                gates = tf.matmul(probabilities, gating_weights)
            else:
                gating_weights = tf.get_variable("gating_prob_weights",
                                                 [input_size, vocab_size],
                                                 initializer=tf.random_normal_initializer(
                                                     stddev=1 / math.sqrt(vocab_size)))

                gates = tf.matmul(model_input, gating_weights)

            if remove_diag:
                # removes diagonals coefficients
                diagonals = tf.matrix_diag_part(gating_weights)
                gates = gates - tf.multiply(diagonals, probabilities)

            gates = slim.batch_norm(
                gates,
                center=True,
                scale=True,
                is_training=is_training,
                scope="gating_prob_bn")

            gates = tf.sigmoid(gates)

            probabilities = tf.multiply(probabilities, gates)

        return {"predictions": probabilities}


class FishMoeModel(models.BaseModel):
    """A softmax over a mixture of logistic models (with L2 regularization)."""

    def create_model(self,
                     model_input,
                     vocab_size,
                     is_training,
                     num_mixtures=None,
                     l2_penalty=1e-8,
                     filter_size=2,
                     **unused_params):
        """Creates a Mixture of (Logistic) Experts model.
         It also includes the possibility of gating the probabilities
         The model consists of a per-class softmax distribution over a
         configurable number of logistic classifiers. One of the classifiers in the
         mixture is not trained, and always predicts 0.
        Args:
          model_input: 'batch_size' x 'num_features' matrix of input features.
          vocab_size: The number of classes in the dataset.
          is_training: Is this the training phase ?
          num_mixtures: The number of mixtures (excluding a dummy 'expert' that
            always predicts the non-existence of an entity).
          l2_penalty: How much to penalize the squared magnitudes of parameter
            values.
        Returns:
          A dictionary with a tensor containing the probability predictions of the
          model in the 'predictions' key. The dimensions of the tensor are
          batch_size x num_classes.
        """
        num_mixtures = num_mixtures or FLAGS.moe_num_mixtures
        l2_penalty = FLAGS.moe_l2

        gate_activations = slim.fully_connected(
            model_input,
            vocab_size * (num_mixtures + 1),
            activation_fn=None,
            biases_initializer=None,
            weights_regularizer=slim.l2_regularizer(l2_penalty),
            scope="gates")

        expert_activations = slim.fully_connected(
            model_input,
            vocab_size * num_mixtures,
            activation_fn=None,
            weights_regularizer=slim.l2_regularizer(l2_penalty),
            scope="experts")

        gating_distribution = tf.nn.softmax(tf.reshape(
            gate_activations,
            [-1, num_mixtures + 1]))  # (Batch * #Labels) x (num_mixtures + 1)
        expert_distribution = tf.nn.sigmoid(tf.reshape(
            expert_activations,
            [-1, num_mixtures]))  # (Batch * #Labels) x num_mixtures

        probabilities_by_class_and_batch = tf.reduce_sum(
            gating_distribution[:, :num_mixtures] * expert_distribution, 1)
        probabilities = tf.reshape(probabilities_by_class_and_batch,
                                   [-1, vocab_size])
        probabilities = tf.layers.batch_normalization(probabilities, training=is_training)

        fish_gate = fish_modules.FishGate(hidden_size=vocab_size,
                                          k=2,
                                          dropout_rate=0.9,
                                          is_training=is_training)

        probabilities = fish_gate.forward(probabilities)
        probabilities = tf.contrib.layers.layer_norm(probabilities)

        probabilities = tf.layers.dense(probabilities, vocab_size, use_bias=True, activation=tf.nn.softmax)

        return {"predictions": probabilities}


class FishMoeModel2(models.BaseModel):
    """A softmax over a mixture of logistic models (with L2 regularization)."""

    def create_model(self,
                     model_input,
                     vocab_size,
                     is_training,
                     num_mixtures=None,
                     l2_penalty=1e-8,
                     filter_size=2,
                     **unused_params):
        """Creates a Mixture of (Logistic) Experts model.
         It also includes the possibility of gating the probabilities
         The model consists of a per-class softmax distribution over a
         configurable number of logistic classifiers. One of the classifiers in the
         mixture is not trained, and always predicts 0.
        Args:
          model_input: 'batch_size' x 'num_features' matrix of input features.
          vocab_size: The number of classes in the dataset.
          is_training: Is this the training phase ?
          num_mixtures: The number of mixtures (excluding a dummy 'expert' that
            always predicts the non-existence of an entity).
          l2_penalty: How much to penalize the squared magnitudes of parameter
            values.
        Returns:
          A dictionary with a tensor containing the probability predictions of the
          model in the 'predictions' key. The dimensions of the tensor are
          batch_size x num_classes.
        """
        num_mixtures = num_mixtures or FLAGS.moe_num_mixtures
        l2_penalty = FLAGS.moe_l2

        gate_activations = slim.fully_connected(
            model_input,
            vocab_size * (num_mixtures + 1),
            activation_fn=None,
            biases_initializer=None,
            weights_regularizer=slim.l2_regularizer(l2_penalty),
            scope="gates")

        expert_activations = slim.fully_connected(
            model_input,
            vocab_size * num_mixtures,
            activation_fn=None,
            weights_regularizer=slim.l2_regularizer(l2_penalty),
            scope="experts")

        gating_distribution = tf.nn.softmax(tf.reshape(
            gate_activations,
            [-1, num_mixtures + 1]))  # (Batch * #Labels) x (num_mixtures + 1)
        expert_distribution = tf.nn.sigmoid(tf.reshape(
            expert_activations,
            [-1, num_mixtures]))  # (Batch * #Labels) x num_mixtures

        probabilities_by_class_and_batch = tf.reduce_sum(
            gating_distribution[:, :num_mixtures] * expert_distribution, 1)
        probabilities = tf.reshape(probabilities_by_class_and_batch,
                                   [-1, vocab_size])

        fish_gate = fish_modules.FishGate(hidden_size=vocab_size,
                                          k=filter_size,
                                          dropout_rate=0.9,
                                          is_training=is_training)

        probabilities = fish_gate.forward(probabilities)

        # probabilities = tf.layers.dense(probabilities, vocab_size, use_bias=True, activation=tf.nn.softmax)

        return {"predictions": probabilities}


class FishMoeModel4(models.BaseModel):
    """A softmax over a mixture of logistic models (with L2 regularization)."""

    def create_model(self,
                     model_input,
                     vocab_size,
                     is_training,
                     num_mixtures=None,
                     l2_penalty=1e-8,
                     filter_size=2,
                     **unused_params):
        """Creates a Mixture of (Logistic) Experts model.
         It also includes the possibility of gating the probabilities
         The model consists of a per-class softmax distribution over a
         configurable number of logistic classifiers. One of the classifiers in the
         mixture is not trained, and always predicts 0.
        Args:
          model_input: 'batch_size' x 'num_features' matrix of input features.
          vocab_size: The number of classes in the dataset.
          is_training: Is this the training phase ?
          num_mixtures: The number of mixtures (excluding a dummy 'expert' that
            always predicts the non-existence of an entity).
          l2_penalty: How much to penalize the squared magnitudes of parameter
            values.
        Returns:
          A dictionary with a tensor containing the probability predictions of the
          model in the 'predictions' key. The dimensions of the tensor are
          batch_size x num_classes.
        """
        num_mixtures = num_mixtures or FLAGS.moe_num_mixtures
        l2_penalty = FLAGS.moe_l2

        fc1 = slim.fully_connected(
            model_input, vocab_size, activation_fn=tf.nn.relu, weights_regularizer=slim.l2_regularizer(l2_penalty))
        fc1 = tf.layers.batch_normalization(fc1, training=is_training)
        if is_training:
            fc1 = tf.nn.dropout(fc1, keep_prob=0.9)

        fc2 = slim.fully_connected(
            fc1, vocab_size, activation_fn=tf.nn.relu, weights_regularizer=slim.l2_regularizer(l2_penalty))
        fc2 = tf.layers.batch_normalization(fc2, training=is_training)
        if is_training:
            fc2 = tf.nn.dropout(fc2, keep_prob=0.9)

        fc3 = slim.fully_connected(
            fc2, vocab_size, activation_fn=tf.nn.sigmoid, weights_regularizer=slim.l2_regularizer(l2_penalty))
        fc3 = tf.layers.batch_normalization(fc3, training=is_training)
        if is_training:
            fc3 = tf.nn.dropout(fc3, keep_prob=0.9)

        fish_gate = fish_modules.FishGate(hidden_size=vocab_size,
                                          k=filter_size,
                                          dropout_rate=0.9,
                                          is_training=is_training)

        probabilities = fish_gate.forward(fc3)

        # probabilities = tf.layers.dense(probabilities, vocab_size, use_bias=True, activation=tf.nn.softmax)

        return {"predictions": probabilities}


class FishMoeModel3(models.BaseModel):
    """A softmax over a mixture of logistic models (with L2 regularization)."""

    def create_model(self,
                     model_input,
                     vocab_size,
                     is_training,
                     num_mixtures=None,
                     l2_penalty=1e-6,
                     filter_size=2,
                     **unused_params):
        """Creates a Mixture of (Logistic) Experts model.
         It also includes the possibility of gating the probabilities
         The model consists of a per-class softmax distribution over a
         configurable number of logistic classifiers. One of the classifiers in the
         mixture is not trained, and always predicts 0.
        Args:
          model_input: 'batch_size' x 'num_features' matrix of input features.
          vocab_size: The number of classes in the dataset.
          is_training: Is this the training phase ?
          num_mixtures: The number of mixtures (excluding a dummy 'expert' that
            always predicts the non-existence of an entity).
          l2_penalty: How much to penalize the squared magnitudes of parameter
            values.
        Returns:
          A dictionary with a tensor containing the probability predictions of the
          model in the 'predictions' key. The dimensions of the tensor are
          batch_size x num_classes.
        """
        num_mixtures = num_mixtures or FLAGS.moe_num_mixtures
        l2_penalty = FLAGS.moe_l2

        gate_activations = slim.fully_connected(
            model_input,
            vocab_size * (num_mixtures + 1),
            activation_fn=None,
            biases_initializer=None,
            weights_regularizer=slim.l2_regularizer(l2_penalty),
            scope="gates")

        expert_activations = slim.fully_connected(
            model_input,
            vocab_size * num_mixtures,
            activation_fn=None,
            weights_regularizer=slim.l2_regularizer(l2_penalty),
            scope="experts")

        gating_distribution = tf.nn.softmax(tf.reshape(
            gate_activations,
            [-1, num_mixtures + 1]))  # (Batch * #Labels) x (num_mixtures + 1)
        expert_distribution = tf.nn.sigmoid(tf.reshape(
            expert_activations,
            [-1, num_mixtures]))  # (Batch * #Labels) x num_mixtures

        probabilities_by_class_and_batch = tf.reduce_sum(
            gating_distribution[:, :num_mixtures] * expert_distribution, 1)
        probabilities0 = tf.reshape(probabilities_by_class_and_batch,
                                    [-1, vocab_size])
        probabilities0 = tf.layers.batch_normalization(probabilities0, training=is_training)

        r_activation0 = tf.layers.dense(probabilities0, vocab_size * filter_size, use_bias=True, activation=tf.nn.relu)
        r_activation0 = tf.layers.batch_normalization(r_activation0, training=is_training)
        if is_training:
            r_activation0 = tf.layers.dropout(r_activation0, 0.9)
        r_activation1 = tf.layers.dense(r_activation0, vocab_size, use_bias=True, activation=None)

        probabilities1 = probabilities0 + r_activation1
        probabilities1 = tf.contrib.layers.layer_norm(probabilities1)
        probabilities1 = tf.layers.batch_normalization(probabilities1, training=is_training)
        probabilities2 = tf.layers.dense(probabilities1, vocab_size, use_bias=True, activation=tf.nn.softmax)

        return {"predictions": probabilities2}


class MoeModel2(models.BaseModel):
    """A softmax over a mixture of logistic models (with L2 regularization)."""

    def create_model(self,
                     model_input,
                     vocab_size,
                     is_training,
                     num_mixtures=None,
                     l2_penalty=1e-8,
                     **unused_params):
        """Creates a Mixture of (Logistic) Experts model.
         It also includes the possibility of gating the probabilities
         The model consists of a per-class softmax distribution over a
         configurable number of logistic classifiers. One of the classifiers in the
         mixture is not trained, and always predicts 0.
        Args:
          model_input: 'batch_size' x 'num_features' matrix of input features.
          vocab_size: The number of classes in the dataset.
          is_training: Is this the training phase ?
          num_mixtures: The number of mixtures (excluding a dummy 'expert' that
            always predicts the non-existence of an entity).
          l2_penalty: How much to penalize the squared magnitudes of parameter
            values.
        Returns:
          A dictionary with a tensor containing the probability predictions of the
          model in the 'predictions' key. The dimensions of the tensor are
          batch_size x num_classes.
        """
        num_mixtures = 3
        low_rank_gating = FLAGS.moe_low_rank_gating
        l2_penalty = FLAGS.moe_l2
        gating_probabilities = FLAGS.moe_prob_gating
        gating_input = FLAGS.moe_prob_gating_input

        if low_rank_gating == -1:
            gate_activations = slim.fully_connected(
                model_input,
                vocab_size * (num_mixtures + 1),
                activation_fn=None,
                biases_initializer=None,
                weights_regularizer=slim.l2_regularizer(l2_penalty),
                scope="gates")
        else:
            gate_activations1 = slim.fully_connected(
                model_input,
                low_rank_gating,
                activation_fn=None,
                biases_initializer=None,
                weights_regularizer=slim.l2_regularizer(l2_penalty),
                scope="gates1")
            gate_activations = slim.fully_connected(
                gate_activations1,
                vocab_size * (num_mixtures + 1),
                activation_fn=None,
                biases_initializer=None,
                weights_regularizer=slim.l2_regularizer(l2_penalty),
                scope="gates2")

        expert_activations = slim.fully_connected(
            model_input,
            vocab_size * num_mixtures,
            activation_fn=None,
            weights_regularizer=slim.l2_regularizer(l2_penalty),
            scope="experts")

        gating_distribution = tf.nn.softmax(tf.reshape(
            gate_activations,
            [-1, num_mixtures + 1]))  # (Batch * #Labels) x (num_mixtures + 1)
        expert_distribution = tf.nn.sigmoid(tf.reshape(
            expert_activations,
            [-1, num_mixtures]))  # (Batch * #Labels) x num_mixtures

        probabilities_by_class_and_batch = tf.reduce_sum(
            gating_distribution[:, :num_mixtures] * expert_distribution, 1)
        probabilities = tf.reshape(probabilities_by_class_and_batch,
                                   [-1, vocab_size])

        filter1 = tf.layers.dense(probabilities,
                                  vocab_size * 2,
                                  use_bias=True,
                                  activation=tf.nn.relu,
                                  name="v-filter1")
        filter1 = tf.layers.batch_normalization(filter1, training=is_training)

        if is_training:
            filter1 = tf.nn.dropout(filter1, 0.8)

        filter2 = tf.layers.dense(filter1,
                                  vocab_size,
                                  use_bias=False,
                                  activation=None,
                                  name="v-filter2")

        probabilities = probabilities + filter2
        probabilities = tf.nn.relu(probabilities)
        probabilities = tf.layers.batch_normalization(probabilities, training=is_training)

        probabilities = tf.layers.dense(probabilities, vocab_size, use_bias=True,
                                        activation=tf.nn.sigmoid, name="v-final_output")

        return {"predictions": probabilities}


class JuhanMoeModel(models.BaseModel):
    """A softmax over a mixture of logistic models (with L2 regularization)."""

    def create_model(self,
                     model_input,
                     vocab_size,
                     is_training,
                     num_mixtures=None,
                     l2_penalty=1e-8,
                     **unused_params):
        """Creates a Mixture of (Logistic) Experts model.
         The model consists of a per-class softmax distribution over a
         configurable number of logistic classifiers. One of the classifiers in the
         mixture is not trained, and always predicts 0.
        Args:
          model_input: 'batch_size' x 'num_features' matrix of input features.
          vocab_size: The number of classes in the dataset.
          num_mixtures: The number of mixtures (excluding a dummy 'expert' that
            always predicts the non-existence of an entity).
          l2_penalty: How much to penalize the squared magnitudes of parameter
            values.
        Returns:
          A dictionary with a tensor containing the probability predictions of the
          model in the 'predictions' key. The dimensions of the tensor are
          batch_size x num_classes.
        """
        num_mixtures = 3

        gate_activations = slim.fully_connected(
            model_input,
            vocab_size * (num_mixtures + 1),
            activation_fn=None,
            biases_initializer=None,
            weights_regularizer=slim.l2_regularizer(l2_penalty),
            scope="gates")
        expert_activations = slim.fully_connected(
            model_input,
            vocab_size * num_mixtures,
            activation_fn=None,
            weights_regularizer=slim.l2_regularizer(l2_penalty),
            scope="experts")

        gating_distribution = tf.nn.softmax(tf.reshape(
            gate_activations,
            [-1, num_mixtures + 1]))  # (Batch * #Labels) x (num_mixtures + 1)
        expert_distribution = tf.nn.sigmoid(tf.reshape(
            expert_activations,
            [-1, num_mixtures]))  # (Batch * #Labels) x num_mixtures

        final_probabilities_by_class_and_batch = tf.reduce_sum(
            gating_distribution[:, :num_mixtures] * expert_distribution, 1)
        probabilities = tf.reshape(final_probabilities_by_class_and_batch,
                                   [-1, vocab_size])
        if is_training:
            probabilities = tf.nn.dropout(probabilities, 0.8)

        filter1 = tf.layers.dense(probabilities,
                                  vocab_size * 2,
                                  use_bias=True,
                                  activation=tf.nn.leaky_relu,
                                  name="v-filter1")
        filter1 = tf.layers.batch_normalization(filter1, training=is_training)
        if is_training:
            filter1 = tf.nn.dropout(filter1, 0.8)

        filter2 = tf.layers.dense(filter1,
                                  vocab_size,
                                  use_bias=False,
                                  activation=None,
                                  name="v-filter2")

        probabilities = probabilities + filter2
        probabilities = tf.nn.leaky_relu(probabilities)
        probabilities = tf.layers.batch_normalization(probabilities, training=is_training)

        probabilities = tf.layers.dense(probabilities, vocab_size, use_bias=True,
                                        activation=tf.nn.sigmoid, name="v-final_output")

        return {"predictions": probabilities}


class FourLayerBatchNeuralModel(models.BaseModel):
    def create_model(self,
                     model_input,
                     vocab_size,
                     is_training,
                     l2_penalty=1e-7,
                     **unused_params):
        model_input_dim = model_input.get_shape().as_list()[1]
        fc1_weights = tf.get_variable("fc1_weights",
                                      [model_input_dim, vocab_size],
                                      initializer=tf.contrib.layers.xavier_initializer())
        tf.summary.histogram("fc1_weights", fc1_weights)
        fc1_activation = tf.matmul(model_input, fc1_weights)
        fc1_activation = tf.nn.relu(fc1_activation)
        fc1_activation = slim.batch_norm(
            fc1_activation,
            center=True,
            scale=True,
            is_training=is_training,
            scope="fc1_activation_bn")

        fc2_weights = tf.get_variable("fc2_weights",
                                      [vocab_size, vocab_size],
                                      initializer=tf.contrib.layers.xavier_initializer())
        tf.summary.histogram("fc2_weights", fc2_weights)
        fc2_activation = tf.matmul(fc1_activation, fc2_weights)
        fc2_activation = tf.nn.relu(fc2_activation)
        fc2_activation = slim.batch_norm(
            fc2_activation,
            center=True,
            scale=True,
            is_training=is_training,
            scope="fc2_activation_bn")

        fc3_weights = tf.get_variable("fc3_weights",
                                      [vocab_size, vocab_size],
                                      initializer=tf.contrib.layers.xavier_initializer())
        tf.summary.histogram("fc3_weights", fc3_weights)
        fc3_activation = tf.matmul(fc2_activation, fc3_weights)
        fc3_activation = tf.nn.relu(fc3_activation)
        fc3_activation = slim.batch_norm(
            fc3_activation,
            center=True,
            scale=True,
            is_training=is_training,
            scope="fc3_activation_bn")

        fc4_weights = tf.get_variable("fc4_weights",
                                      [vocab_size, vocab_size],
                                      initializer=tf.contrib.layers.xavier_initializer())
        fc4_activation = tf.matmul(fc3_activation, fc4_weights)
        cluster_biases = tf.get_variable("fc4_bias",
                                         [vocab_size],
                                         initializer=tf.constant_initializer(0.01))
        tf.summary.histogram("fc4_bias", cluster_biases)
        fc4_activation += cluster_biases

        fc4_activation = tf.sigmoid(fc4_activation)

        return {"predictions": fc4_activation}


class ClassLearningThreeNnModel(models.BaseModel):
    def create_model(self,
                     model_input,
                     vocab_size,
                     is_training,
                     l2_penalty=1e-8,
                     ortho_reg=0,
                     **unused_params):
        fc1 = slim.fully_connected(
            model_input, vocab_size, activation_fn=None, biases_initializer=None,
            weights_regularizer=slim.l2_regularizer(l2_penalty))
        fc1 = tf.contrib.layers.layer_norm(inputs=fc1, center=True, scale=True, activation_fn=tf.nn.leaky_relu)
        if is_training:
            fc1 = tf.nn.dropout(fc1, keep_prob=0.5)

        fc2 = slim.fully_connected(
            fc1, vocab_size, activation_fn=None, biases_initializer=None,
            weights_regularizer=slim.l2_regularizer(l2_penalty))
        fc2 = tf.contrib.layers.layer_norm(inputs=fc2, center=True, scale=True, activation_fn=tf.nn.leaky_relu)
        if is_training:
            fc2 = tf.nn.dropout(fc2, keep_prob=0.5)

        fc3 = slim.fully_connected(
            fc2, vocab_size, activation_fn=tf.nn.sigmoid, biases_initializer=tf.constant_initializer(0.1),
            weights_regularizer=slim.l2_regularizer(l2_penalty))

        return {"predictions": fc3,
                "regularization_loss": ortho_reg}


class ClassLearningFourNnModel(models.BaseModel):
    def create_model(self,
                     model_input,
                     vocab_size,
                     is_training,
                     l2_penalty=1e-8,
                     ortho_reg=0,
                     **unused_params):
        fc1 = slim.fully_connected(
            model_input, vocab_size, activation_fn=None, biases_initializer=None,
            weights_regularizer=slim.l2_regularizer(l2_penalty))
        fc1 = tf.contrib.layers.layer_norm(inputs=fc1, center=True, scale=True, activation_fn=tf.nn.leaky_relu)
        # if is_training:
        #     fc1 = tf.nn.dropout(fc1, keep_prob=0.5)

        fc2 = slim.fully_connected(
            fc1, vocab_size, activation_fn=None, biases_initializer=None,
            weights_regularizer=slim.l2_regularizer(l2_penalty))
        fc2 = tf.contrib.layers.layer_norm(inputs=fc2, center=True, scale=True, activation_fn=tf.nn.leaky_relu)
        # if is_training:
        #     fc2 = tf.nn.dropout(fc2, keep_prob=0.5)

        fc3 = slim.fully_connected(
            fc2, vocab_size, activation_fn=None, biases_initializer=None,
            weights_regularizer=slim.l2_regularizer(l2_penalty))
        fc3 = tf.contrib.layers.layer_norm(inputs=fc3, center=True, scale=True, activation_fn=tf.nn.leaky_relu)

        fc4 = slim.fully_connected(
            fc3, vocab_size, activation_fn=tf.nn.sigmoid, biases_initializer=tf.constant_initializer(0.1),
            weights_regularizer=slim.l2_regularizer(l2_penalty))

        return {"predictions": fc4,
                "regularization_loss": ortho_reg}