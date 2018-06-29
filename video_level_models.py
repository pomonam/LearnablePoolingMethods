# Copyright 2016 Google Inc. All Rights Reserved.
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
import tensorflow as tf
import tensorflow.contrib.slim as slim
import models

FLAGS = flags.FLAGS
flags.DEFINE_integer(
    "moe_num_mixtures", 2,
    "The number of mixtures (excluding the dummy 'expert') used for MoeModel.")


###############################################################################
# Baseline (Benchmark) models #################################################
###############################################################################
# Flags for WillowModel
flags.DEFINE_float(
    "moe_l2", 1e-6,
    "L2 penalty for MoeModel.")
flags.DEFINE_integer(
    "moe_low_rank_gating", -1,
    "Low rank gating for MoeModel.")
flags.DEFINE_bool(
    "moe_prob_gating", True,
    "Prob gating for MoeModel.")


class WillowMoeModel(models.BaseModel):
    """A softmax over a mixture of logistic models (with L2 regularization)."""
    def create_model(self,
                     model_input,
                     vocab_size,
                     is_training,
                     num_mixtures=None,
                     l2_penalty=1e-8,
                     det_reg=None,
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
            gating = attention_modules.ContextGateV1(vocab_size, batch_norm=True, is_training=is_training)
            probabilities = gating.forward(probabilities)

        if det_reg is not None:
            return {"predictions": probabilities,
                    "regularization_loss": det_reg}
        else:
            return {"predictions": probabilities}


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