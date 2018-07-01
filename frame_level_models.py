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

"""Contains a collection of models which operate on variable-length sequences."""
# noinspection PyUnresolvedReferences
import pathmagic
from tensorflow import flags
import tensorflow.contrib.layers as layers
import tensorflow as tf
import model_utils as utils
import tensorflow.contrib.slim as slim
import video_pooling_modules
import attention_modules
import aggregation_modules
import loupe_modules
import video_level_models
import rnn_modules
import math
import models


###############################################################################
# Necessary FLAGS #############################################################
###############################################################################
FLAGS = flags.FLAGS
flags.DEFINE_integer("iterations", 30,
                     "Number of frames per batch for DBoF.")
flags.DEFINE_bool("dbof_add_batch_norm", True,
                  "Adds batch normalization to the DBoF model.")
flags.DEFINE_bool(
    "sample_random_frames", True,
    "If true samples random frames (for frame level models). If false, a random"
    "sequence of frames is sampled instead.")
flags.DEFINE_integer("dbof_cluster_size", 8192,
                     "Number of units in the DBoF cluster layer.")
flags.DEFINE_integer("dbof_hidden_size", 1024,
                     "Number of units in the DBoF hidden layer.")
flags.DEFINE_string("dbof_pooling_method", "max",
                    "The pooling method used in the DBoF cluster layer. "
                    "Choices are 'average' and 'max'.")
flags.DEFINE_string("video_level_classifier_model", "MoeModel",
                    "Some Frame-Level models can be decomposed into a "
                    "generalized pooling operation followed by a "
                    "classifier layer")


###############################################################################
# Triangulation Prototype models ##############################################
###############################################################################

# NOTE: These are the best achievable parameters for a P100 GPU (16gb RAM), V100 are to be decided...
#
# All flags start with 'sftm_' to differentiate from original flags.
flags.DEFINE_integer("sftm_iterations", 64,
                     "Number of frames per batch.")
flags.DEFINE_bool("sftm_add_batch_norm", True,
                     "Add batch normalization.")
flags.DEFINE_bool("sftm_sample_random_frames", True,
                  "Iff true, sftm samples random frames.")
flags.DEFINE_integer("sftm_video_anchor_size", 128,
                     "Number of anchors for video features.")
flags.DEFINE_integer("sftm_audio_anchor_size", 16,
                     "Number of anchors for audio features.")
flags.DEFINE_integer("sftm_video_bottleneck", 100,
                     "Size of bottleneck weights for video features.")
flags.DEFINE_integer("sftm_audio_bottleneck", 16,
                     "Size of bottleneck weights for audio features.")
flags.DEFINE_string("sftm_video_level_classifier_model", "MoeModel",
                    "Some Frame-Level models can be decomposed into a "
                    "generalized pooling operation followed by a "
                    "classifier layer")


class SoftAttentionTriangulationModel(models.BaseModel):
    def create_model(self,
                     model_input,
                     vocab_size,
                     num_frames,
                     iterations=None,
                     add_batch_norm=None,
                     sample_random_frames=None,
                     hidden_size=None,
                     is_training=True,
                     **unused_params):
        iterations = iterations or FLAGS.sftm_iterations
        add_batch_norm = add_batch_norm or FLAGS.sftm_add_batch_norm
        video_anchor_size = FLAGS.sftm_video_anchor_size
        audio_anchor_size = FLAGS.sftm_audio_anchor_size
        video_bottleneck = FLAGS.sftm_video_bottleneck
        audio_bottleneck = FLAGS.sftm_audio_bottleneck

        num_frames      = tf.cast(tf.expand_dims(num_frames, 1), tf.float32)
        model_input     = utils.SampleRandomFrames(model_input, num_frames, iterations)
        # model_input: batch_size x max_frames x feature_size
        max_frames      = model_input.get_shape().as_list()[1]
        feature_size    = model_input.get_shape().as_list()[2]
        # model_input: (batch_size * max_frames) x feature_size
        reshaped_input  = tf.reshape(model_input, [-1, feature_size])

        video_features = reshaped_input[:, 0:1024]
        audio_features = reshaped_input[:, 1024:]

        if add_batch_norm:
            video_features = slim.batch_norm(
                video_features,
                center=True,
                scale=True,
                is_training=is_training,
                scope="video_bn")
            audio_features = slim.batch_norm(
                audio_features,
                center=True,
                scale=True,
                is_training=is_training,
                scope="audio_bn")

        # Initialize all modules.
        video_d_module = video_pooling_modules.TriangulationEmbedding(1024,
                                                                      max_frames,
                                                                      video_anchor_size,
                                                                      add_batch_norm,
                                                                      is_training)
        audio_d_module = video_pooling_modules.TriangulationEmbedding(128,
                                                                      max_frames,
                                                                      audio_anchor_size,
                                                                      add_batch_norm,
                                                                      is_training)
        cluster_pool = aggregation_modules.IndirectClusterMaxMeanPoolModule(l2_normalize=False)
        video_t_module = video_pooling_modules.TriangulationTemporalEmbedding(1024,
                                                                              max_frames,
                                                                              video_anchor_size,
                                                                              add_batch_norm,
                                                                              is_training)
        audio_t_module = video_pooling_modules.TriangulationTemporalEmbedding(128,
                                                                              max_frames,
                                                                              audio_anchor_size,
                                                                              add_batch_norm,
                                                                              is_training)

        with tf.variable_scope("video_triangulation_embedding"):
            video_d = video_d_module.forward(video_features)
            # -> batch_size x max_frames x (feature_size * anchor_size)
            video_t = video_t_module.forward(video_d)

            # -> batch_size x max_frames x (feature_size * anchor_size)
            video_d     = tf.reshape(video_d, [-1, max_frames, 1024 * video_anchor_size])

            agg_video_d = cluster_pool.forward(video_d)
            agg_video_t = cluster_pool.forward(video_t)

        with tf.variable_scope("audio_triangulation_embedding"):
            audio_d = audio_d_module.forward(audio_features)
            # -> batch_size x max_frames x (feature_size * anchor_size)
            audio_t = audio_t_module.forward(audio_d)

            # -> batch_size x max_frames x (feature_size * anchor_size)
            audio_d = tf.reshape(audio_d, [-1, max_frames, 128 * audio_anchor_size])

            agg_audio_d = cluster_pool.forward(audio_d)
            agg_audio_t = cluster_pool.forward(audio_t)

        # Projection to a lower dimension space for better discrimination.
        agg_video_dim = agg_video_d.get_shape().as_list()[1]
        video_d_projection = tf.get_variable("video_d_projection",
                                             [agg_video_dim, video_bottleneck],
                                             initializer=tf.random_normal_initializer(
                                                 stddev=1 / math.sqrt(video_bottleneck)))
        video_t_projection = tf.get_variable("video_t_projection",
                                             [agg_video_dim, video_bottleneck],
                                             initializer=tf.random_normal_initializer(
                                                 stddev=1 / math.sqrt(video_bottleneck)))
        agg_audio_dim = agg_audio_d.get_shape().as_list()[1]
        audio_d_projection = tf.get_variable("audio_d_projection",
                                             [agg_audio_dim, audio_bottleneck],
                                             initializer=tf.random_normal_initializer(
                                                 stddev=1 / math.sqrt(audio_bottleneck)))
        audio_t_projection = tf.get_variable("audio_t_projection",
                                             [agg_audio_dim, audio_bottleneck],
                                             initializer=tf.random_normal_initializer(
                                                 stddev=1 / math.sqrt(audio_bottleneck)))

        video_d_activation = tf.matmul(agg_video_d, video_d_projection)
        video_t_activation = tf.matmul(agg_video_t, video_t_projection)
        audio_d_activation = tf.matmul(agg_audio_d, audio_d_projection)
        audio_t_activation = tf.matmul(agg_audio_t, audio_t_projection)

        if add_batch_norm:
            video_d_activation = slim.batch_norm(
                video_d_activation,
                center=True,
                scale=True,
                is_training=is_training,
                scope="video_d_activation_bn")
            video_t_activation = slim.batch_norm(
                video_t_activation,
                center=True,
                scale=True,
                is_training=is_training,
                scope="video_t_activation_bn")
            audio_d_activation = slim.batch_norm(
                audio_d_activation,
                center=True,
                scale=True,
                is_training=is_training,
                scope="audio_d_activation_bn")
            audio_t_activation = slim.batch_norm(
                audio_t_activation,
                center=True,
                scale=True,
                is_training=is_training,
                scope="audio_t_activation_bn")

        # Fuse distribution and temporal features.
        video_activation = tf.concat([video_d_activation, video_t_activation], 1)
        audio_activation = tf.concat([audio_d_activation, audio_t_activation], 1)

        video_activation_dim = video_activation.get_shape().as_list()[1]
        video_activation_weights = tf.get_variable("video_projection",
                                            [video_activation_dim, video_bottleneck],
                                            initializer=tf.random_normal_initializer(
                                                 stddev=1 / math.sqrt(video_bottleneck)))
        audio_activation_dim = audio_activation.get_shape().as_list()[1]
        audio_activation_weights = tf.get_variable("audio_projection",
                                                   [audio_activation_dim, audio_bottleneck],
                                                   initializer=tf.random_normal_initializer(
                                                       stddev=1 / math.sqrt(audio_bottleneck)))

        video_activation = tf.matmul(video_activation, video_activation_weights)
        audio_activation = tf.matmul(audio_activation, audio_activation_weights)

        if add_batch_norm:
            video_activation = slim.batch_norm(
                video_activation,
                center=True,
                scale=True,
                is_training=is_training,
                scope="video_activation_bn")
            audio_activation = slim.batch_norm(
                audio_activation,
                center=True,
                scale=True,
                is_training=is_training,
                scope="audio_activation_bn")

        # Fuse video and audio features.
        activation = tf.concat([video_activation, audio_activation], 1)

        aggregated_model = getattr(video_level_models,
                                   "ClassLearningFourNnModel")
        return aggregated_model().create_model(
            model_input=activation,
            vocab_size=vocab_size,
            is_training=is_training,
            **unused_params)


class RegularizedTriangulationModel(models.BaseModel):
    def create_model(self,
                     model_input,
                     vocab_size,
                     num_frames,
                     iterations=None,
                     add_batch_norm=None,
                     sample_random_frames=None,
                     hidden_size=None,
                     is_training=True,
                     **unused_params):
        iterations = iterations or FLAGS.iterations
        video_anchor_size = FLAGS.wtm_video_anchor_size
        audio_anchor_size = FLAGS.wtm_audio_anchor_size

        num_frames = tf.cast(tf.expand_dims(num_frames, 1), tf.float32)
        model_input = utils.SampleRandomFrames(model_input, num_frames,
                                               iterations)

        # model_input: batch_size x max_frames x feature_size
        max_frames = model_input.get_shape().as_list()[1]
        feature_size = model_input.get_shape().as_list()[2]
        # model_input: (batch_size * max_frames) x feature_size
        reshaped_input = tf.reshape(model_input, [-1, feature_size])

        video_d_module = video_pooling_modules.WeightedTriangulationEmbedding(1024,
                                                                              max_frames,
                                                                              video_anchor_size,
                                                                              add_batch_norm,
                                                                              is_training)
        audio_d_module = video_pooling_modules.WeightedTriangulationEmbedding(128,
                                                                              max_frames,
                                                                              audio_anchor_size,
                                                                              add_batch_norm,
                                                                              is_training)
        mean_max_pool = aggregation_modules.MaxMeanPoolingModule(l2_normalize=False)
        video_t_module = video_pooling_modules.TriangulationTemporalEmbedding(1024,
                                                                              max_frames,
                                                                              video_anchor_size,
                                                                              add_batch_norm,
                                                                              is_training)
        audio_t_module = video_pooling_modules.TriangulationTemporalEmbedding(128,
                                                                              max_frames,
                                                                              audio_anchor_size,
                                                                              add_batch_norm,
                                                                              is_training)
        reshaped_input = slim.batch_norm(
            reshaped_input,
            center=True,
            scale=True,
            is_training=is_training,
            scope="input_bn")

        with tf.variable_scope("video_t_emb"):
            video_d, ortho_reg_v = video_d_module.forward(reshaped_input[:, 0:1024])
            # -> batch_size x max_frames x (feature_size * anchor_size)
            video_t = video_t_module.forward(video_d)

            agg_video_d = mean_max_pool.forward(video_d)
            agg_video_t = mean_max_pool.forward(video_t)

        with tf.variable_scope("audio_t_emb"):
            audio_d, ortho_reg_a = audio_d_module.forward(reshaped_input[:, 1024:])
            # -> batch_size x max_frames x (feature_size * anchor_size)
            video_t = audio_t_module.forward(audio_d)

            agg_audio_d = mean_max_pool.forward(audio_d)
            agg_audio_t = mean_max_pool.forward(video_t)

        orthogonal_reg = ortho_reg_v + ortho_reg_a

        # Video Distribution Projection.
        agg_video_d_dim = agg_video_d.get_shape().as_list()[1]
        video_projection_weight = tf.get_variable("video_projection",
                                                  [agg_video_d_dim, 1024],
                                                  initializer=tf.random_normal_initializer(
                                                      stddev=1 / math.sqrt(1024)))
        video_projection_activation = tf.matmul(agg_video_d, video_projection_weight)
        video_projection_activation = slim.batch_norm(
            video_projection_activation,
            center=True,
            scale=True,
            is_training=is_training,
            scope="video_projection_bn")

        # Audio Distribution Projection.
        agg_audio_d_dim = agg_audio_d.get_shape().as_list()[1]
        audio_projection_weights_1 = tf.get_variable("audio_projection",
                                                  [agg_audio_d_dim, 128],
                                                  initializer=tf.random_normal_initializer(
                                                      stddev=1 / math.sqrt(128)))
        audio_projection_activation = tf.matmul(agg_audio_d_dim, audio_projection_weights_1)
        audio_projection_activation = slim.batch_norm(
            audio_projection_activation,
            center=True,
            scale=True,
            is_training=is_training,
            scope="audio_projection_bn")

        dis_projection_activation = tf.concat([video_projection_activation,
                                               audio_projection_activation], 1)

        # Temporal Features Projection.
        agg_temp = tf.concat([agg_video_t, agg_audio_t], 1)
        agg_temp_dim = agg_temp.get_shape().as_list()[1]
        temp_projection_weight = tf.get_variable("temp_projection_1",
                                                 [agg_temp_dim, 1152],
                                                 initializer=tf.random_normal_initializer(
                                                     stddev=1 / math.sqrt(1152)))
        temp_projection_activation = tf.matmul(agg_temp, temp_projection_weight)
        temp_projection_activation = slim.batch_norm(
            temp_projection_activation,
            center=True,
            scale=True,
            is_training=is_training,
            scope="temp_projection_bn")


        # Higher dimension projection.
        dis_projection_activation_dim = dis_projection_activation.get_shape().as_list()[1]
        dis_projection_weights = tf.get_variable("dis_projection_2",
                                                 [dis_projection_activation_dim, 2048],
                                                 initializer=tf.random_normal_initializer(
                                                     stddev=1 / math.sqrt(2048)),
                                                 regularizer=layers.l1_l2_regularizer(1e-5))
        dis_activation = tf.matmul(dis_projection_activation, dis_projection_weights)
        dis_activation = slim.batch_norm(
            dis_activation,
            center=True,
            scale=True,
            is_training=is_training,
            scope="dis_activation_bn")
        # dis_activation = tf.nn.leaky_relu(dis_activation)

        temp_projection_activation_dim = temp_projection_activation.get_shape().as_list()[1]
        temp_projection_weights_2 = tf.get_variable("temp_projection_2",
                                                 [temp_projection_activation_dim, 2048],
                                                 initializer=tf.random_normal_initializer(
                                                     stddev=1 / math.sqrt(2048)),
                                                 regularizer=layers.l1_l2_regularizer(1e-5))
        temp_activation = tf.matmul(temp_projection_activation, temp_projection_weights_2)
        temp_activation = slim.batch_norm(
            temp_activation,
            center=True,
            scale=True,
            is_training=is_training,
            scope="temp_activation_bn")
        # temp_activation = tf.nn.leaky_relu(temp_activation)

        activation = tf.concat([dis_activation, temp_activation], 1)

        aggregated_model = getattr(video_level_models,
                                   "ClassLearningThreeNnModel")

        return aggregated_model().create_model(
            model_input=activation,
            vocab_size=vocab_size,
            is_training=is_training,
            ortho_reg=orthogonal_reg,
            **unused_params)




# Flags
flags.DEFINE_integer("wtm_video_anchor_size", 64,
                     "Size for anchor points for video features.")
flags.DEFINE_integer("wtm_audio_anchor_size", 64,
                     "Size for anchor points for audio features.")


class WeightedTriangulationModel(models.BaseModel):
    def create_model(self,
                     model_input,
                     vocab_size,
                     num_frames,
                     iterations=None,
                     add_batch_norm=None,
                     sample_random_frames=None,
                     hidden_size=None,
                     is_training=True,
                     **unused_params):
        iterations = iterations or FLAGS.iterations
        random_frames = sample_random_frames or FLAGS.sample_random_frames
        add_batch_norm = add_batch_norm or FLAGS.batch_norm
        video_anchor_size = FLAGS.wtm_video_anchor_size
        audio_anchor_size = FLAGS.wtm_audio_anchor_size

        if random_frames:
            num_frames_2 = tf.cast(tf.expand_dims(num_frames, 1), tf.float32)
            model_input = utils.SampleRandomFrames(model_input, num_frames_2,
                                                   iterations)

        # model_input: batch_size x max_frames x feature_size
        max_frames = model_input.get_shape().as_list()[1]
        feature_size = model_input.get_shape().as_list()[2]
        # model_input: (batch_size * max_frames) x feature_size
        reshaped_input = tf.reshape(model_input, [-1, feature_size])

        video_t_emb = video_pooling_modules.WeightedTriangulationEmbedding(1024,
                                                                           max_frames,
                                                                           video_anchor_size,
                                                                           add_batch_norm,
                                                                           is_training)
        audio_t_emb = video_pooling_modules.WeightedTriangulationEmbedding(128,
                                                                   max_frames,
                                                                   audio_anchor_size,
                                                                   add_batch_norm,
                                                                   is_training)

        mean_pool = aggregation_modules.SpocPoolingModule(l2_normalize=False)

        video_t_temp_emb = video_pooling_modules.TriangulationTemporalEmbedding(1024, max_frames,
                                                                                video_anchor_size,
                                                                                add_batch_norm,
                                                                                is_training)
        audio_t_temp_emb = video_pooling_modules.TriangulationTemporalEmbedding(128, max_frames,
                                                                                audio_anchor_size,
                                                                                add_batch_norm,
                                                                                is_training)

        if add_batch_norm:
            reshaped_input = slim.batch_norm(
                reshaped_input,
                center=True,
                scale=True,
                is_training=is_training,
                scope="input_bn")

        with tf.variable_scope("video_t_emb"):
            t_emb_video, det_reg_v = video_t_emb.forward(reshaped_input[:, 0:1024])
            # -> batch_size x max_frames x (feature_size * cluster_size)
            t_emb_temp_video = video_t_temp_emb.forward(t_emb_video)

            v_distrib_pool = mean_pool.forward(t_emb_video)
            v_temp_pool = mean_pool.forward(t_emb_temp_video)

        with tf.variable_scope("audio_t_emb"):
            t_emb_audio, det_reg_a = audio_t_emb.forward(reshaped_input[:, 1024:])
            # -> batch_size x max_frames x (feature_size * cluster_size)
            t_emb_temp_audio = audio_t_temp_emb.forward(t_emb_audio)

            a_distrib_pool = mean_pool.forward(t_emb_audio)
            a_temp_pool = mean_pool.forward(t_emb_temp_audio)

        if det_reg_v is not None and det_reg_a is not None:
            det_reg = det_reg_v + det_reg_a
        else:
            det_reg = 0

        video_concat = tf.concat([v_distrib_pool, v_temp_pool], 1)

        video_concat_dim = video_concat.get_shape().as_list()[1]
        video_hidden_1 = tf.get_variable("video_hidden_1",
                                       [video_concat_dim, 1024],
                                       initializer=tf.random_normal_initializer(
                                           stddev=1 / math.sqrt(1024)))

        video_hidden_2 = tf.get_variable("video_hidden_2",
                                         [1024, 1024],
                                         initializer=tf.random_normal_initializer(
                                             stddev=1 / math.sqrt(1024)))

        video_activation = tf.matmul(video_concat, video_hidden_1)
        if add_batch_norm:
            video_activation = slim.batch_norm(
                video_activation,
                center=True,
                scale=True,
                is_training=is_training,
                scope="video_activation_1_bn")
        video_activation = tf.nn.leaky_relu(video_activation)

        video_activation = tf.matmul(video_activation, video_hidden_2)
        if add_batch_norm:
            video_activation = slim.batch_norm(
                video_activation,
                center=True,
                scale=True,
                is_training=is_training,
                scope="video_activation_2_bn")
        video_activation = tf.nn.leaky_relu(video_activation)

        audio_concat = tf.concat([a_distrib_pool, a_temp_pool], 1)

        audio_concat_dim = audio_concat.get_shape().as_list()[1]
        audio_hidden_1 = tf.get_variable("audio_hidden_1",
                                       [audio_concat_dim, 128],
                                       initializer=tf.random_normal_initializer(
                                           stddev=1 / math.sqrt(128)))

        audio_hidden_2 = tf.get_variable("audio_hidden_2",
                                       [128, 128],
                                       initializer=tf.random_normal_initializer(
                                           stddev=1 / math.sqrt(128)))

        audio_activation = tf.matmul(audio_concat, audio_hidden_1)
        if add_batch_norm:
            audio_activation = slim.batch_norm(
                audio_activation,
                center=True,
                scale=True,
                is_training=is_training,
                scope="audio_activation_1_bn")
        audio_activation = tf.nn.leaky_relu(audio_activation)

        audio_activation = tf.matmul(audio_activation, audio_hidden_2)
        if add_batch_norm:
            audio_activation = slim.batch_norm(
                audio_activation,
                center=True,
                scale=True,
                is_training=is_training,
                scope="audio_activation_2_bn")
        audio_activation = tf.nn.leaky_relu(audio_activation)

        total_activation = tf.concat([video_activation, audio_activation], 1)

        # Context Gating
        input_dim = total_activation.get_shape().as_list()[1]
        gating_weights = tf.get_variable("gating_weights",
                                         [input_dim, input_dim],
                                         initializer=tf.random_normal_initializer(
                                             stddev=1 / math.sqrt(input_dim)))

        gates = tf.matmul(total_activation, gating_weights)

        if add_batch_norm:
            gates = slim.batch_norm(
                gates,
                center=True,
                scale=True,
                is_training=is_training,
                scope="gating_bn")
        else:
            gating_biases = tf.get_variable("gating_biases",
                                            [input_dim],
                                            initializer=tf.random_normal_initializer(stddev=1 / math.sqrt(input_dim)))
            gates += gating_biases

        gates = tf.sigmoid(gates)
        activation = tf.multiply(total_activation, gates)

        aggregated_model = getattr(video_level_models,
                                   "WillowMoeModel")

        return aggregated_model().create_model(
            model_input=activation,
            vocab_size=vocab_size,
            is_training=is_training,
            det_reg=det_reg,
            **unused_params)


# Flags
flags.DEFINE_bool("batch_norm", True,
                  "True iff add batch normalization.")
flags.DEFINE_integer("audio_triangulation_anchor_size_v1", 4,
                     "Size for anchor points for video features.")
flags.DEFINE_integer("video_triangulation_anchor_size_v1", 16,
                     "Size for anchor points for audio features.")


class TriangulationRelationalModel(models.BaseModel):
    def create_model(self,
                     model_input,
                     vocab_size,
                     num_frames,
                     iterations=None,
                     add_batch_norm=None,
                     sample_random_frames=None,
                     hidden_size=None,
                     is_training=True,
                     **unused_params):
        iterations = iterations or FLAGS.iterations
        random_frames = sample_random_frames or FLAGS.sample_random_frames
        add_batch_norm = add_batch_norm or FLAGS.batch_norm
        video_anchor_size = FLAGS.video_triangulation_anchor_size_v1
        audio_anchor_size = FLAGS.audio_triangulation_anchor_size_v1

        if random_frames:
            num_frames_2 = tf.cast(tf.expand_dims(num_frames, 1), tf.float32)
            model_input = utils.SampleRandomFrames(model_input, num_frames_2,
                                                   iterations)

        # model_input: batch_size x max_frames x feature_size
        max_frames = model_input.get_shape().as_list()[1]
        feature_size = model_input.get_shape().as_list()[2]
        # model_input: (batch_size * max_frames) x feature_size
        reshaped_input = tf.reshape(model_input, [-1, feature_size])

        video_t_emb = video_pooling_modules.TriangulationEmbedding(1024,
                                                                   max_frames,
                                                                   video_anchor_size,
                                                                   add_batch_norm,
                                                                   is_training)
        audio_t_emb = video_pooling_modules.TriangulationEmbedding(128,
                                                                   max_frames,
                                                                   audio_anchor_size,
                                                                   add_batch_norm,
                                                                   is_training)

        video_lstm = rnn_modules.LstmLastHiddenModule(lstm_size=1024 * video_anchor_size,
                                                      lstm_layers=1,
                                                      output_dim=1024 * video_anchor_size,
                                                      num_frames=num_frames,
                                                      scope_id=None)
        audio_lstm = rnn_modules.LstmLastHiddenModule(lstm_size=128 * audio_anchor_size,
                                                      lstm_layers=1,
                                                      output_dim=128 * audio_anchor_size,
                                                      num_frames=num_frames,
                                                      scope_id=None)

        if add_batch_norm:
            reshaped_input = slim.batch_norm(
                reshaped_input,
                center=True,
                scale=True,
                is_training=is_training,
                scope="input_bn")

        with tf.variable_scope("video_t_emb"):
            t_emb_video, det_reg_v = video_t_emb.forward(reshaped_input[:, 0:1024])
            # -> (batch_size * max_frames) x (feature_size * cluster_size)
            t_emb_video = tf.reshape(t_emb_video, [-1, max_frames, 1024 * video_anchor_size])
            lstm_video_output = video_lstm.forward(t_emb_video)

        with tf.variable_scope("audio_t_emb"):
            t_emb_audio, det_reg_a = audio_t_emb.forward(reshaped_input[:, 1024:])
            # -> (batch_size * max_frames) x (feature_size * cluster_size)
            t_emb_audio = tf.reshape(t_emb_audio, [-1, max_frames, 128 * audio_anchor_size])
            lstm_audio_output = audio_lstm.forward(t_emb_audio)

        if det_reg_v is not None and det_reg_a is not None:
            det_reg = det_reg_a + det_reg_v
        else:
            det_reg = 0

        lstm_output = tf.concat([lstm_video_output, lstm_audio_output], 1)
        # -> batch_size * output_dim

        lstm_output_dim = lstm_output.get_shape().as_list()[1]
        lstm_hidden_1 = tf.get_variable("lstm_hidden_1",
                                        [lstm_output_dim, 2048],
                                        initializer=tf.random_normal_initializer(
                                            stddev=1 / math.sqrt(2048)))
        activation = tf.matmul(lstm_output, lstm_hidden_1)
        if add_batch_norm:
            activation = slim.batch_norm(
                activation,
                center=True,
                scale=True,
                is_training=is_training,
                scope="activation_1_bn")
        activation = tf.nn.leaky_relu(activation)
        if is_training:
            activation = tf.nn.dropout(activation, keep_prob=0.5)

        lstm_hidden_2 = tf.get_variable("lstm_hidden_2",
                                        [2048, 2048],
                                        initializer=tf.random_normal_initializer(
                                            stddev=1 / math.sqrt(2048)))
        activation = tf.matmul(activation, lstm_hidden_2)
        if add_batch_norm:
            activation = slim.batch_norm(
                activation,
                center=True,
                scale=True,
                is_training=is_training,
                scope="activation_2_bn")
        activation = tf.nn.leaky_relu(activation)
        if is_training:
            activation = tf.nn.dropout(activation, keep_prob=0.5)

        aggregated_model = getattr(video_level_models,
                                   "WillowMoeModel")

        return aggregated_model().create_model(
            model_input=activation,
            vocab_size=vocab_size,
            is_training=is_training,
            det_reg=det_reg,
            **unused_params)


###############################################################################
# Triangulation Embedding Model V1 ############################################
###############################################################################
# Triangulation Embedding methods Version 1

# Flags
flags.DEFINE_bool("tembed_v1_batch_norm", True,
                  "True iff add batch normalization.")
flags.DEFINE_integer("tembed_v1_video_anchor_size", 64,
                     "Size for anchor points for video features.")
flags.DEFINE_integer("tembed_v1_audio_anchor_size", 32,
                     "Size for anchor points for audio features.")
flags.DEFINE_integer("tembed_v1_video_concat_hidden_size", 1024,
                     "Hidden weights for concatenated (t-embedded video features.")
flags.DEFINE_integer("tembed_v1_audio_concat_hidden_size", 128,
                     "Hidden weights for concatenated (t-embedded audio features.")
flags.DEFINE_integer("tembed_v1_full_concat_hidden_size", 1024,
                     "Hidden weights for concatenated (t-embedded video, audio features.")


class TembedModelV1(models.BaseModel):
    def create_model(self,
                     model_input,
                     vocab_size,
                     num_frames,
                     iterations=None,
                     add_batch_norm=None,
                     sample_random_frames=None,
                     hidden_size=None,
                     is_training=True,
                     **unused_params):
        iterations = iterations or FLAGS.iterations
        random_frames = sample_random_frames or FLAGS.sample_random_frames
        gating = FLAGS.gating
        add_batch_norm = add_batch_norm or FLAGS.tembed_v1_batch_norm
        video_anchor_size = FLAGS.tembed_v1_video_anchor_size
        audio_anchor_size = FLAGS.tembed_v1_audio_anchor_size
        video_concat_hidden_size = FLAGS.tembed_v1_video_concat_hidden_size
        audio_concat_hidden_size = FLAGS.tembed_v1_audio_concat_hidden_size
        full_concat_hidden_size = FLAGS.tembed_v1_full_concat_hidden_size

        num_frames = tf.cast(tf.expand_dims(num_frames, 1), tf.float32)
        if random_frames:
            model_input = utils.SampleRandomFrames(model_input,
                                                   num_frames,
                                                   iterations)
        else:
            model_input = utils.SampleRandomSequence(model_input,
                                                     num_frames,
                                                     iterations)

        # model_input: batch_size x max_frames x feature_size
        max_frames = model_input.get_shape().as_list()[1]
        feature_size = model_input.get_shape().as_list()[2]
        # model_input: (batch_size * max_frames) x feature_size
        reshaped_input = tf.reshape(model_input, [-1, feature_size])

        video_t_emb = video_pooling_modules.TriangulationEmbedding(1024, max_frames,
                                                             video_anchor_size,
                                                             add_batch_norm,
                                                             is_training)
        audio_t_emb = video_pooling_modules.TriangulationEmbedding(128, max_frames,
                                                             audio_anchor_size,
                                                             add_batch_norm,
                                                             is_training)

        video_spoc_pooling = aggregation_modules.SpocPoolingModule(1024, max_frames)
        audio_spoc_pooling = aggregation_modules.SpocPoolingModule(128, max_frames)

        video_t_temp_emb = video_pooling_modules.TriangulationTemporalEmbedding(1024, max_frames,
                                                                      video_anchor_size,
                                                                      add_batch_norm,
                                                                      is_training)
        audio_t_temp_emb = video_pooling_modules.TriangulationTemporalEmbedding(128, max_frames,
                                                                      audio_anchor_size,
                                                                      add_batch_norm,
                                                                      is_training)

        if add_batch_norm:
            reshaped_input = slim.batch_norm(
                reshaped_input,
                center=True,
                scale=True,
                is_training=is_training,
                scope="input_bn")

        with tf.variable_scope("video_t_emb"):
            t_emb_video = video_t_emb.forward(reshaped_input[:, 0:1024])
            # -> (batch_size * max_frames) x (feature_size * cluster_size)
            t_emb_video = tf.reshape(t_emb_video, [-1, max_frames, 1024 * video_anchor_size])
            t_temp_video = video_t_temp_emb.forward(t_emb_video)
            # -> batch_size x (max_frames - 1) x (feature_size * cluster_size)

            t_emb_video = video_spoc_pooling.forward(t_emb_video)
            t_temp_video = video_spoc_pooling.forward(t_temp_video)
            # -> batch_size x (feature_size * cluster_size)

        t_video_concat = tf.concat([t_emb_video, t_temp_video], 1)
        # -> batch_size x (feature_size * cluster_size * 2)
        t_video_concat_dim = t_video_concat.get_shape().as_list()[1]
        video_hidden_1 = tf.get_variable("video_hidden_1",
                                         [t_video_concat_dim, video_concat_hidden_size],
                                         initializer=tf.random_normal_initializer(
                                             stddev=1 / math.sqrt(video_concat_hidden_size)),
                                         dtype=tf.float32)
        video_activation = tf.matmul(t_video_concat, video_hidden_1)
        video_activation = tf.nn.relu6(video_activation)

        with tf.variable_scope("audio_t_emb"):
            t_emb_audio = audio_t_emb.forward(reshaped_input[:, 1024:])
            # -> (batch_size * max_frames) x (feature_size * cluster_size)
            t_emb_audio = tf.reshape(t_emb_audio, [-1, max_frames, 128 * audio_anchor_size])
            t_temp_audio = audio_t_temp_emb.forward(t_emb_audio)
            # -> batch_size x (max_frames - 1) x (feature_size * cluster_size)

            t_emb_audio = audio_spoc_pooling.forward(t_emb_audio)
            t_temp_audio = audio_spoc_pooling.forward(t_temp_audio)
            # -> batch_size x (feature_size * cluster_size)

        t_audio_concat = tf.concat([t_emb_audio, t_temp_audio], 1)
        # -> batch_size x (feature_size * cluster_size * 2)
        t_audio_concat_dim = t_audio_concat.get_shape().as_list()[1]
        audio_hidden_1 = tf.get_variable("audio_hidden_1",
                                         [t_audio_concat_dim, audio_concat_hidden_size],
                                         initializer=tf.random_normal_initializer(
                                             stddev=1 / math.sqrt(audio_concat_hidden_size)))
        audio_activation = tf.matmul(t_audio_concat, audio_hidden_1)
        audio_activation = tf.nn.relu6(audio_activation)

        video_audio_concat = tf.concat([video_activation, audio_activation], 1)

        video_audio_concat_dim = video_audio_concat.get_shape().as_list()[1]
        # -> batch_size x (feature_size * cluster_size)
        hidden1_weights = tf.get_variable("hidden_weights",
                                          [video_audio_concat_dim, full_concat_hidden_size],
                                          initializer=tf.random_normal_initializer(
                                              stddev=1 / math.sqrt(full_concat_hidden_size)))
        activation = tf.matmul(video_audio_concat, hidden1_weights)
        activation = slim.batch_norm(
            activation,
            center=True,
            scale=True,
            is_training=is_training,
            scope="hidden1_bn")
        activation = tf.nn.leaky_relu(activation)

        if gating:
            gating_weights = tf.get_variable("gating_weights_2",
                                             [full_concat_hidden_size, full_concat_hidden_size],
                                             initializer=tf.random_normal_initializer(
                                                 stddev=1 / math.sqrt(full_concat_hidden_size)))
            gates = tf.matmul(activation, gating_weights)

            if add_batch_norm:
                gates = slim.batch_norm(
                    gates,
                    center=True,
                    scale=True,
                    is_training=is_training,
                    scope="gating_bn")
            else:
                gating_biases = tf.get_variable("gating_biases",
                                                [full_concat_hidden_size],
                                                initializer=tf.random_normal(
                                                    stddev=1 / math.sqrt(full_concat_hidden_size)))
                gates += gating_biases
            gates = tf.sigmoid(gates)
            activation = tf.multiply(activation, gates)

        aggregated_model = getattr(video_level_models,
                                   "WillowMoeModel")

        return aggregated_model().create_model(
            model_input=activation,
            vocab_size=vocab_size,
            is_training=is_training,
            **unused_params)


###############################################################################
# Triangulation Embedding Model V2 ############################################
###############################################################################
# Triangulation Embedding methods Version 2

# Flags
flags.DEFINE_bool("tembed_v2_batch_norm", True,
                  "True iff add batch normalization.")
flags.DEFINE_integer("tembed_v2_video_anchor_size", 64,
                     "Size for anchor points for video features.")
flags.DEFINE_integer("tembed_v2_audio_anchor_size", 32,
                     "Size for anchor points for audio features.")
flags.DEFINE_integer("tembed_v2_distrib_concat_hidden_size", 1024,
                     "Hidden weights for concatenated (t-embedded distribution features.")
flags.DEFINE_integer("tembed_v2_temporal_concat_hidden_size", 128,
                     "Hidden weights for concatenated (t-embedded temporal features.")
flags.DEFINE_integer("tembed_v2_full_concat_hidden_size", 2048,
                     "Hidden weights for concatenated (t-embedded video, audio features.")


class TembedModelV2(models.BaseModel):
    def create_model(self,
                     model_input,
                     vocab_size,
                     num_frames,
                     iterations=None,
                     add_batch_norm=None,
                     sample_random_frames=None,
                     cluster_size=None,
                     hidden_size=None,
                     is_training=True,
                     **unused_params):
        iterations = iterations or FLAGS.iterations
        random_frames = sample_random_frames or FLAGS.sample_random_frames
        gating = FLAGS.gating
        add_batch_norm = add_batch_norm or FLAGS.tembed_v2_batch_norm
        video_anchor_size = FLAGS.tembed_v2_video_anchor_size
        audio_anchor_size = FLAGS.tembed_v2_audio_anchor_size
        distrib_concat_hidden_size = FLAGS.tembed_v2_distrib_concat_hidden_size
        temporal_concat_hidden_size = FLAGS.tembed_v2_temporal_concat_hidden_size
        full_concat_hidden_size = FLAGS.tembed_v2_full_concat_hidden_size

        num_frames = tf.cast(tf.expand_dims(num_frames, 1), tf.float32)
        if random_frames:
            model_input = utils.SampleRandomFrames(model_input, num_frames,
                                                   iterations)
        else:
            model_input = utils.SampleRandomSequence(model_input, num_frames,
                                                     iterations)

        # model_input: batch_size x max_frames x feature_size
        max_frames = model_input.get_shape().as_list()[1]
        feature_size = model_input.get_shape().as_list()[2]
        # model_input: (batch_size * max_frames) x feature_size
        reshaped_input = tf.reshape(model_input, [-1, feature_size])

        video_t_emb = video_pooling_modules.TriangulationEmbedding(1024, max_frames,
                                                             video_anchor_size,
                                                             add_batch_norm,
                                                             is_training)
        audio_t_emb = video_pooling_modules.TriangulationEmbedding(128, max_frames,
                                                             audio_anchor_size,
                                                             add_batch_norm,
                                                             is_training)

        video_spoc_pooling = aggregation_modules.SpocPoolingModule(1024, max_frames)
        audio_spoc_pooling = aggregation_modules.SpocPoolingModule(128, max_frames)

        video_t_temp_emb = video_pooling_modules.TriangulationTemporalEmbedding(1024, max_frames,
                                                                      video_anchor_size,
                                                                      add_batch_norm,
                                                                      is_training)
        audio_t_temp_emb = video_pooling_modules.TriangulationTemporalEmbedding(128, max_frames,
                                                                      audio_anchor_size,
                                                                      add_batch_norm,
                                                                      is_training)

        if add_batch_norm:
            reshaped_input = slim.batch_norm(
                reshaped_input,
                center=True,
                scale=True,
                is_training=is_training,
                scope="input_bn")

        with tf.variable_scope("video_t_emb"):
            t_distrib_video = video_t_emb.forward(reshaped_input[:, 0:1024])
            # -> (batch_size * max_frames) x (feature_size * anchor_size)
            t_distrib_video = tf.reshape(t_distrib_video, [-1, max_frames, 1024 * video_anchor_size])
            t_temp_video = video_t_temp_emb.forward(t_distrib_video)
            # -> batch_size x (max_frames - 1) x (feature_size * anchor_size)

            t_distrib_video = video_spoc_pooling.forward(t_distrib_video)
            t_temp_video = video_spoc_pooling.forward(t_temp_video)
            # -> batch_size x (feature_size * cluster_size)

        with tf.variable_scope("audio_t_emb"):
            t_emb_audio = audio_t_emb.forward(reshaped_input[:, 1024:])
            # -> (batch_size * max_frames) x (feature_size * cluster_size)
            t_emb_audio = tf.reshape(t_emb_audio, [-1, max_frames, 128 * audio_anchor_size])
            t_temp_audio = audio_t_temp_emb.forward(t_emb_audio)
            # -> batch_size x (max_frames - 1) x (feature_size * cluster_size)

            t_emb_audio = audio_spoc_pooling.forward(t_emb_audio)
            t_temp_audio = audio_spoc_pooling.forward(t_temp_audio)
            # -> batch_size x (feature_size * cluster_size)

        t_distrib_concat = tf.concat([t_emb_video, t_emb_audio], 1)
        t_distrib_concat_dim = t_distrib_concat.get_shape().as_list()[1]
        t_distrib_hidden = tf.get_variable("distrib_concat",
                                           [t_distrib_concat_dim, distrib_concat_hidden_size],
                                           initializer=tf.random_normal_initializer(
                                               stddev=1 / math.sqrt(distrib_concat_hidden_size)))
        distrib_activation = tf.matmul(t_distrib_concat, t_distrib_hidden)
        distrib_activation = tf.nn.relu6(distrib_activation)

        t_temp_concat = tf.concat([t_temp_video, t_temp_audio], 1)
        t_temp_concat_dim = t_temp_concat.get_shape().as_list()[1]
        t_temp_hidden = tf.get_variable("temp_concat",
                                        [t_temp_concat_dim, temporal_concat_hidden_size],
                                        initializer=tf.random_normal_initializer(
                                            stddev=1 / math.sqrt(temporal_concat_hidden_size)))
        temp_activation = tf.matmul(t_temp_concat, t_temp_hidden)
        temp_activation = tf.nn.relu6(temp_activation)

        t_concat = tf.concat([distrib_activation, temp_activation], 1)
        t_concat_dim = t_concat.get_shape().as_list()[1]
        concat_hidden_1 = tf.get_variable("concat_hidden_1",
                                          [t_concat_dim, full_concat_hidden_size],
                                          initializer=tf.random_normal_initializer(
                                              stddev=1 / math.sqrt(full_concat_hidden_size)))
        activation = tf.matmul(t_concat, concat_hidden_1)
        activation = tf.nn.relu6(activation)

        activation = slim.batch_norm(
            activation,
            center=True,
            scale=True,
            is_training=is_training,
            scope="hidden1_bn")
        activation = tf.nn.relu6(activation)

        if gating:
            gating_weights = tf.get_variable("gating_weights_2",
                                             [full_concat_hidden_size, full_concat_hidden_size],
                                             initializer=tf.random_normal_initializer(
                                                 stddev=1 / math.sqrt(full_concat_hidden_size)))
            gates = tf.matmul(activation, gating_weights)

            if add_batch_norm:
                gates = slim.batch_norm(
                    gates,
                    center=True,
                    scale=True,
                    is_training=is_training,
                    scope="gating_bn")
            else:
                gating_biases = tf.get_variable("gating_biases",
                                                [full_concat_hidden_size],
                                                initializer=tf.random_normal(
                                                    stddev=1 / math.sqrt(full_concat_hidden_size)))
                gates += gating_biases

            gates = tf.sigmoid(gates)
            activation = tf.multiply(activation, gates)

        aggregated_model = getattr(video_level_models,
                                   "WillowMoeModel")

        return aggregated_model().create_model(
            model_input=activation,
            vocab_size=vocab_size,
            is_training=is_training,
            **unused_params)


# Flags
flags.DEFINE_bool("tembed_v5_batch_norm", True,
                  "True iff add batch normalization.")
flags.DEFINE_integer("tembed_v5_video_anchor_size", 16,
                     "Size for anchor points for video features.")
flags.DEFINE_integer("tembed_v5_audio_anchor_size", 4,
                     "Size for anchor points for audio features.")


class TriangulationModelV2(models.BaseModel):
    def create_model(self,
                     model_input,
                     vocab_size,
                     num_frames,
                     iterations=None,
                     add_batch_norm=None,
                     sample_random_frames=None,
                     cluster_size=None,
                     hidden_size=None,
                     is_training=True,
                     **unused_params):
        add_batch_norm = add_batch_norm or FLAGS.tembed_v3_batch_norm
        video_anchor_size = FLAGS.tembed_v4_video_anchor_size
        audio_anchor_size = FLAGS.tembed_v4_audio_anchor_size

        # Do not sub-sample frames in between.

        # model_input: batch_size x max_frames x feature_size
        max_frames = model_input.get_shape().as_list()[1]
        feature_size = model_input.get_shape().as_list()[2]
        reshaped_input = tf.reshape(model_input, [-1, feature_size])
        # -> (batch_size * max_frames) x feature_size

        video_embedding = video_pooling_modules.TriangulationEmbedding(1024,
                                                                       max_frames,
                                                                       video_anchor_size,
                                                                       add_batch_norm,
                                                                       is_training)
        audio_embedding = video_pooling_modules.TriangulationEmbedding(128,
                                                                       max_frames,
                                                                       audio_anchor_size,
                                                                       add_batch_norm,
                                                                       is_training)

        mean_max_pool = aggregation_modules.SpocPoolingModule(l2_normalize=False)

        video_temp_embedding = video_pooling_modules.TriangulationTemporalEmbedding(1024,
                                                                                    max_frames,
                                                                                    video_anchor_size,
                                                                                    add_batch_norm,
                                                                                    is_training)
        audio_temp_embedding = video_pooling_modules.TriangulationTemporalEmbedding(128,
                                                                                    max_frames,
                                                                                    audio_anchor_size,
                                                                                    add_batch_norm,
                                                                                    is_training)
        if add_batch_norm:
            reshaped_input = slim.batch_norm(
                reshaped_input,
                center=True,
                scale=True,
                is_training=is_training,
                scope="input_bn")

        with tf.variable_scope("video_t_emb"):
            v_distribution_embedding = video_embedding.forward(reshaped_input[:, 0:1024])
            # -> (batch_size * max_frames) x (1024 * anchor_size)
            v_distribution_embedding = tf.reshape(v_distribution_embedding, [-1, max_frames, 1024 * video_anchor_size])
            v_temporal_embedding = video_temp_embedding.forward(v_distribution_embedding)
            # -> batch_size x (max_frames - 1) x (1024 * anchor_size)

            agg_v_temporal_embedding = mean_max_pool.forward(v_temporal_embedding)
            agg_v_distrib_embedding = mean_max_pool.forward(v_distribution_embedding)

        with tf.variable_scope("audio_t_emb"):
            a_distribution_embedding = audio_embedding.forward(reshaped_input[:, 1024:])
            # -> (batch_size * max_frames) x (128 * anchor_size)
            a_distribution_embedding = tf.reshape(a_distribution_embedding, [-1, max_frames, 128 * audio_anchor_size])
            a_temporal_embedding = audio_temp_embedding.forward(a_distribution_embedding)
            # -> batch_size x (max_frames - 1) x (128 * cluster_size)

            agg_a_temporal_embedding = mean_max_pool.forward(a_temporal_embedding)
            agg_a_distrib_embedding = mean_max_pool.forward(a_distribution_embedding)

        # Aggregating video features.
        # 1. Temporal features.
        agg_v_temporal_embedding_dim = agg_v_temporal_embedding.get_shape().as_list()[1]
        video_temporal_hidden_weights = tf.get_variable("video_temporal_hidden",
                                            [agg_v_temporal_embedding_dim, 1024],
                                            initializer=tf.random_normal_initializer(
                                               stddev=1 / math.sqrt(768)))
        v_temp_activation = tf.matmul(agg_v_temporal_embedding, video_temporal_hidden_weights)
        if add_batch_norm:
            v_temp_activation = slim.batch_norm(
                v_temp_activation,
                center=True,
                scale=True,
                is_training=is_training,
                scope="v_temp_activation_bn")
        v_temp_activation = tf.nn.leaky_relu(v_temp_activation)
        # if is_training:
        #     v_temp_activation = tf.nn.dropout(v_temp_activation, keep_prob=0.5)

        # 2. Distribution features.
        agg_v_distrib_embedding_dim = agg_v_distrib_embedding.get_shape().as_list()[1]
        video_distribution_hidden_weights = tf.get_variable("video_distribution_hidden",
                                               [agg_v_distrib_embedding_dim, 1024],
                                               initializer=tf.random_normal_initializer(
                                                   stddev=1 / math.sqrt(768)))
        v_distrib_activation = tf.matmul(agg_v_distrib_embedding, video_distribution_hidden_weights)
        if add_batch_norm:
            v_distrib_activation = slim.batch_norm(
                v_distrib_activation,
                center=True,
                scale=True,
                is_training=is_training,
                scope="v_distrib_activation_bn")
        v_distrib_activation = tf.nn.leaky_relu(v_distrib_activation)
        # if is_training:
        #     v_distrib_activation = tf.nn.dropout(v_distrib_activation, keep_prob=0.5)

        # Aggregating audio features.
        # 1. Temporal features.
        agg_a_temporal_embedding_dim = agg_a_temporal_embedding.get_shape().as_list()[1]
        audio_temporal_hidden_weights = tf.get_variable("audio_temporal_hidden",
                                            [agg_a_temporal_embedding_dim, 128],
                                            initializer=tf.random_normal_initializer(
                                               stddev=1 / math.sqrt(128)))
        a_temp_activation = tf.matmul(agg_a_temporal_embedding, audio_temporal_hidden_weights)
        if add_batch_norm:
            a_temp_activation = slim.batch_norm(
                a_temp_activation,
                center=True,
                scale=True,
                is_training=is_training,
                scope="a_temp_activation_bn")
        a_temp_activation = tf.nn.leaky_relu(a_temp_activation)
        # if is_training:
        #     a_temp_activation = tf.nn.dropout(a_temp_activation, keep_prob=0.5)

        # 2. Distribution features.
        agg_a_distrib_embedding_dim = agg_a_distrib_embedding.get_shape().as_list()[1]
        audio_distribution_hidden_weights = tf.get_variable("audio_distribution_hidden",
                                               [agg_a_distrib_embedding_dim, 128],
                                               initializer=tf.random_normal_initializer(
                                                   stddev=1 / math.sqrt(128)))
        a_distrib_activation = tf.matmul(agg_a_distrib_embedding, audio_distribution_hidden_weights)
        if add_batch_norm:
            a_distrib_activation = slim.batch_norm(
                a_distrib_activation,
                center=True,
                scale=True,
                is_training=is_training,
                scope="a_distrib_activation_bn")
        a_distrib_activation = tf.nn.leaky_relu(a_distrib_activation)
        # if is_training:
        #     a_distrib_activation = tf.nn.dropout(a_distrib_activation, keep_prob=0.5)

        agg_video = tf.concat([v_distrib_activation, v_temp_activation], 1)
        agg_audio = tf.concat([a_distrib_activation, a_temp_activation], 1)

        agg_video_hidden_weights = tf.get_variable("aggregation_video_hidden",
                                       [2048, 1024],
                                       initializer=tf.random_normal_initializer(
                                           stddev=1 / math.sqrt(1024)))
        agg_audio_hidden_weights = tf.get_variable("aggregation_audio_hidden",
                                       [256, 128],
                                       initializer=tf.random_normal_initializer(
                                           stddev=1 / math.sqrt(128)))

        agg_video_activation = tf.matmul(agg_video, agg_video_hidden_weights)
        if add_batch_norm:
            agg_video_activation = slim.batch_norm(
                agg_video_activation,
                center=True,
                scale=True,
                is_training=is_training,
                scope="agg_video_activation_bn")
        agg_video_activation = tf.nn.leaky_relu(agg_video_activation)
        # if is_training:
        #     agg_video_activation = tf.nn.dropout(agg_video_activation, keep_prob=0.5)

        agg_audio_activation = tf.matmul(agg_audio, agg_audio_hidden_weights)
        if add_batch_norm:
            agg_audio_activation = slim.batch_norm(
                agg_audio_activation,
                center=True,
                scale=True,
                is_training=is_training,
                scope="agg_audio_activation_bn")
        agg_audio_activation = tf.nn.leaky_relu(agg_audio_activation)
        # if is_training:
        #     agg_audio_activation = tf.nn.dropout(agg_audio_activation, keep_prob=0.5)

        activation = tf.concat([agg_video_activation, agg_audio_activation], 1)

        aggregated_model = getattr(video_level_models,
                                   "NN")

        return aggregated_model().create_model(
            model_input=activation,
            vocab_size=vocab_size,
            is_training=is_training,
            **unused_params)

###############################################################################
# NetVLAD Prototype ###########################################################
###############################################################################
# Flags for WillowModel
flags.DEFINE_bool("netvlad_add_batch_norm", True,
                  "Adds batch normalization to the DBoF model.")
flags.DEFINE_integer("netvlad_cluster_size", 256,
                     "Number of units in the NetVLAD cluster layer.")
flags.DEFINE_integer("netvlad_hidden_size", 1024,
                     "Number of units in the NetVLAD hidden layer.")
flags.DEFINE_bool("netvlad_relu", False,
                  'add ReLU to hidden layer')
flags.DEFINE_bool("gating", True,
                  "Gating for NetVLAD")
flags.DEFINE_bool("gating_remove_diag", False,
                  "Remove diag for self gating")
flags.DEFINE_float("audio_det_reg", 1e-4,
                    "The coefficient that determines the strength of the "
                    "determinant regularization penalty (of the VLAD cluster"
                    "centres, for audio features).")
flags.DEFINE_float("rgb_det_reg", 1e-4,
                    "The coefficient that determines the strength of the "
                    "determinant regularization penalty (of the VLAD cluster"
                    "centres, for rgb features).")


class WillowModelReg(models.BaseModel):
    """ WILLOW model with orthogonal regularization for robust features. """
    def create_model(self,
                     model_input,
                     vocab_size,
                     num_frames,
                     iterations=None,
                     add_batch_norm=None,
                     sample_random_frames=None,
                     cluster_size=None,
                     hidden_size=None,
                     is_training=True,
                     **unused_params):
        iterations = iterations or FLAGS.iterations
        add_batch_norm = add_batch_norm or FLAGS.netvlad_add_batch_norm
        random_frames = sample_random_frames or FLAGS.sample_random_frames
        cluster_size = cluster_size or FLAGS.netvlad_cluster_size
        hidden1_size = hidden_size or FLAGS.netvlad_hidden_size
        relu = FLAGS.netvlad_relu
        gating = FLAGS.gating
        remove_diag = FLAGS.gating_remove_diag

        num_frames = tf.cast(tf.expand_dims(num_frames, 1), tf.float32)
        if random_frames:
            model_input = utils.SampleRandomFrames(model_input, num_frames,
                                                   iterations)
        else:
            model_input = utils.SampleRandomSequence(model_input, num_frames,
                                                     iterations)

        # model_input: batch_size x max_frames x feature_size
        max_frames = model_input.get_shape().as_list()[1]
        feature_size = model_input.get_shape().as_list()[2]
        # model_input: (batch_size * max_frames) x feature_size
        reshaped_input = tf.reshape(model_input, [-1, feature_size])

        video_NetVLAD = video_pooling_modules.NetVladOrthoReg(1024, max_frames, cluster_size,
                                                              add_batch_norm, is_training,
                                                              FLAGS.rgb_det_reg, "netvlad_rgb_scope")
        audio_NetVLAD = video_pooling_modules.NetVladOrthoReg(128, max_frames, cluster_size / 4,
                                                              add_batch_norm, is_training,
                                                              FLAGS.audio_det_reg, "netvlad_audio_scope")
        if add_batch_norm:
            reshaped_input = slim.batch_norm(
                reshaped_input,
                center=True,
                scale=True,
                is_training=is_training,
                scope="input_bn")

        with tf.variable_scope("video_VLAD"):
            # (batch_size * max_frames) x 1024
            vlad_video = video_NetVLAD.forward(reshaped_input[:, 0:1024])

        with tf.variable_scope("audio_VLAD"):
            vlad_audio = audio_NetVLAD.forward(reshaped_input[:, 1024:])

        vlad = tf.concat([vlad_video, vlad_audio], 1)

        vlad_dim = vlad.get_shape().as_list()[1]
        hidden1_weights = tf.get_variable("hidden1_weights",
                                          [vlad_dim, hidden1_size],
                                          initializer=tf.random_normal_initializer(stddev=1 / math.sqrt(cluster_size)))

        activation = tf.matmul(vlad, hidden1_weights)

        if add_batch_norm and relu:
            activation = slim.batch_norm(
                activation,
                center=True,
                scale=True,
                is_training=is_training,
                scope="hidden1_bn")

        else:
            hidden1_biases = tf.get_variable("hidden1_biases",
                                             [hidden1_size],
                                             initializer=tf.random_normal_initializer(stddev=0.01))
            tf.summary.histogram("hidden1_biases", hidden1_biases)
            activation += hidden1_biases

        if relu:
            activation = tf.nn.relu6(activation)

        if gating:
            gating_weights = tf.get_variable("gating_weights_2",
                                             [hidden1_size, hidden1_size],
                                             initializer=tf.random_normal_initializer(
                                                 stddev=1 / math.sqrt(hidden1_size)))
            gates = tf.matmul(activation, gating_weights)

            if remove_diag:
                # Removes diagonals coefficients
                diagonals = tf.matrix_diag_part(gating_weights)
                gates = gates - tf.multiply(diagonals, activation)

            if add_batch_norm:
                gates = slim.batch_norm(
                    gates,
                    center=True,
                    scale=True,
                    is_training=is_training,
                    scope="gating_bn")
            else:
                gating_biases = tf.get_variable("gating_biases",
                                                [cluster_size],
                                                initializer=tf.random_normal(stddev=1 / math.sqrt(feature_size)))
                gates += gating_biases

            gates = tf.sigmoid(gates)
            activation = tf.multiply(activation, gates)

        aggregated_model = getattr(video_level_models,
                                   "WillowMoeModel")

        return aggregated_model().create_model(
            model_input=activation,
            vocab_size=vocab_size,
            is_training=is_training,
            **unused_params)
