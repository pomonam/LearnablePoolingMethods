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
import aggregation_modules
import loupe_modules
import video_level_models
import rnn_modules
import math
import models
import attention_modules
import juhan_modules
import transformer_utils
import crazy_utils

###############################################################################
# Transformer #################################################################
###############################################################################
flags.DEFINE_integer("crazyt_v1_iteration", 64,
                     "Number of frames per batch")
flags.DEFINE_integer("crazyt_v1_v_hidden", 1024,
                     "Number of hidden units")
flags.DEFINE_integer("crazyt_v1_a_hidden", 128,
                     "Number of hidden units")
flags.DEFINE_integer("crazyt_v1_v_filter_size", 4096,
                     "Number of heads")
flags.DEFINE_integer("crazyt_v1_a_filter_size", 512,
                     "Number of heads")
flags.DEFINE_integer("crazyt_v1_v_num_heads", 64,
                     "Number of heads")
flags.DEFINE_integer("crazyt_v1_a_num_heads", 16,
                     "Number of heads")
flags.DEFINE_integer("crazyt_v1_v_num_clusters", 64,
                     "Number of heads")
flags.DEFINE_integer("crazyt_v1_a_num_clusters", 16,
                     "Number of heads")
flags.DEFINE_integer("crazyt_v1_v_num_units", 64,
                     "Number of heads")
flags.DEFINE_integer("crazyt_v1_a_num_units", 16,
                     "Number of heads")
flags.DEFINE_float("crazyt_v1_v_attention_dropout", 0.1,
                   "Number of heads")
flags.DEFINE_float("crazyt_v1_a_attention_dropout", 0.1,
                   "Number of heads")
flags.DEFINE_integer("crazyt_v1_output_dim", 2048,
                     "Number of heads")
flags.DEFINE_string("crazyt_v1_video_model", "WillowMoeModel",
                    "Number of heads")


class CrazyTestV1(models.BaseModel):
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
        iterations = iterations or FLAGS.crazyt_v1_iteration
        video_hidden_size = FLAGS.crazyt_v1_v_hidden
        audio_hidden_size = FLAGS.crazyt_v1_a_hidden
        video_num_heads = FLAGS.crazyt_v1_v_num_heads
        audio_num_heads = FLAGS.crazyt_v1_a_num_heads
        video_num_clusters = FLAGS.crazyt_v1_v_num_clusters
        audio_num_clusters = FLAGS.crazyt_v1_a_num_clusters
        video_dropout = FLAGS.crazyt_v1_v_attention_dropout
        audio_dropout = FLAGS.crazyt_v1_a_attention_dropout
        video_num_units = FLAGS.crazyt_v1_v_num_units
        audio_num_units = FLAGS.crazyt_v1_a_num_units
        video_filter_size = FLAGS.crazyt_v1_v_filter_size
        audio_filter_size = FLAGS.crazyt_v1_a_filter_size
        output_dim = FLAGS.crazyt_v1_output_dim
        final_model = FLAGS.crazyt_v1_video_model

        num_frames = tf.cast(tf.expand_dims(num_frames, 1), tf.float32)
        model_input = utils.SampleRandomFrames(model_input, num_frames, iterations)
        # model_input: batch_size x max_frames x feature_size
        max_frames = model_input.get_shape().as_list()[1]
        feature_size = model_input.get_shape().as_list()[2]
        reshaped_input = tf.reshape(model_input, [-1, feature_size])

        # Differentiate video & audio features.
        video_features = reshaped_input[:, 0:1024]
        audio_features = reshaped_input[:, 1024:]

        video_features = slim.batch_norm(
            video_features,
            center=True,
            scale=True,
            is_training=is_training,
            scope="video_features_bn")
        audio_features = slim.batch_norm(
            audio_features,
            center=True,
            scale=True,
            is_training=is_training,
            scope="audio_features_bn")

        video_features = tf.reshape(video_features, [-1, max_frames, 1024])
        audio_features = tf.reshape(audio_features, [-1, max_frames, 128])

        crazy_v_mhd_v1 = crazy_utils.CrazyMultiHeadV2(feature_size=1024,
                                                      num_units=video_num_units,
                                                      num_heads=video_num_heads,
                                                      max_frames=max_frames,
                                                      is_training=is_training)
        crazy_v_ff_v1 = crazy_utils.CrazyFeedForwardV2(feature_size=1024,
                                                       max_frames=max_frames,
                                                       filter_size=video_filter_size,
                                                       relu_dropout=0.0,
                                                       is_train=is_training,
                                                       scope_id="video")
        crazy_a_mhd_v1 = crazy_utils.CrazyMultiHeadV2(feature_size=128,
                                                      num_units=audio_num_units,
                                                      num_heads=audio_num_heads,
                                                      max_frames=max_frames,
                                                      is_training=is_training)
        crazy_a_ff_v1 = crazy_utils.CrazyFeedForwardV2(feature_size=128,
                                                       max_frames=max_frames,
                                                       filter_size=video_filter_size,
                                                       relu_dropout=0.0,
                                                       is_train=is_training,
                                                       scope_id="audio")

        with tf.variable_scope("video"):
            with tf.variable_scope("encode"):
                with tf.variable_scope("block_1"):
                    encode1 = crazy_v_mhd_v1.forward(video_features)
                    with tf.variable_scope("ff"):
                        encode1 = crazy_v_ff_v1.forward(encode1)
                with tf.variable_scope("block_2"):
                    encode2 = crazy_v_mhd_v1.forward(encode1)
                    with tf.variable_scope("ff"):
                        encode2 = crazy_v_ff_v1.forward(encode2)
                with tf.variable_scope("block_3"):
                    encode3 = crazy_v_mhd_v1.forward(encode2)
                    with tf.variable_scope("ff"):
                        encode3 = crazy_v_ff_v1.forward(encode3)

            video_out = tf.reshape(encode3, [-1, video_num_clusters * 1024])

        with tf.variable_scope("audio"):
            with tf.variable_scope("encode"):
                with tf.variable_scope("block_1"):
                    encode1 = crazy_a_mhd_v1.forward(audio_features)
                    with tf.variable_scope("ff"):
                        encode1 = crazy_a_ff_v1.forward(encode1)
                with tf.variable_scope("block_2"):
                    encode2 = crazy_a_mhd_v1.forward(encode1)
                    with tf.variable_scope("ff"):
                        encode2 = crazy_a_ff_v1.forward(encode2)
                with tf.variable_scope("block_4"):
                    encode3 = crazy_a_mhd_v1.forward(encode2)
                    with tf.variable_scope("ff"):
                        encode3 = crazy_a_ff_v1.forward(encode3)

            audio_out = tf.reshape(encode3, [-1, audio_num_clusters * 128])

        activation = tf.concat([video_out, audio_out], 1)
        activation = tf.layers.dense(activation, output_dim, use_bias=False, activation=None)

        aggregated_model = getattr(video_level_models,
                                   final_model)

        return aggregated_model().create_model(
            model_input=activation,
            vocab_size=vocab_size,
            is_training=is_training,
            **unused_params)



flags.DEFINE_integer("crazy_v1_iteration", 64,
                     "Number of frames per batch")
flags.DEFINE_integer("crazy_v1_v_hidden", 1024,
                     "Number of hidden units")
flags.DEFINE_integer("crazy_v1_a_hidden", 128,
                     "Number of hidden units")
flags.DEFINE_integer("crazy_v1_v_filter_size", 4096,
                     "Number of heads")
flags.DEFINE_integer("crazy_v1_a_filter_size", 512,
                     "Number of heads")
flags.DEFINE_integer("crazy_v1_v_num_heads", 64,
                     "Number of heads")
flags.DEFINE_integer("crazy_v1_a_num_heads", 16,
                     "Number of heads")
flags.DEFINE_integer("crazy_v1_v_num_clusters", 64,
                     "Number of heads")
flags.DEFINE_integer("crazy_v1_a_num_clusters", 16,
                     "Number of heads")
flags.DEFINE_float("crazy_v1_v_attention_dropout", 0.1,
                   "Number of heads")
flags.DEFINE_float("crazy_v1_a_attention_dropout", 0.1,
                   "Number of heads")
flags.DEFINE_integer("crazy_v1_output_dim", 2048,
                     "Number of heads")
flags.DEFINE_string("crazy_v1_video_model", "WillowMoeModel",
                    "Number of heads")


class CrazyEncoderV1(models.BaseModel):
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
        iterations = iterations or FLAGS.crazy_v1_iteration
        video_hidden_size = FLAGS.crazy_v1_v_hidden
        audio_hidden_size = FLAGS.crazy_v1_a_hidden
        video_num_heads = FLAGS.crazy_v1_v_num_heads
        audio_num_heads = FLAGS.crazy_v1_a_num_heads
        video_num_clusters = FLAGS.crazy_v1_v_num_clusters
        audio_num_clusters = FLAGS.crazy_v1_a_num_clusters
        video_dropout = FLAGS.crazy_v1_v_attention_dropout
        audio_dropout = FLAGS.crazy_v1_a_attention_dropout
        video_filter_size = FLAGS.crazy_v1_v_filter_size
        audio_filter_size = FLAGS.crazy_v1_a_filter_size
        output_dim = FLAGS.crazy_v1_output_dim
        final_model = FLAGS.crazy_v1_video_model

        num_frames = tf.cast(tf.expand_dims(num_frames, 1), tf.float32)
        model_input = utils.SampleRandomFrames(model_input, num_frames, iterations)
        # model_input: batch_size x max_frames x feature_size
        max_frames = model_input.get_shape().as_list()[1]
        feature_size = model_input.get_shape().as_list()[2]
        reshaped_input = tf.reshape(model_input, [-1, feature_size])

        # Differentiate video & audio features.
        video_features = reshaped_input[:, 0:1024]
        audio_features = reshaped_input[:, 1024:]

        video_features = slim.batch_norm(
            video_features,
            center=True,
            scale=True,
            is_training=is_training,
            scope="video_features_bn")
        audio_features = slim.batch_norm(
            audio_features,
            center=True,
            scale=True,
            is_training=is_training,
            scope="audio_features_bn")

        video_features = tf.reshape(video_features, [-1, max_frames, 1024])
        audio_features = tf.reshape(audio_features, [-1, max_frames, 128])

        crazy_v_mhd_v1 = transformer_utils.CrazyMultiHead(feature_size=1024,
                                                          num_heads=video_num_heads,
                                                          max_frames=max_frames,
                                                          is_training=is_training)
        crazy_a_mhd_v1 = transformer_utils.CrazyMultiHead(feature_size=128,
                                                          num_heads=audio_num_heads,
                                                          max_frames=max_frames,
                                                          is_training=is_training)
        crazy_v_mhd_v2 = transformer_utils.CrazyMultiHead(feature_size=1024,
                                                          num_heads=video_num_heads,
                                                          max_frames=video_num_clusters,
                                                          is_training=is_training)
        crazy_a_mhd_v2 = transformer_utils.CrazyMultiHead(feature_size=128,
                                                          num_heads=audio_num_heads,
                                                          max_frames=audio_num_clusters,
                                                          is_training=is_training)
        crazy_v_ffn = transformer_utils.CrazyFeedForward(feature_size=1024,
                                                         filter_size=video_filter_size,
                                                         relu_dropout=0.0,
                                                         is_train=is_training,
                                                         scope_id="video")
        crazy_a_ffn = transformer_utils.CrazyFeedForward(feature_size=128,
                                                         filter_size=audio_filter_size,
                                                         relu_dropout=0.0,
                                                         is_train=is_training,
                                                         scope_id="audio")
        crazy_v_cluster = transformer_utils.CrazyCluster(feature_size=1024,
                                                         hidden_size=1024,
                                                         num_frames=max_frames,
                                                         last_layer=False,
                                                         num_cluster=video_num_clusters,
                                                         do_shift=True)
        crazy_a_cluster = transformer_utils.CrazyCluster(feature_size=1024,
                                                         hidden_size=1024,
                                                         num_frames=max_frames,
                                                         last_layer=False,
                                                         num_cluster=video_num_clusters,
                                                         do_shift=True)

        with tf.variable_scope("video"):
            with tf.variable_scope("encode"):
                with tf.variable_scope("block_1"):
                    encode1 = crazy_v_mhd_v1.forward(video_features)
                    encode1 = crazy_v_ffn.forward(encode1)
                with tf.variable_scope("block_2"):
                    encode2 = crazy_v_mhd_v1.forward(encode1)
                    encode2 = crazy_v_ffn.forward(encode2)
                with tf.variable_scope("block_4"):
                    encode4 = crazy_v_cluster.forward(encode2)
                with tf.variable_scope("block_5"):
                    encode5 = crazy_v_mhd_v2.forward(encode4)
                    encode5 = crazy_v_ffn.forward(encode5)
                with tf.variable_scope("block_6"):
                    encode6 = crazy_v_mhd_v2.forward(encode5)
                    encode6 = crazy_v_ffn.forward(encode6)
                with tf.variable_scope("block_7"):
                    encode7 = crazy_v_cluster.forward(encode6)
            video_out = tf.reshape(encode7, [-1, video_num_clusters * 1024])

        with tf.variable_scope("audio"):
            with tf.variable_scope("encode"):
                with tf.variable_scope("block_1"):
                    encode1 = crazy_a_mhd_v1.forward(audio_features)
                    encode1 = crazy_a_ffn.forward(encode1)
                with tf.variable_scope("block_2"):
                    encode2 = crazy_a_mhd_v1.forward(encode1)
                    encode2 = crazy_a_ffn.forward(encode2)
                with tf.variable_scope("block_4"):
                    encode4 = crazy_a_cluster.forward(encode2)
                with tf.variable_scope("block_5"):
                    encode5 = crazy_a_mhd_v2.forward(encode4)
                    encode5 = crazy_a_ffn.forward(encode5)
                with tf.variable_scope("block_6"):
                    encode6 = crazy_a_mhd_v2.forward(encode5)
                    encode6 = crazy_a_ffn.forward(encode6)
                with tf.variable_scope("block_7"):
                    encode7 = crazy_a_cluster.forward(encode6)
            audio_out = tf.reshape(encode7, [-1, audio_num_clusters * 128])

        activation = tf.concat([video_out, audio_out], 1)
        activation = tf.layers.dense(activation, output_dim, use_bias=False, activation=None)

        aggregated_model = getattr(video_level_models,
                                   final_model)

        return aggregated_model().create_model(
            model_input=activation,
            vocab_size=vocab_size,
            is_training=is_training,
            **unused_params)


flags.DEFINE_integer("jbtev5_iteration", 64,
                     "Number of frames per batch")
flags.DEFINE_integer("jbtev5_v_hidden", 1024,
                     "Number of hidden units")
flags.DEFINE_integer("jbtev5_a_hidden", 128,
                     "Number of hidden units")
flags.DEFINE_integer("jbtev5_v_filter_size", 4096,
                     "Number of heads")
flags.DEFINE_integer("jbtev5_a_filter_size", 512,
                     "Number of heads")
flags.DEFINE_integer("jbtev5_v_num_heads", 64,
                     "Number of heads")
flags.DEFINE_integer("jbtev5_a_num_heads", 16,
                     "Number of heads")
flags.DEFINE_float("jbtev5_v_attention_dropout", 0.1,
                   "Number of heads")
flags.DEFINE_float("jbtev5_a_attention_dropout", 0.1,
                   "Number of heads")
flags.DEFINE_string("jbtev5_video_model", "WillowMoeModel",
                     "Number of heads")


class JbTransformerEncoderV5(models.BaseModel):
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
        iterations = iterations or FLAGS.jbtev5_iteration
        video_hidden_size = FLAGS.jbtev5_v_hidden
        audio_hidden_size = FLAGS.jbtev5_a_hidden
        video_num_heads = FLAGS.jbtev5_v_num_heads
        audio_num_heads = FLAGS.jbtev5_a_num_heads
        video_dropout = FLAGS.jbtev5_v_attention_dropout
        audio_dropout = FLAGS.jbtev5_a_attention_dropout
        video_filter_size = FLAGS.jbtev5_v_filter_size
        audio_filter_size = FLAGS.jbtev5_a_filter_size
        final_model = FLAGS.jbtev5_video_model

        num_frames = tf.cast(tf.expand_dims(num_frames, 1), tf.float32)
        model_input = utils.SampleRandomFrames(model_input, num_frames, iterations)
        # model_input: batch_size x max_frames x feature_size
        max_frames = model_input.get_shape().as_list()[1]
        feature_size = model_input.get_shape().as_list()[2]
        reshaped_input = tf.reshape(model_input, [-1, feature_size])

        # Differentiate video & audio features.
        video_features = reshaped_input[:, 0:1024]
        audio_features = reshaped_input[:, 1024:]
        video_features = tf.reshape(video_features, [-1, max_frames, 1024])
        audio_features = tf.reshape(audio_features, [-1, max_frames, 128])

        v_block_1 = transformer_utils.JuhanBlock(feature_size=1024,
                                                 filter_size=video_filter_size,
                                                 num_cluster=video_num_heads,
                                                 num_units=video_hidden_size,
                                                 max_frames=max_frames,
                                                 is_training=is_training,
                                                 last_layer=False,
                                                 block_id=1)

        v_block_2 = transformer_utils.JuhanBlock(feature_size=1024,
                                                 filter_size=video_filter_size,
                                                 num_cluster=video_num_heads,
                                                 num_units=video_hidden_size,
                                                 max_frames=video_num_heads,
                                                 is_training=is_training,
                                                 last_layer=False,
                                                 block_id=2)

        v_block_3 = transformer_utils.JuhanBlock(feature_size=1024,
                                                 filter_size=video_filter_size,
                                                 num_cluster=video_num_heads,
                                                 num_units=video_hidden_size,
                                                 max_frames=video_num_heads,
                                                 is_training=is_training,
                                                 last_layer=False,
                                                 block_id=3)

        v_block_4 = transformer_utils.JuhanBlock(feature_size=1024,
                                                 filter_size=video_filter_size,
                                                 num_cluster=video_num_heads,
                                                 num_units=video_hidden_size,
                                                 max_frames=video_num_heads,
                                                 is_training=is_training,
                                                 last_layer=False,
                                                 block_id=4)

        v_block_5 = transformer_utils.JuhanBlock(feature_size=1024,
                                                 filter_size=video_filter_size,
                                                 num_cluster=video_num_heads,
                                                 num_units=video_hidden_size,
                                                 max_frames=video_num_heads,
                                                 is_training=is_training,
                                                 last_layer=False,
                                                 block_id=5)

        v_block_6 = transformer_utils.JuhanBlock(feature_size=1024,
                                                 filter_size=video_filter_size,
                                                 num_cluster=video_num_heads,
                                                 num_units=video_hidden_size,
                                                 max_frames=video_num_heads,
                                                 is_training=is_training,
                                                 last_layer=False,
                                                 block_id=6)

        a_block_1 = transformer_utils.JuhanBlock(feature_size=128,
                                                 filter_size=audio_filter_size,
                                                 num_cluster=audio_num_heads,
                                                 num_units=audio_hidden_size,
                                                 max_frames=max_frames,
                                                 is_training=is_training,
                                                 last_layer=False,
                                                 block_id=1)

        a_block_2 = transformer_utils.JuhanBlock(feature_size=128,
                                                 filter_size=audio_filter_size,
                                                 num_cluster=audio_num_heads,
                                                 num_units=audio_hidden_size,
                                                 max_frames=audio_num_heads,
                                                 is_training=is_training,
                                                 last_layer=False,
                                                 block_id=2)

        a_block_3 = transformer_utils.JuhanBlock(feature_size=128,
                                                 filter_size=audio_filter_size,
                                                 num_cluster=audio_num_heads,
                                                 num_units=audio_hidden_size,
                                                 max_frames=audio_num_heads,
                                                 is_training=is_training,
                                                 last_layer=False,
                                                 block_id=3)

        a_block_4 = transformer_utils.JuhanBlock(feature_size=128,
                                                 filter_size=audio_filter_size,
                                                 num_cluster=audio_num_heads,
                                                 num_units=audio_hidden_size,
                                                 max_frames=audio_num_heads,
                                                 is_training=is_training,
                                                 last_layer=False,
                                                 block_id=4)

        a_block_5 = transformer_utils.JuhanBlock(feature_size=128,
                                                 filter_size=audio_filter_size,
                                                 num_cluster=audio_num_heads,
                                                 num_units=audio_hidden_size,
                                                 max_frames=audio_num_heads,
                                                 is_training=is_training,
                                                 last_layer=False,
                                                 block_id=5)

        a_block_6 = transformer_utils.JuhanBlock(feature_size=128,
                                                 filter_size=audio_filter_size,
                                                 num_cluster=audio_num_heads,
                                                 num_units=audio_hidden_size,
                                                 max_frames=audio_num_heads,
                                                 is_training=is_training,
                                                 last_layer=False,
                                                 block_id=5)

        with tf.variable_scope("video"):
            with tf.variable_scope("encode"):
                with tf.variable_scope("block_1"):
                    encode1 = v_block_1.forward(video_features)
                with tf.variable_scope("block_2"):
                    encode2 = v_block_2.forward(encode1)
                with tf.variable_scope("block_3"):
                    encode3 = v_block_3.forward(encode2)
                with tf.variable_scope("block_4"):
                    encode4 = v_block_4.forward(encode3)
                with tf.variable_scope("block_5"):
                    encode5 = v_block_5.forward(encode4)
                with tf.variable_scope("block_6"):
                    encode6 = v_block_6.forward(encode5)

            video_out = tf.reshape(encode6, [-1, video_num_heads * 1024])

        with tf.variable_scope("audio"):
            with tf.variable_scope("encode"):
                with tf.variable_scope("block_1"):
                    encode1 = a_block_1.forward(audio_features)
                with tf.variable_scope("block_2"):
                    encode2 = a_block_2.forward(encode1)
                with tf.variable_scope("block_3"):
                    encode3 = a_block_3.forward(encode2)
                with tf.variable_scope("block_4"):
                    encode4 = a_block_4.forward(encode3)
                with tf.variable_scope("block_5"):
                    encode5 = a_block_5.forward(encode4)
                with tf.variable_scope("block_6"):
                    encode6 = a_block_6.forward(encode5)

            audio_out = tf.reshape(encode6, [-1, audio_num_heads * 128])

        activation = tf.concat([video_out, audio_out], 1)

        aggregated_model = getattr(video_level_models,
                                   final_model)

        return aggregated_model().create_model(
            model_input=activation,
            vocab_size=vocab_size,
            is_training=is_training,
            **unused_params)


flags.DEFINE_integer("jbtev4_iteration", 64,
                     "Number of frames per batch")
flags.DEFINE_integer("jbtev4_v_hidden", 1024,
                     "Number of hidden units")
flags.DEFINE_integer("jbtev4_a_hidden", 128,
                     "Number of hidden units")
flags.DEFINE_integer("jbtev4_v_filter_size", 4096,
                     "Number of heads")
flags.DEFINE_integer("jbtev4_a_filter_size", 512,
                     "Number of heads")
flags.DEFINE_integer("jbtev4_v_num_heads", 64,
                     "Number of heads")
flags.DEFINE_integer("jbtev4_a_num_heads", 16,
                     "Number of heads")
flags.DEFINE_float("jbtev4_v_attention_dropout", 0.1,
                   "Number of heads")
flags.DEFINE_float("jbtev4_a_attention_dropout", 0.1,
                   "Number of heads")
flags.DEFINE_string("jbtev4_video_model", "WillowMoeModel",
                     "Number of heads")


class JbTransformerEncoderV4(models.BaseModel):
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
        iterations = iterations or FLAGS.jbtev4_iteration
        video_hidden_size = FLAGS.jbtev4_v_hidden
        audio_hidden_size = FLAGS.jbtev4_a_hidden
        video_num_heads = FLAGS.jbtev4_v_num_heads
        audio_num_heads = FLAGS.jbtev4_a_num_heads
        video_dropout = FLAGS.jbtev4_v_attention_dropout
        audio_dropout = FLAGS.jbtev4_a_attention_dropout
        video_filter_size = FLAGS.jbtev4_v_filter_size
        audio_filter_size = FLAGS.jbtev4_a_filter_size
        final_model = FLAGS.jbtev4_video_model

        num_frames = tf.cast(tf.expand_dims(num_frames, 1), tf.float32)
        model_input = utils.SampleRandomFrames(model_input, num_frames, iterations)
        # model_input: batch_size x max_frames x feature_size
        max_frames = model_input.get_shape().as_list()[1]
        feature_size = model_input.get_shape().as_list()[2]
        reshaped_input = tf.reshape(model_input, [-1, feature_size])

        # Differentiate video & audio features.
        video_features = reshaped_input[:, 0:1024]
        audio_features = reshaped_input[:, 1024:]
        video_features = tf.reshape(video_features, [-1, max_frames, 1024])
        audio_features = tf.reshape(audio_features, [-1, max_frames, 128])

        v_block_1 = transformer_utils.JuhanBlock(feature_size=1024,
                                                 filter_size=video_filter_size,
                                                 num_cluster=video_num_heads,
                                                 num_units=video_hidden_size,
                                                 max_frames=max_frames,
                                                 is_training=is_training,
                                                 last_layer=False,
                                                 block_id=1)

        v_block_2 = transformer_utils.JuhanBlock(feature_size=1024,
                                                 filter_size=video_filter_size,
                                                 num_cluster=video_num_heads,
                                                 num_units=video_hidden_size,
                                                 max_frames=video_num_heads,
                                                 is_training=is_training,
                                                 last_layer=False,
                                                 block_id=2)

        v_block_3 = transformer_utils.JuhanBlock(feature_size=1024,
                                                 filter_size=video_filter_size,
                                                 num_cluster=video_num_heads,
                                                 num_units=video_hidden_size,
                                                 max_frames=video_num_heads,
                                                 is_training=is_training,
                                                 last_layer=False,
                                                 block_id=3)

        v_block_4 = transformer_utils.JuhanBlock(feature_size=1024,
                                                 filter_size=video_filter_size,
                                                 num_cluster=video_num_heads,
                                                 num_units=video_hidden_size,
                                                 max_frames=video_num_heads,
                                                 is_training=is_training,
                                                 last_layer=False,
                                                 block_id=4)

        a_block_1 = transformer_utils.JuhanBlock(feature_size=128,
                                                 filter_size=audio_filter_size,
                                                 num_cluster=audio_num_heads,
                                                 num_units=audio_hidden_size,
                                                 max_frames=max_frames,
                                                 is_training=is_training,
                                                 last_layer=False,
                                                 block_id=1)

        a_block_2 = transformer_utils.JuhanBlock(feature_size=128,
                                                 filter_size=audio_filter_size,
                                                 num_cluster=audio_num_heads,
                                                 num_units=audio_hidden_size,
                                                 max_frames=audio_num_heads,
                                                 is_training=is_training,
                                                 last_layer=False,
                                                 block_id=2)

        a_block_3 = transformer_utils.JuhanBlock(feature_size=128,
                                                 filter_size=audio_filter_size,
                                                 num_cluster=audio_num_heads,
                                                 num_units=audio_hidden_size,
                                                 max_frames=audio_num_heads,
                                                 is_training=is_training,
                                                 last_layer=False,
                                                 block_id=3)

        a_block_4 = transformer_utils.JuhanBlock(feature_size=128,
                                                 filter_size=audio_filter_size,
                                                 num_cluster=audio_num_heads,
                                                 num_units=audio_hidden_size,
                                                 max_frames=audio_num_heads,
                                                 is_training=is_training,
                                                 last_layer=False,
                                                 block_id=4)


        with tf.variable_scope("video"):
            with tf.variable_scope("encode"):
                with tf.variable_scope("block_1"):
                    encode1 = v_block_1.forward(video_features)
                with tf.variable_scope("block_2"):
                    encode2 = v_block_2.forward(encode1)
                with tf.variable_scope("block_3"):
                    encode3 = v_block_3.forward(encode2)
                with tf.variable_scope("block_4"):
                    encode4 = v_block_4.forward(encode3)

            video_out = tf.reshape(encode4, [-1, video_num_heads * 1024])

        with tf.variable_scope("audio"):
            with tf.variable_scope("encode"):
                with tf.variable_scope("block_1"):
                    encode1 = a_block_1.forward(audio_features)
                with tf.variable_scope("block_2"):
                    encode2 = a_block_2.forward(encode1)
                with tf.variable_scope("block_3"):
                    encode3 = a_block_3.forward(encode2)
                with tf.variable_scope("block_4"):
                    encode4 = a_block_4.forward(encode3)

            audio_out = tf.reshape(encode4, [-1, audio_num_heads * 128])

        activation = tf.concat([video_out, audio_out], 1)

        aggregated_model = getattr(video_level_models,
                                   final_model)

        return aggregated_model().create_model(
            model_input=activation,
            vocab_size=vocab_size,
            is_training=is_training,
            **unused_params)



flags.DEFINE_integer("jbtev3_iteration", 64,
                     "Number of frames per batch")
flags.DEFINE_integer("jbtev3_v_hidden", 1024,
                     "Number of hidden units")
flags.DEFINE_integer("jbtev3_a_hidden", 128,
                     "Number of hidden units")
flags.DEFINE_integer("jbtev3_v_filter_size", 4096,
                     "Number of heads")
flags.DEFINE_integer("jbtev3_a_filter_size", 512,
                     "Number of heads")
flags.DEFINE_integer("jbtev3_v_num_heads", 64,
                     "Number of heads")
flags.DEFINE_integer("jbtev3_a_num_heads", 16,
                     "Number of heads")
flags.DEFINE_float("jbtev3_v_attention_dropout", 0.1,
                     "Number of heads")
flags.DEFINE_float("jbtev3_a_attention_dropout", 0.1,
                     "Number of heads")
flags.DEFINE_string("jbtev3_video_model", "WillowMoeModel",
                     "Number of heads")


class JbTransformerEncoderV3(models.BaseModel):
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
        iterations = iterations or FLAGS.jbtev3_iteration
        video_hidden_size = FLAGS.jbtev3_v_hidden
        audio_hidden_size = FLAGS.jbtev3_a_hidden
        video_num_heads = FLAGS.jbtev3_v_num_heads
        audio_num_heads = FLAGS.jbtev3_a_num_heads
        video_dropout = FLAGS.jbtev3_v_attention_dropout
        audio_dropout = FLAGS.jbtev3_a_attention_dropout
        video_filter_size = FLAGS.jbtev3_v_filter_size
        audio_filter_size = FLAGS.jbtev3_a_filter_size
        final_model = FLAGS.jbtev3_video_model

        num_frames = tf.cast(tf.expand_dims(num_frames, 1), tf.float32)
        model_input = utils.SampleRandomFrames(model_input, num_frames, iterations)
        # model_input: batch_size x max_frames x feature_size
        max_frames = model_input.get_shape().as_list()[1]
        feature_size = model_input.get_shape().as_list()[2]
        reshaped_input = tf.reshape(model_input, [-1, feature_size])

        # Differentiate video & audio features.
        video_features = reshaped_input[:, 0:1024]
        audio_features = reshaped_input[:, 1024:]
        video_features = tf.reshape(video_features, [-1, max_frames, 1024])
        audio_features = tf.reshape(audio_features, [-1, max_frames, 128])

        v_block_1 = transformer_utils.JuhanBlock(feature_size=1024,
                                                 filter_size=video_filter_size,
                                                 num_cluster=video_num_heads,
                                                 num_units=video_hidden_size,
                                                 max_frames=max_frames,
                                                 is_training=is_training,
                                                 last_layer=False,
                                                 block_id=1)

        v_block_2 = transformer_utils.JuhanBlock(feature_size=1024,
                                                 filter_size=video_filter_size,
                                                 num_cluster=video_num_heads,
                                                 num_units=video_hidden_size,
                                                 max_frames=video_num_heads,
                                                 is_training=is_training,
                                                 last_layer=False,
                                                 block_id=2)

        a_block_1 = transformer_utils.JuhanBlock(feature_size=128,
                                                 filter_size=audio_filter_size,
                                                 num_cluster=audio_num_heads,
                                                 num_units=audio_hidden_size,
                                                 max_frames=max_frames,
                                                 is_training=is_training,
                                                 last_layer=False,
                                                 block_id=1)

        a_block_2 = transformer_utils.JuhanBlock(feature_size=128,
                                                 filter_size=audio_filter_size,
                                                 num_cluster=audio_num_heads,
                                                 num_units=audio_hidden_size,
                                                 max_frames=audio_num_heads,
                                                 is_training=is_training,
                                                 last_layer=False,
                                                 block_id=2)

        with tf.variable_scope("video"):
            with tf.variable_scope("encode"):
                with tf.variable_scope("block_1"):
                    encode1 = v_block_1.forward(video_features)
                with tf.variable_scope("block_2"):
                    encode2 = v_block_2.forward(encode1)

            video_out = tf.reshape(encode2, [-1, video_num_heads * 1024])

        with tf.variable_scope("audio"):
            with tf.variable_scope("encode"):
                with tf.variable_scope("block_1"):
                    encode1 = a_block_1.forward(audio_features)
                with tf.variable_scope("block_2"):
                    encode2 = a_block_2.forward(encode1)

            audio_out = tf.reshape(encode2, [-1, audio_num_heads * 128])

        activation = tf.concat([video_out, audio_out], 1)

        aggregated_model = getattr(video_level_models,
                                   final_model)

        return aggregated_model().create_model(
            model_input=activation,
            vocab_size=vocab_size,
            is_training=is_training,
            **unused_params)




flags.DEFINE_integer("jbtev2_iteration", 100,
                     "Number of frames per batch")
flags.DEFINE_integer("jbtev2_v_hidden", 1024,
                     "Number of hidden units")
flags.DEFINE_integer("jbtev2_a_hidden", 128,
                     "Number of hidden units")
flags.DEFINE_integer("jbtev2_v_filter_size", 4096,
                     "Number of heads")
flags.DEFINE_integer("jbtev2_a_filter_size", 512,
                     "Number of heads")
flags.DEFINE_integer("jbtev2_v_num_heads", 64,
                     "Number of heads")
flags.DEFINE_integer("jbtev2_a_num_heads", 16,
                     "Number of heads")
flags.DEFINE_float("jbtev2_v_attention_dropout", 0.1,
                     "Number of heads")
flags.DEFINE_float("jbtev2_a_attention_dropout", 0.1,
                     "Number of heads")
flags.DEFINE_string("jbtev2_video_model", "WillowMoeModel",
                     "Number of heads")


class JbTransformerEncoderV2(models.BaseModel):
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
        iterations = iterations or FLAGS.jbtev2_iteration
        video_hidden_size = FLAGS.jbtev2_v_hidden
        audio_hidden_size = FLAGS.jbtev2_a_hidden
        video_num_heads = FLAGS.jbtev2_v_num_heads
        audio_num_heads = FLAGS.jbtev2_a_num_heads
        video_dropout = FLAGS.jbtev2_v_attention_dropout
        audio_dropout = FLAGS.jbtev2_a_attention_dropout
        video_filter_size = FLAGS.jbtev2_v_filter_size
        audio_filter_size = FLAGS.jbtev2_a_filter_size

        final_model = FLAGS.jbtev2_video_model

        num_frames = tf.cast(tf.expand_dims(num_frames, 1), tf.float32)
        model_input = utils.SampleRandomFrames(model_input, num_frames, iterations)
        # model_input: batch_size x max_frames x feature_size
        max_frames = model_input.get_shape().as_list()[1]
        feature_size = model_input.get_shape().as_list()[2]
        reshaped_input = tf.reshape(model_input, [-1, feature_size])

        # Obtain video & audio features.
        video_features = reshaped_input[:, 0:1024]
        audio_features = reshaped_input[:, 1024:]

        video_features = tf.reshape(video_features, [-1, max_frames, 1024])
        audio_features = tf.reshape(audio_features, [-1, max_frames, 128])

        v_encoder_block = transformer_utils.TransformerEncoder(feature_size=1024,
                                                               hidden_size=video_hidden_size,
                                                               num_heads=video_num_heads,
                                                               attention_dropout=video_dropout,
                                                               ff_filter_size=video_filter_size,
                                                               ff_relu_dropout=0.1,
                                                               is_train=is_training,
                                                               scope_id="encode")

        a_encoder_block = transformer_utils.TransformerEncoder(feature_size=128,
                                                               hidden_size=audio_hidden_size,
                                                               num_heads=audio_num_heads,
                                                               attention_dropout=audio_dropout,
                                                               ff_filter_size=audio_filter_size,
                                                               ff_relu_dropout=0.1,
                                                               is_train=is_training,
                                                               scope_id="encode")

        with tf.variable_scope("video"):
            with tf.variable_scope("encode"):
                with tf.variable_scope("block_1"):
                    encode1 = v_encoder_block.forward(video_features)
                with tf.variable_scope("block_2"):
                    encode2 = v_encoder_block.forward(encode1)
                with tf.variable_scope("block_3"):
                    encode3 = v_encoder_block.forward(encode2)
                with tf.variable_scope("block_4"):
                    encode4 = v_encoder_block.forward(encode3)
                with tf.variable_scope("block_5"):
                    encode5 = v_encoder_block.forward(encode4)
                with tf.variable_scope("block_6"):
                    encode6 = v_encoder_block.forward(encode5)

            video_out = tf.reshape(encode6, [-1, iterations * 1024])

        with tf.variable_scope("audio"):
            with tf.variable_scope("encode"):
                with tf.variable_scope("block_1"):
                    encode1 = a_encoder_block.forward(audio_features)
                with tf.variable_scope("block_2"):
                    encode2 = a_encoder_block.forward(encode1)
                with tf.variable_scope("block_3"):
                    encode3 = a_encoder_block.forward(encode2)
                with tf.variable_scope("block_4"):
                    encode4 = a_encoder_block.forward(encode3)
                with tf.variable_scope("block_5"):
                    encode5 = a_encoder_block.forward(encode4)
                with tf.variable_scope("block_6"):
                    encode6 = a_encoder_block.forward(encode5)

            audio_out = tf.reshape(encode6, [-1, iterations * 128])

        activation = tf.concat([video_out, audio_out], 1)

        aggregated_model = getattr(video_level_models,
                                   final_model)

        return aggregated_model().create_model(
            model_input=activation,
            vocab_size=vocab_size,
            is_training=is_training,
            **unused_params)


flags.DEFINE_integer("jbtev1_iteration", 30,
                     "Number of frames per batch")
flags.DEFINE_integer("jbtev1_v_hidden", 64,
                     "Number of hidden units")
flags.DEFINE_integer("jbtev1_a_hidden", 16,
                     "Number of hidden units")
flags.DEFINE_integer("jbtev1_v_filter_size", 4096,
                     "Number of heads")
flags.DEFINE_integer("jbtev1_a_filter_size", 512,
                     "Number of heads")
flags.DEFINE_integer("jbtev1_v_num_heads", 16,
                     "Number of heads")
flags.DEFINE_integer("jbtev1_a_num_heads", 4,
                     "Number of heads")
flags.DEFINE_float("jbtev1_v_attention_dropout", 0.1,
                     "Number of heads")
flags.DEFINE_float("jbtev1_a_attention_dropout", 0.1,
                     "Number of heads")
flags.DEFINE_string("jbtev1_video_model", "WillowMoeModel",
                     "Number of heads")


class JbTransformerEncoderV1(models.BaseModel):
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
        iterations = iterations or FLAGS.jbtev1_iteration
        video_hidden_size = FLAGS.jbtev1_v_hidden
        audio_hidden_size = FLAGS.jbtev1_a_hidden
        video_num_heads = FLAGS.jbtev1_v_num_heads
        audio_num_heads = FLAGS.jbtev1_a_num_heads
        video_dropout = FLAGS.jbtev1_v_attention_dropout
        audio_dropout = FLAGS.jbtev1_a_attention_dropout
        video_filter_size = FLAGS.jbtev1_v_filter_size
        audio_filter_size = FLAGS.jbtev1_a_filter_size

        final_model = FLAGS.jbtev1_video_model

        num_frames = tf.cast(tf.expand_dims(num_frames, 1), tf.float32)
        model_input = utils.SampleRandomFrames(model_input, num_frames, iterations)
        # model_input: batch_size x max_frames x feature_size
        max_frames = model_input.get_shape().as_list()[1]
        feature_size = model_input.get_shape().as_list()[2]
        reshaped_input = tf.reshape(model_input, [-1, feature_size])

        # Obtain video & audio features.
        video_features = reshaped_input[:, 0:1024]
        audio_features = reshaped_input[:, 1024:]

        video_features = tf.reshape(video_features, [-1, max_frames, 1024])
        audio_features = tf.reshape(audio_features, [-1, max_frames, 128])

        v_encoder_block = transformer_utils.TransformerEncoder(feature_size=1024,
                                                               hidden_size=video_hidden_size,
                                                               num_heads=video_num_heads,
                                                               attention_dropout=video_dropout,
                                                               ff_filter_size=video_filter_size,
                                                               ff_relu_dropout=0.1,
                                                               is_train=is_training,
                                                               scope_id="encode")

        v_decoder_block = transformer_utils.TransformerDecoder(feature_size=1024,
                                                               hidden_size=video_hidden_size,
                                                               num_heads=video_num_heads,
                                                               attention_dropout=video_dropout,
                                                               ff_filter_size=video_filter_size,
                                                               ff_relu_dropout=0.1,
                                                               is_train=is_training,
                                                               scope_id="decode")

        a_encoder_block = transformer_utils.TransformerEncoder(feature_size=128,
                                                               hidden_size=audio_hidden_size,
                                                               num_heads=audio_num_heads,
                                                               attention_dropout=audio_dropout,
                                                               ff_filter_size=audio_filter_size,
                                                               ff_relu_dropout=0.1,
                                                               is_train=is_training,
                                                               scope_id="encode")

        a_decoder_block = transformer_utils.TransformerDecoder(feature_size=128,
                                                               hidden_size=audio_hidden_size,
                                                               num_heads=audio_num_heads,
                                                               attention_dropout=audio_dropout,
                                                               ff_filter_size=audio_filter_size,
                                                               ff_relu_dropout=0.1,
                                                               is_train=is_training,
                                                               scope_id="decode")

        video_vlad = loupe_modules.NetVLAD(feature_size=1024,
                                           max_samples=max_frames,
                                           cluster_size=iterations,
                                           output_dim=1024,
                                           gating=False,
                                           add_batch_norm=True,
                                           is_training=is_training)

        audio_vlad = loupe_modules.NetVLAD(feature_size=128,
                                           max_samples=max_frames,
                                           cluster_size=iterations,
                                           output_dim=128,
                                           gating=False,
                                           add_batch_norm=True,
                                           is_training=is_training)

        with tf.variable_scope("video"):
            with tf.variable_scope("encode"):
                with tf.variable_scope("block_1"):
                    encode1 = v_encoder_block.forward(video_features)
                with tf.variable_scope("block_2"):
                    encode2 = v_encoder_block.forward(encode1)
                with tf.variable_scope("block_3"):
                    encode3 = v_encoder_block.forward(encode2)
                with tf.variable_scope("block_4"):
                    encode4 = v_encoder_block.forward(encode3)
                with tf.variable_scope("block_5"):
                    encode5 = v_encoder_block.forward(encode4)
                with tf.variable_scope("block_6"):
                    encode6 = v_encoder_block.forward(encode5)

            with tf.variable_scope("netvlad"):
                reshaped_encode6 = tf.reshape(encode6, [-1, 1024])
                v_vlad_out = video_vlad.forward(reshaped_encode6)
                v_vlad_out = tf.reshape(v_vlad_out, [-1, iterations, 1024])

            with tf.variable_scope("decode"):
                with tf.variable_scope("block_1"):
                    decode1 = v_decoder_block.forward(v_vlad_out, encode1)
                with tf.variable_scope("block_2"):
                    decode2 = v_decoder_block.forward(decode1, encode2)
                with tf.variable_scope("block_3"):
                    decode3 = v_decoder_block.forward(decode2, encode3)
                with tf.variable_scope("block_4"):
                    decode4 = v_decoder_block.forward(decode3, encode4)
                with tf.variable_scope("block_5"):
                    decode5 = v_decoder_block.forward(decode4, encode5)
                with tf.variable_scope("block_6"):
                    decode6 = v_decoder_block.forward(decode5, encode6)

            video_out = tf.reshape(decode6, [-1, iterations * 1024])

        with tf.variable_scope("audio"):
            with tf.variable_scope("encode"):
                with tf.variable_scope("block_1"):
                    encode1 = a_encoder_block.forward(audio_features)
                with tf.variable_scope("block_2"):
                    encode2 = a_encoder_block.forward(encode1)
                with tf.variable_scope("block_3"):
                    encode3 = a_encoder_block.forward(encode2)
                with tf.variable_scope("block_4"):
                    encode4 = a_encoder_block.forward(encode3)
                with tf.variable_scope("block_5"):
                    encode5 = a_encoder_block.forward(encode4)
                with tf.variable_scope("block_6"):
                    encode6 = a_encoder_block.forward(encode5)

            with tf.variable_scope("netvlad"):
                reshaped_encode6 = tf.reshape(encode6, [-1, 128])
                a_vlad_out = audio_vlad.forward(reshaped_encode6)
                a_vlad_out = tf.reshape(a_vlad_out, [-1, iterations, 128])

            with tf.variable_scope("decode"):
                with tf.variable_scope("block_1"):
                    decode1 = a_decoder_block.forward(a_vlad_out, encode1)
                with tf.variable_scope("block_2"):
                    decode2 = a_decoder_block.forward(decode1, encode2)
                with tf.variable_scope("block_3"):
                    decode3 = a_decoder_block.forward(decode2, encode3)
                with tf.variable_scope("block_4"):
                    decode4 = a_decoder_block.forward(decode3, encode4)
                with tf.variable_scope("block_5"):
                    decode5 = a_decoder_block.forward(decode4, encode5)
                with tf.variable_scope("block_6"):
                    decode6 = a_decoder_block.forward(decode5, encode6)

            audio_out = tf.reshape(decode6, [-1, iterations * 128])

        activation = tf.concat([video_out, audio_out], 1)

        aggregated_model = getattr(video_level_models,
                                   final_model)

        return aggregated_model().create_model(
            model_input=activation,
            vocab_size=vocab_size,
            is_training=is_training,
            **unused_params)


flags.DEFINE_integer("transformer_iteration", 30,
                     "Number of frames per batch")
flags.DEFINE_integer("transformer_v_hidden", 256,
                     "Number of hidden units")
flags.DEFINE_integer("transformer_a_hidden", 128,
                     "Number of hidden units")
flags.DEFINE_integer("transformer_num_heads", 8,
                     "Number of heads")


class TransformerEncoder(models.BaseModel):
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
        iterations = iterations or FLAGS.transformer_iteration
        video_hidden_size = FLAGS.transformer_v_hidden
        audio_hidden_size = FLAGS.transformer_a_hidden
        num_heads = FLAGS.transformer_num_heads

        num_frames = tf.cast(tf.expand_dims(num_frames, 1), tf.float32)
        model_input = utils.SampleRandomFrames(model_input, num_frames, iterations)
        # model_input: batch_size x max_frames x feature_size
        max_frames = model_input.get_shape().as_list()[1]
        feature_size = model_input.get_shape().as_list()[2]
        # model_input: (batch_size * max_frames) x feature_size
        reshaped_input = tf.reshape(model_input, [-1, feature_size])

        video_layer1 = attention_modules.TransformerEncoderBlock(is_training, video_hidden_size, max_frames, 1024,
                                                                 num_heads, 0)
        video_layer2 = attention_modules.TransformerEncoderBlock(is_training, video_hidden_size, max_frames, 1024,
                                                                 num_heads, 1)
        video_layer3 = attention_modules.TransformerEncoderBlock(is_training, video_hidden_size, max_frames, 1024,
                                                                 num_heads, 2)
        video_layer4 = attention_modules.TransformerEncoderBlock(is_training, video_hidden_size, max_frames, 1024,
                                                                 num_heads, 3)

        audio_layer1 = attention_modules.TransformerEncoderBlock(is_training, audio_hidden_size, max_frames, 128,
                                                                 num_heads, 0)
        audio_layer2 = attention_modules.TransformerEncoderBlock(is_training, audio_hidden_size, max_frames, 128,
                                                                 num_heads, 1)
        audio_layer3 = attention_modules.TransformerEncoderBlock(is_training, audio_hidden_size, max_frames, 128,
                                                                 num_heads, 2)
        audio_layer4 = attention_modules.TransformerEncoderBlock(is_training, audio_hidden_size, max_frames, 128,
                                                                 num_heads, 3)

        with tf.variable_scope("video_encoder"):
            video_activation = video_layer1.forward(reshaped_input[:, 0:1024])
            video_activation = video_layer2.forward(video_activation)
            video_activation = video_layer3.forward(video_activation)
            video_activation = video_layer4.forward(video_activation)
            # video_activation: (batch_size * max_frames) x 1024

            # Attention
            video_activation = tf.reshape(video_activation, [-1, max_frames * 1024])
            # video_activation: batch_size x (max_frames * 1024)
            video_activation_weight = tf.layers.dense(video_activation, max_frames, tf.nn.relu)
            # video_activation_weight: batch_size x max_frames
            video_activation_weight = tf.expand_dims(video_activation_weight, -1)
            # video_activation_weight: batch_size x max_frames x 1
            video_activation_weight = tf.nn.softmax(video_activation_weight, axis=1)

            # Weighted sum
            video_activation = tf.reshape(video_activation, [-1, max_frames, 1024])
            video_activation = tf.reduce_mean(tf.multiply(video_activation, video_activation_weight), 1)
            # video_activation: batch_size x 1024

        with tf.variable_scope("audio_encoder"):
            audio_activation = audio_layer1.forward(reshaped_input[:, 1024:])
            audio_activation = audio_layer2.forward(audio_activation)
            audio_activation = audio_layer3.forward(audio_activation)
            audio_activation = audio_layer4.forward(audio_activation)

            # Attention
            audio_activation = tf.reshape(audio_activation, [-1, max_frames * 128])
            # audio_activation: batch_size x (max_frames * 128)
            audio_activation_weight = tf.layers.dense(audio_activation, max_frames, tf.nn.relu)
            # audio_activation_weight: batch_size x max_frames
            audio_activation_weight = tf.expand_dims(audio_activation_weight, -1)
            # audio_activation_weight: batch_size x max_frames x 1
            audio_activation_weight = tf.nn.softmax(audio_activation_weight, axis=1)

            # Weighted sum
            audio_activation = tf.reshape(audio_activation, [-1, max_frames, 128])
            audio_activation = tf.reduce_mean(tf.multiply(audio_activation, audio_activation_weight), 1)
            # video_activation: batch_size x 128

        # Fusion
        activation = tf.concat([video_activation, audio_activation], 1)
        activation = tf.layers.dense(activation, 1024, tf.nn.relu)
        # batch_size x 1024

        aggregated_model = getattr(video_level_models,
                                   "ClassLearningThreeNnModel")

        return aggregated_model().create_model(
            model_input=activation,
            vocab_size=vocab_size,
            is_training=is_training,
            **unused_params)


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
# All flags start with jtmv1_ to differentiate from other flags.
flags.DEFINE_integer("jtmv1_iteration", 30,
                     "Number of frames per batch.")
flags.DEFINE_bool("jtmv1_add_batch_norm", True,
                  "Add batch normalization.")
flags.DEFINE_bool("jtmv1_sample_random_frames", True,
                  "Iff true, tccm samples random frames.")
flags.DEFINE_integer("jtmv1_video_anchor_size", 64,
                     "Number of anchors for video features.")
flags.DEFINE_integer("jtmv1_audio_anchor_size", 16,
                     "Number of anchors for audio features.")
flags.DEFINE_integer("jtmv1_video_hidden", 1024,
                     "Number of anchors for video features.")
flags.DEFINE_integer("jtmv1_video_output_dim", 2048,
                     "Output dimension for video features.")
flags.DEFINE_integer("jtmv1_audio_hidden", 128,
                     "Number of anchors for audio features.")
flags.DEFINE_integer("jtmv1_audio_output_dim", 256,
                     "Output dimension for audio features.")
flags.DEFINE_bool("jtmv1_use_attention", True,
                  "True -> use attention.")
flags.DEFINE_bool("jtmv1_use_relu", True,
                  "True -> use relu.")


class JuhanTestModelV1(models.BaseModel):
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
        iterations = iterations or FLAGS.jtmv1_iteration
        add_batch_norm = add_batch_norm or FLAGS.jtmv1_add_batch_norm
        video_anchor_size = FLAGS.jtmv1_video_anchor_size
        audio_anchor_size = FLAGS.jtmv1_audio_anchor_size
        video_hidden_size = FLAGS.jtmv1_video_hidden
        audio_hidden_size = FLAGS.jtmv1_audio_hidden
        video_output_dim = FLAGS.jtmv1_video_output_dim
        audio_output_dim = FLAGS.jtmv1_audio_output_dim
        use_attention = FLAGS.jtmv1_use_attention
        use_relu = FLAGS.jtmv1_use_relu

        num_frames = tf.cast(tf.expand_dims(num_frames, 1), tf.float32)
        model_input = utils.SampleRandomFrames(model_input, num_frames, iterations)
        # model_input: batch_size x max_frames x feature_size
        max_frames = model_input.get_shape().as_list()[1]
        feature_size = model_input.get_shape().as_list()[2]
        # model_input: (batch_size * max_frames) x feature_size
        reshaped_input = tf.reshape(model_input, [-1, feature_size])

        video_module = video_pooling_modules.TriangulationCnnIndirectAttentionModule(
            feature_size=1024,
            max_frames=max_frames,
            anchor_size=video_anchor_size,
            self_attention=use_attention,
            hidden_layer_size=video_hidden_size,
            output_dim=video_output_dim,
            add_relu=use_relu,
            batch_norm=add_batch_norm,
            is_training=is_training,
            scope_id=None)

        audio_module = video_pooling_modules.TriangulationCnnIndirectAttentionModule(
            feature_size=128,
            max_frames=max_frames,
            anchor_size=audio_anchor_size,
            self_attention=use_attention,
            hidden_layer_size=audio_hidden_size,
            output_dim=audio_output_dim,
            add_relu=use_relu,
            batch_norm=add_batch_norm,
            is_training=is_training,
            scope_id=None)

        with tf.variable_scope("video_triangulation_embedding"):
            video_feature = video_module.forward(reshaped_input[:, 0:1024])
            # -> (batch_size * max_frames) x video_output_dim

        with tf.variable_scope("audio_triangulation_embedding"):
            audio_feature = audio_module.forward(reshaped_input[:, 1024:])
            # -> (batch_size * max_frames) x audio_output_dim

        activation = tf.concat([video_feature, audio_feature], 1)

        aggregated_model = getattr(video_level_models,
                                   "ClassLearningFourNnModel")
        return aggregated_model().create_model(
            model_input=activation,
            vocab_size=vocab_size,
            is_training=is_training,
            **unused_params)


# All flags start with jtmv2_ to differentiate from other flags.
flags.DEFINE_integer("jtmv2_iteration", 200,
                     "Number of frames per batch.")
flags.DEFINE_bool("jtmv2_add_batch_norm", True,
                  "Add batch normalization.")
flags.DEFINE_bool("jtmv2_sample_random_frames", True,
                  "Iff true, tccm samples random frames.")
flags.DEFINE_integer("jtmv2_video_anchor_size", 32,
                     "Number of anchors for video features.")
flags.DEFINE_integer("jtmv2_audio_anchor_size", 8,
                     "Number of anchors for audio features.")
flags.DEFINE_integer("jtmv2_video_kernel_size", 64,
                     "Number of kernels for video features.")
flags.DEFINE_integer("jtmv2_audio_kernel_size", 16,
                     "Number of kernels for audio features.")
flags.DEFINE_integer("jtmv2_video_hidden", 2048,
                     "Number of anchors for video features.")
flags.DEFINE_integer("jtmv2_video_output_dim", 2048,
                     "Output dimension for video features.")
flags.DEFINE_integer("jtmv2_audio_hidden", 256,
                     "Number of anchors for audio features.")
flags.DEFINE_integer("jtmv2_audio_output_dim", 256,
                     "Output dimension for audio features.")
flags.DEFINE_bool("jtmv2_use_attention", True,
                  "True -> use attention.")
flags.DEFINE_bool("jtmv2_use_relu", False,
                  "True -> use relu.")


class JuhanTestModelV2(models.BaseModel):
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
        iterations = iterations or FLAGS.jtmv2_iteration
        add_batch_norm = add_batch_norm or FLAGS.jtmv2_add_batch_norm
        video_anchor_size = FLAGS.jtmv2_video_anchor_size
        audio_anchor_size = FLAGS.jtmv2_audio_anchor_size
        video_hidden_size = FLAGS.jtmv2_video_hidden
        audio_hidden_size = FLAGS.jtmv2_audio_hidden
        video_kernel_size = FLAGS.jtmv2_video_kernel_size
        audio_kernel_size = FLAGS.jtmv2_audio_kernel_size
        video_output_dim = FLAGS.jtmv2_video_output_dim
        audio_output_dim = FLAGS.jtmv2_audio_output_dim
        use_attention = FLAGS.jtmv2_use_attention
        use_relu = FLAGS.jtmv2_use_relu

        num_frames = tf.cast(tf.expand_dims(num_frames, 1), tf.float32)
        model_input = utils.SampleRandomFrames(model_input, num_frames, iterations)
        # model_input: batch_size x max_frames x feature_size
        max_frames = model_input.get_shape().as_list()[1]
        feature_size = model_input.get_shape().as_list()[2]
        # model_input: (batch_size * max_frames) x feature_size
        reshaped_input = tf.reshape(model_input, [-1, feature_size])

        video_module = video_pooling_modules.TriangulationNsCnnIndirectAttentionModule(
            feature_size=1024,
            max_frames=max_frames,
            anchor_size=video_anchor_size,
            self_attention=use_attention,
            hidden_layer_size=video_hidden_size,
            kernel_size=video_kernel_size,
            output_dim=video_output_dim,
            add_relu=use_relu,
            batch_norm=add_batch_norm,
            is_training=is_training,
            scope_id=None)

        audio_module = video_pooling_modules.TriangulationNsCnnIndirectAttentionModule(
            feature_size=128,
            max_frames=max_frames,
            anchor_size=audio_anchor_size,
            self_attention=use_attention,
            hidden_layer_size=audio_hidden_size,
            kernel_size=audio_kernel_size,
            output_dim=audio_output_dim,
            add_relu=use_relu,
            batch_norm=add_batch_norm,
            is_training=is_training,
            scope_id=None)

        with tf.variable_scope("video_triangulation_embedding"):
            video_feature = video_module.forward(reshaped_input[:, 0:1024])
            # -> (batch_size * max_frames) x video_output_dim

        with tf.variable_scope("audio_triangulation_embedding"):
            audio_feature = audio_module.forward(reshaped_input[:, 1024:])
            # -> (batch_size * max_frames) x audio_output_dim

        activation = tf.concat([video_feature, audio_feature], 1)

        if add_batch_norm:
            activation = slim.batch_norm(
                activation,
                center=True,
                scale=True,
                is_training=is_training,
                scope="final_activation_bn")

        aggregated_model = getattr(video_level_models,
                                   "ClassLearningFourNnModel")
        return aggregated_model().create_model(
            model_input=activation,
            vocab_size=vocab_size,
            is_training=is_training,
            **unused_params)


# All flags start with jtmv3_ to differentiate from other flags.
flags.DEFINE_integer("jtmv3_iteration", 200,
                     "Number of frames per batch.")
flags.DEFINE_bool("jtmv3_add_batch_norm", True,
                  "Add batch normalization.")
flags.DEFINE_bool("jtmv3_sample_random_frames", True,
                  "Iff true, tccm samples random frames.")
flags.DEFINE_integer("jtmv3_video_anchor_size", 32,
                     "Number of anchors for video features.")
flags.DEFINE_integer("jtmv3_audio_anchor_size", 8,
                     "Number of anchors for audio features.")
flags.DEFINE_integer("jtmv3_video_kernel_size", 64,
                     "Number of kernels for video features.")
flags.DEFINE_integer("jtmv3_audio_kernel_size", 16,
                     "Number of kernels for audio features.")
flags.DEFINE_integer("jtmv3_video_hidden", 2048,
                     "Number of anchors for video features.")
flags.DEFINE_integer("jtmv3_video_output_dim", 2048,
                     "Output dimension for video features.")
flags.DEFINE_integer("jtmv3_audio_hidden", 256,
                     "Number of anchors for audio features.")
flags.DEFINE_integer("jtmv3_audio_output_dim", 256,
                     "Output dimension for audio features.")
flags.DEFINE_bool("jtmv3_use_attention", True,
                  "True -> use attention.")
flags.DEFINE_bool("jtmv3_use_relu", False,
                  "True -> use relu.")
flags.DEFINE_string("jtmv3_video_level_model", "ClassLearningFourNnModel",
                    "Model for video level.")


class JuhanTestModelV3(models.BaseModel):
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
        iterations = iterations or FLAGS.jtmv3_iteration
        add_batch_norm = add_batch_norm or FLAGS.jtmv3_add_batch_norm
        video_anchor_size = FLAGS.jtmv3_video_anchor_size
        audio_anchor_size = FLAGS.jtmv3_audio_anchor_size
        video_hidden_size = FLAGS.jtmv3_video_hidden
        audio_hidden_size = FLAGS.jtmv3_audio_hidden
        video_kernel_size = FLAGS.jtmv3_video_kernel_size
        audio_kernel_size = FLAGS.jtmv3_audio_kernel_size
        video_output_dim = FLAGS.jtmv3_video_output_dim
        audio_output_dim = FLAGS.jtmv3_audio_output_dim
        use_attention = FLAGS.jtmv3_use_attention
        use_relu = FLAGS.jtmv3_use_relu
        video_level_model = FLAGS.jtmv3_video_level_model

        num_frames = tf.cast(tf.expand_dims(num_frames, 1), tf.float32)
        model_input = utils.SampleRandomFrames(model_input, num_frames, iterations)
        # model_input: batch_size x max_frames x feature_size
        max_frames = model_input.get_shape().as_list()[1]
        feature_size = model_input.get_shape().as_list()[2]
        # model_input: (batch_size * max_frames) x feature_size
        reshaped_input = tf.reshape(model_input, [-1, feature_size])

        video_module = video_pooling_modules.TriangulationMagnitudeNsCnnIndirectAttentionModule(
            feature_size=1024,
            max_frames=max_frames,
            anchor_size=video_anchor_size,
            self_attention=use_attention,
            hidden_layer_size=video_hidden_size,
            kernel_size=video_kernel_size,
            output_dim=video_output_dim,
            add_relu=use_relu,
            add_norm=True,
            batch_norm=add_batch_norm,
            is_training=is_training,
            scope_id=None)

        audio_module = video_pooling_modules.TriangulationMagnitudeNsCnnIndirectAttentionModule(
            feature_size=128,
            max_frames=max_frames,
            anchor_size=audio_anchor_size,
            self_attention=use_attention,
            hidden_layer_size=audio_hidden_size,
            kernel_size=audio_kernel_size,
            output_dim=audio_output_dim,
            add_norm=True,
            add_relu=use_relu,
            batch_norm=add_batch_norm,
            is_training=is_training,
            scope_id=None)

        with tf.variable_scope("video_triangulation_embedding"):
            video_feature = video_module.forward(reshaped_input[:, 0:1024])
            # -> (batch_size * max_frames) x video_output_dim

        with tf.variable_scope("audio_triangulation_embedding"):
            audio_feature = audio_module.forward(reshaped_input[:, 1024:])
            # -> (batch_size * max_frames) x audio_output_dim

        activation = tf.concat([video_feature, audio_feature], 1)
        aggregated_model = getattr(video_level_models,
                                   video_level_model)
        return aggregated_model().create_model(
            model_input=activation,
            vocab_size=vocab_size,
            is_training=is_training,
            **unused_params)

# All flags start with jtmv3_ to differentiate from other flags.
flags.DEFINE_integer("jtmv4_iteration", 200,
                     "Number of frames per batch.")
flags.DEFINE_bool("jtmv4_add_batch_norm", True,
                  "Add batch normalization.")
flags.DEFINE_bool("jtmv4_sample_random_frames", True,
                  "Iff true, tccm samples random frames.")
flags.DEFINE_integer("jtmv4_video_anchor_size", 32,
                     "Number of anchors for video features.")
flags.DEFINE_integer("jtmv4_audio_anchor_size", 8,
                     "Number of anchors for audio features.")
flags.DEFINE_integer("jtmv4_video_kernel_size", 64,
                     "Number of kernels for video features.")
flags.DEFINE_integer("jtmv4_audio_kernel_size", 16,
                     "Number of kernels for audio features.")
flags.DEFINE_integer("jtmv4_video_hidden", 2048,
                     "Number of anchors for video features.")
flags.DEFINE_integer("jtmv4_video_output_dim", 2048,
                     "Output dimension for video features.")
flags.DEFINE_integer("jtmv4_audio_hidden", 256,
                     "Number of anchors for audio features.")
flags.DEFINE_integer("jtmv4_audio_output_dim", 256,
                     "Output dimension for audio features.")
flags.DEFINE_bool("jtmv4_use_attention", True,
                  "True -> use attention.")
flags.DEFINE_bool("jtmv4_use_relu", False,
                  "True -> use relu.")
flags.DEFINE_string("jtmv4_video_level_model", "ClassLearningFourNnModel",
                    "Model for video level.")


class JuhanTestModelV4(models.BaseModel):
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
        iterations = iterations or FLAGS.jtmv4_iteration
        add_batch_norm = add_batch_norm or FLAGS.jtmv4_add_batch_norm
        video_anchor_size = FLAGS.jtmv4_video_anchor_size
        audio_anchor_size = FLAGS.jtmv4_audio_anchor_size
        video_hidden_size = FLAGS.jtmv4_video_hidden
        audio_hidden_size = FLAGS.jtmv4_audio_hidden
        video_kernel_size = FLAGS.jtmv4_video_kernel_size
        audio_kernel_size = FLAGS.jtmv4_audio_kernel_size
        video_output_dim = FLAGS.jtmv4_video_output_dim
        audio_output_dim = FLAGS.jtmv4_audio_output_dim
        use_attention = FLAGS.jtmv4_use_attention
        use_relu = FLAGS.jtmv4_use_relu
        video_level_model = FLAGS.jtmv3_video_level_model

        num_frames = tf.cast(tf.expand_dims(num_frames, 1), tf.float32)
        model_input = utils.SampleRandomFrames(model_input, num_frames, iterations)
        # model_input: batch_size x max_frames x feature_size
        max_frames = model_input.get_shape().as_list()[1]
        feature_size = model_input.get_shape().as_list()[2]
        # model_input: (batch_size * max_frames) x feature_size
        reshaped_input = tf.reshape(model_input, [-1, feature_size])

        video_module = video_pooling_modules.TriangulationMagnitudeNsCnnNetVladModule(
            feature_size=1024,
            max_frames=max_frames,
            anchor_size=video_anchor_size,
            self_attention=use_attention,
            hidden_layer_size=video_hidden_size,
            kernel_size=video_kernel_size,
            output_dim=video_output_dim,
            add_relu=use_relu,
            add_norm=True,
            batch_norm=add_batch_norm,
            is_training=is_training,
            scope_id=None)

        audio_module = video_pooling_modules.TriangulationMagnitudeNsCnnNetVladModule(
            feature_size=128,
            max_frames=max_frames,
            anchor_size=audio_anchor_size,
            self_attention=use_attention,
            hidden_layer_size=audio_hidden_size,
            kernel_size=audio_kernel_size,
            output_dim=audio_output_dim,
            add_norm=True,
            add_relu=use_relu,
            batch_norm=add_batch_norm,
            is_training=is_training,
            scope_id=None)

        with tf.variable_scope("video_triangulation_embedding"):
            video_feature = video_module.forward(reshaped_input[:, 0:1024])
            # -> (batch_size * max_frames) x video_output_dim

        with tf.variable_scope("audio_triangulation_embedding"):
            audio_feature = audio_module.forward(reshaped_input[:, 1024:])
            # -> (batch_size * max_frames) x audio_output_dim

        activation = tf.concat([video_feature, audio_feature], 1)
        aggregated_model = getattr(video_level_models,
                                   video_level_model)
        return aggregated_model().create_model(
            model_input=activation,
            vocab_size=vocab_size,
            is_training=is_training,
            **unused_params)


# All flags start with jtmv5_ to differentiate from other flags.
flags.DEFINE_integer("jtmv5_iteration", 30,
                     "Number of frames per batch.")
flags.DEFINE_bool("jtmv5_add_batch_norm", True,
                  "Add batch normalization.")
flags.DEFINE_bool("jtmv5_sample_random_frames", True,
                  "Iff true, tccm samples random frames.")
flags.DEFINE_integer("jtmv5_video_anchor_size", 256,
                     "Number of anchors for video features.")
flags.DEFINE_integer("jtmv5_audio_anchor_size", 32,
                     "Number of anchors for audio features.")
flags.DEFINE_integer("jtmv5_video_kernel_size", 512,
                     "Number of kernels for video features.")
flags.DEFINE_integer("jtmv5_audio_kernel_size", 64,
                     "Number of kernels for audio features.")
flags.DEFINE_integer("jtmv5_video_hidden", 2048,
                     "Number of anchors for video features.")
flags.DEFINE_integer("jtmv5_video_output_dim", 4096,
                     "Output dimension for video features.")
flags.DEFINE_integer("jtmv5_audio_hidden", 256,
                     "Number of anchors for audio features.")
flags.DEFINE_integer("jtmv5_audio_output_dim", 512,
                     "Output dimension for audio features.")


class JuhanTestModelV5(models.BaseModel):
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
        iterations = iterations or FLAGS.jtmv5_iteration
        add_batch_norm = add_batch_norm or FLAGS.jtmv5_add_batch_norm
        video_anchor_size = FLAGS.jtmv5_video_anchor_size
        audio_anchor_size = FLAGS.jtmv5_audio_anchor_size
        video_kernel_size = FLAGS.jtmv5_video_kernel_size
        video_hidden_size = FLAGS.jtmv5_video_hidden
        audio_kernel_size = FLAGS.jtmv5_audio_kernel_size
        audio_hidden_size = FLAGS.jtmv5_audio_hidden
        video_output_dim = FLAGS.jtmv5_video_output_dim
        audio_output_dim = FLAGS.jtmv5_audio_output_dim

        num_frames = tf.cast(tf.expand_dims(num_frames, 1), tf.float32)
        model_input = utils.SampleRandomFrames(model_input, num_frames, iterations)
        # model_input: batch_size x max_frames x feature_size
        max_frames = model_input.get_shape().as_list()[1]
        feature_size = model_input.get_shape().as_list()[2]
        # model_input: (batch_size * max_frames) x feature_size
        reshaped_input = tf.reshape(model_input, [-1, feature_size])

        # Obtain video & audio features.
        video_features = reshaped_input[:, 0:1024]
        audio_features = reshaped_input[:, 1024:]

        # Batch normalize video & audio inputs for fixing scales.
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

        video_module = video_pooling_modules.TriangulationV5Module(
            feature_size=1024,
            max_frames=max_frames,
            anchor_size=video_anchor_size,
            kernel_size=video_kernel_size,
            self_attention=False,
            hidden_layer_size=video_hidden_size,
            output_dim=video_output_dim,
            add_relu=True,
            batch_norm=add_batch_norm,
            is_training=is_training,
            scope_id=None)

        audio_module = video_pooling_modules.TriangulationV5Module(
            feature_size=128,
            max_frames=max_frames,
            anchor_size=audio_anchor_size,
            kernel_size=audio_kernel_size,
            self_attention=False,
            hidden_layer_size=audio_hidden_size,
            output_dim=audio_output_dim,
            add_relu=True,
            batch_norm=add_batch_norm,
            is_training=is_training,
            scope_id=None)

        with tf.variable_scope("video_triangulation_embedding"):
            video_feature = video_module.forward(video_features)
            # -> (batch_size * max_frames) x video_output_dim

        with tf.variable_scope("audio_triangulation_embedding"):
            audio_feature = audio_module.forward(audio_features)
            # -> (batch_size * max_frames) x audio_output_dim

        activation = tf.concat([video_feature, audio_feature], 1)

        aggregated_model = getattr(video_level_models,
                                   "FourLayerBatchNeuralModel")
        return aggregated_model().create_model(
            model_input=activation,
            vocab_size=vocab_size,
            is_training=is_training,
            **unused_params)


# All flags start with jtmv5_ to differentiate from other flags.
flags.DEFINE_integer("jtmv6_iteration", 30,
                     "Number of frames per batch.")
flags.DEFINE_bool("jtmv6_add_batch_norm", True,
                  "Add batch normalization.")
flags.DEFINE_bool("jtmv6_sample_random_frames", True,
                  "Iff true, tccm samples random frames.")
flags.DEFINE_integer("jtmv6_video_anchor_size", 256,
                     "Number of anchors for video features.")
flags.DEFINE_integer("jtmv6_audio_anchor_size", 32,
                     "Number of anchors for audio features.")
flags.DEFINE_integer("jtmv6_video_kernel_size", 512,
                     "Number of kernels for video features.")
flags.DEFINE_integer("jtmv6_audio_kernel_size", 64,
                     "Number of kernels for audio features.")
flags.DEFINE_integer("jtmv6_video_hidden", 2048,
                     "Number of anchors for video features.")
flags.DEFINE_integer("jtmv6_video_output_dim", 4096,
                     "Output dimension for video features.")
flags.DEFINE_integer("jtmv6_audio_hidden", 256,
                     "Number of anchors for audio features.")
flags.DEFINE_integer("jtmv6_audio_output_dim", 512,
                     "Output dimension for audio features.")
flags.DEFINE_integer("jtmv6_video_cluster_size", 64,
                     "Output dimension for audio features.")
flags.DEFINE_integer("jtmv6_audio_cluster_size", 8,
                     "Output dimension for audio features.")


class JuhanTestModelV6(models.BaseModel):
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
        iterations = iterations or FLAGS.jtmv6_iteration
        add_batch_norm = add_batch_norm or FLAGS.jtmv6_add_batch_norm
        video_anchor_size = FLAGS.jtmv6_video_anchor_size
        audio_anchor_size = FLAGS.jtmv6_audio_anchor_size
        video_kernel_size = FLAGS.jtmv6_video_kernel_size
        video_hidden_size = FLAGS.jtmv6_video_hidden
        audio_kernel_size = FLAGS.jtmv6_audio_kernel_size
        audio_hidden_size = FLAGS.jtmv6_audio_hidden
        video_output_dim = FLAGS.jtmv6_video_output_dim
        audio_output_dim = FLAGS.jtmv6_audio_output_dim
        video_cluster_size = FLAGS.jtmv6_video_cluster_size
        audio_cluster_size = FLAGS.jtmv6_audio_cluster_size

        num_frames = tf.cast(tf.expand_dims(num_frames, 1), tf.float32)
        model_input = utils.SampleRandomFrames(model_input, num_frames, iterations)
        # model_input: batch_size x max_frames x feature_size
        max_frames = model_input.get_shape().as_list()[1]
        feature_size = model_input.get_shape().as_list()[2]
        # model_input: (batch_size * max_frames) x feature_size
        reshaped_input = tf.reshape(model_input, [-1, feature_size])

        # Obtain video & audio features.
        video_features = reshaped_input[:, 0:1024]
        audio_features = reshaped_input[:, 1024:]

        # Batch normalize video & audio inputs for fixing scales.
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

        video_module = video_pooling_modules.TriangulationV6Module(
            feature_size=1024,
            max_frames=max_frames,
            anchor_size=video_anchor_size,
            kernel_size=video_kernel_size,
            self_attention=False,
            cluster_size=video_cluster_size,
            hidden_layer_size=video_hidden_size,
            output_dim=video_output_dim,
            add_relu=True,
            batch_norm=add_batch_norm,
            is_training=is_training,
            scope_id=None)

        audio_module = video_pooling_modules.TriangulationV6Module(
            feature_size=128,
            max_frames=max_frames,
            anchor_size=audio_anchor_size,
            kernel_size=audio_kernel_size,
            self_attention=False,
            hidden_layer_size=audio_hidden_size,
            cluster_size=audio_cluster_size,
            output_dim=audio_output_dim,
            add_relu=True,
            batch_norm=add_batch_norm,
            is_training=is_training,
            scope_id=None)

        with tf.variable_scope("video_triangulation_embedding"):
            video_feature = video_module.forward(video_features)
            # -> (batch_size * max_frames) x video_output_dim

        with tf.variable_scope("audio_triangulation_embedding"):
            audio_feature = audio_module.forward(audio_features)
            # -> (batch_size * max_frames) x audio_output_dim

        activation = tf.concat([video_feature, audio_feature], 1)

        aggregated_model = getattr(video_level_models,
                                   "FourLayerBatchNeuralModel")
        return aggregated_model().create_model(
            model_input=activation,
            vocab_size=vocab_size,
            is_training=is_training,
            **unused_params)


# All flags start with tccm_ to differentiate from other flags.
flags.DEFINE_integer("tccm_iterations", 200,
                     "Number of frames per batch.")
flags.DEFINE_bool("tccm_add_batch_norm", True,
                  "Add batch normalization.")
flags.DEFINE_bool("tccm_sample_random_frames", True,
                  "Iff true, tccm samples random frames.")
flags.DEFINE_integer("tccm_video_anchor_size", 128,
                     "Number of anchors for video features.")
flags.DEFINE_integer("tccm_audio_anchor_size", 32,
                     "Number of anchors for audio features.")
flags.DEFINE_integer("tccm_video_kernel_size", 128,
                     "Number of anchors for video features.")
flags.DEFINE_integer("tccm_audio_kernel_size", 128,
                     "Number of anchors for video features.")
flags.DEFINE_integer("tccm_video_hidden", 2048,
                     "Number of anchors for video features.")
flags.DEFINE_integer("tccm_audio_hidden", 256,
                     "Number of anchors for video features.")


class TriangulationCnnClusterModel(models.BaseModel):
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
        iterations = iterations or FLAGS.tccm_iterations
        add_batch_norm = add_batch_norm or FLAGS.tccm_add_batch_norm
        video_anchor_size = FLAGS.tccm_video_anchor_size
        audio_anchor_size = FLAGS.tccm_audio_anchor_size
        video_kernel_size = FLAGS.tccm_video_kernel_size
        audio_kernel_size = FLAGS.tccm_audio_kernel_size
        video_hidden_size = FLAGS.tccm_video_hidden
        audio_hidden_size = FLAGS.tccm_audio_hidden

        num_frames = tf.cast(tf.expand_dims(num_frames, 1), tf.float32)
        model_input = utils.SampleRandomFrames(model_input, num_frames, iterations)
        # model_input: batch_size x max_frames x feature_size
        max_frames = model_input.get_shape().as_list()[1]
        feature_size = model_input.get_shape().as_list()[2]
        # model_input: (batch_size * max_frames) x feature_size
        reshaped_input = tf.reshape(model_input, [-1, feature_size])

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

        video_d_cnn_module = video_pooling_modules.TriangulationCnnModule(1024,
                                                                          max_frames,
                                                                          video_kernel_size,
                                                                          video_anchor_size,
                                                                          add_batch_norm,
                                                                          is_training,
                                                                          "video_d")

        video_t_cnn_module = video_pooling_modules.TriangulationCnnModule(1024,
                                                                          max_frames - 1,
                                                                          video_kernel_size,
                                                                          video_anchor_size,
                                                                          add_batch_norm,
                                                                          is_training,
                                                                          "video_t")

        audio_d_cnn_module = video_pooling_modules.TriangulationCnnModule(128,
                                                                          max_frames,
                                                                          audio_kernel_size,
                                                                          audio_anchor_size,
                                                                          add_batch_norm,
                                                                          is_training,
                                                                          "audio_d")

        audio_t_cnn_module = video_pooling_modules.TriangulationCnnModule(128,
                                                                          max_frames - 1,
                                                                          audio_kernel_size,
                                                                          audio_anchor_size,
                                                                          add_batch_norm,
                                                                          is_training,
                                                                          "audio_t")

        ic_mean_pool = aggregation_modules.IndirectClusterMeanPoolModule(l2_normalize=False)
        mean_std_pool = aggregation_modules.MeanStdPoolModule(l2_normalize=False)

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
            # -> (batch_size * max_frames) x (feature_size * anchor_size)
            with tf.variable_scope("video_d"):
                video_d_cnn = video_d_cnn_module.forward(video_d)
                # -> batch_size x max_frames x (anchor_size * num_filters)

            video_d_temp = tf.reshape(video_d, [-1, max_frames, 1024 * video_anchor_size])
            agg_video_d = ic_mean_pool.forward(video_d_temp, video_d_cnn)
            # -> batch_size x (anchor_size * num_filters)

            video_t = video_t_module.forward(video_d)
            # -> batch_size x (max_frames - 1) x (feature_size * anchor_size)
            video_t = tf.reshape(video_t, [-1, 1024 * video_anchor_size])
            with tf.variable_scope("video_t"):
                video_t_cnn = video_t_cnn_module.forward(video_t)
            # -> batch_size x (max_frames - 1) x (anchor_size * num_filters)
            agg_video_t = mean_std_pool.forward(video_t_cnn)

            agg_video = tf.concat([agg_video_d, agg_video_t], 1)

            if add_batch_norm:
                agg_video = slim.batch_norm(
                    agg_video,
                    center=True,
                    scale=True,
                    is_training=is_training,
                    scope="agg_video_bn")

        with tf.variable_scope("audio_triangulation_embedding"):
            audio_d = audio_d_module.forward(audio_features)
            # -> (batch_size * max_frames) x (feature_size * anchor_size)
            with tf.variable_scope("audio_d"):
                audio_d_cnn = audio_d_cnn_module.forward(audio_d)
                # -> batch_size x max_frames x (anchor_size * num_filters)

            audio_d_temp = tf.reshape(audio_d, [-1, max_frames, 128 * audio_anchor_size])
            agg_audio_d = ic_mean_pool.forward(audio_d_temp, audio_d_cnn)
            # -> batch_size x (anchor_size * num_filters)

            audio_t = audio_t_module.forward(audio_d)
            # -> batch_size x (max_frames - 1) x (feature_size * anchor_size)
            audio_t = tf.reshape(audio_t, [-1, 128 * audio_anchor_size])
            with tf.variable_scope("audio_t"):
                audio_t_cnn = audio_t_cnn_module.forward(audio_t)
                # -> batch_size x (max_frames - 1) x (anchor_size * num_filters)
            agg_audio_t = mean_std_pool.forward(audio_t_cnn)

            agg_audio = tf.concat([agg_audio_d, agg_audio_t], 1)

            if add_batch_norm:
                agg_audio = slim.batch_norm(
                    agg_audio,
                    center=True,
                    scale=True,
                    is_training=is_training,
                    scope="agg_audio_bn")

        agg_video_dim = agg_video.get_shape().as_list()[1]
        video_hidden_weight = tf.get_variable("video_hidden",
                                              [agg_video_dim, video_hidden_size],
                                              initializer=tf.random_normal_initializer(
                                                  stddev=1 / math.sqrt(video_hidden_size)))
        video_activation = tf.matmul(agg_video, video_hidden_weight)

        agg_audio_dim = agg_audio.get_shape().as_list()[1]
        audio_hidden_weight = tf.get_variable("audio_hidden",
                                              [agg_audio_dim, audio_hidden_size],
                                              initializer=tf.random_normal_initializer(
                                                  stddev=1 / math.sqrt(audio_hidden_size)))
        audio_activation = tf.matmul(agg_audio, audio_hidden_weight)

        activation = tf.concat([video_activation, audio_activation], 1)

        aggregated_model = getattr(video_level_models,
                                   "ClassLearningFourNnModel")
        return aggregated_model().create_model(
            model_input=activation,
            vocab_size=vocab_size,
            is_training=is_training,
            **unused_params)


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
