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

""" A script to batch-evaluate the algorithm on different epochs. """
import glob
import json
import os
import time
import sys
import eval_util
import losses
import video_level_models
import frame_level_models
import readers
import tensorflow as tf
from tensorflow.python.lib.io import file_io
from tensorflow import app
from tensorflow import flags
from tensorflow import gfile
from tensorflow import logging
import utils

FLAGS = flags.FLAGS



def find_class_by_name(name, modules):
    """ Searches the provided modules for the named class and returns it. """
    modules = [getattr(module, name, None) for module in modules]
    return next(a for a in modules if a)


def get_input_evaluation_tensors(reader,
                                 data_pattern,
                                 batch_size=1024,
                                 num_readers=1):
    """Creates the section of the graph which reads the evaluation data.

      Args:
        reader: A class which parses the training data.
        data_pattern: A 'glob' style path to the data files.
        batch_size: How many examples to process at a time.
        num_readers: How many I/O threads to use.

      Returns:
        A tuple containing the features tensor, labels tensor, and optionally a
        tensor containing the number of frames per video. The exact dimensions
        depend on the reader being used.

      Raises:
        IOError: If no files matching the given pattern were found.
    """
    logging.info("Using batch size of " + str(batch_size) + " for evaluation.")
    with tf.name_scope("eval_input"):
        files = gfile.Glob(data_pattern)
        if not files:
            raise IOError("Unable to find the evaluation files.")
        logging.info("number of evaluation files: " + str(len(files)))
        filename_queue = tf.train.string_input_producer(
            files, shuffle=False, num_epochs=1)
        eval_data = [
            reader.prepare_reader(filename_queue) for _ in range(num_readers)
        ]
        return tf.train.batch_join(
            eval_data,
            batch_size=batch_size,
            capacity=3 * batch_size,
            allow_smaller_final_batch=True,
            enqueue_many=True)


def build_graph(reader,
                model,
                eval_data_pattern,
                label_loss_fn,
                batch_size=1024,
                num_readers=1):
    """Creates the Tensorflow graph for evaluation.

      Args:
        reader: The data file reader. It should inherit from BaseReader.
        model: The core model (e.g. logistic or neural net). It should inherit
               from BaseModel.
        eval_data_pattern: glob path to the evaluation data files.
        label_loss_fn: What kind of loss to apply to the model. It should inherit
                    from BaseLoss.
        batch_size: How many examples to process at a time.
        num_readers: How many threads to use for I/O operations.
    """

    global_step = tf.Variable(0, trainable=False, name="global_step")
    video_id_batch, model_input_raw, labels_batch, num_frames = get_input_evaluation_tensors(
        # pylint: disable=g-line-too-long
        reader,
        eval_data_pattern,
        batch_size=batch_size,
        num_readers=num_readers)
    tf.summary.histogram("model_input_raw", model_input_raw)

    feature_dim = len(model_input_raw.get_shape()) - 1

    # Normalize input features.
    model_input = tf.nn.l2_normalize(model_input_raw, feature_dim)

    with tf.variable_scope("tower"):
        result = model.create_model(model_input,
                                    num_frames=num_frames,
                                    vocab_size=reader.num_classes,
                                    labels=labels_batch,
                                    is_training=False)
        predictions = result["predictions"]
        tf.summary.histogram("model_activations", predictions)
        if "loss" in result.keys():
            label_loss = result["loss"]
        else:
            label_loss = label_loss_fn.calculate_loss(predictions, labels_batch)

    tf.add_to_collection("global_step", global_step)
    tf.add_to_collection("loss", label_loss)
    tf.add_to_collection("predictions", predictions)
    tf.add_to_collection("input_batch", model_input)
    tf.add_to_collection("input_batch_raw", model_input_raw)
    tf.add_to_collection("video_id_batch", video_id_batch)
    tf.add_to_collection("num_frames", num_frames)
    tf.add_to_collection("labels", tf.cast(labels_batch, tf.float32))
    tf.add_to_collection("summary_op", tf.summary.merge_all())


def get_latest_checkpoint():
    index_files = file_io.get_matching_files(os.path.join(FLAGS.train_dir, 'model.ckpt-*.index'))
    tf.logging.debug("Looking at {}".format(index_files))

    # No files
    if not index_files:
        return None

    # Index file path with the maximum step size.
    latest_index_file = sorted(
        [(int(os.path.basename(f).split("-")[-1].split(".")[0]), f)
         for f in index_files])[-1][1]

    # Chop off .index suffix and return
    return latest_index_file[:-6]


def evaluate():
    tf.set_random_seed(0)  # for reproducibility

    # Write json of flags
    model_flags_path = os.path.join(FLAGS.train_dir, "model_flags.json")
    if not file_io.file_exists(model_flags_path):
        raise IOError(("Cannot find file %s. Did you run train.py on the same "
                       "--train_dir?") % model_flags_path)
    flags_dict = json.loads(file_io.FileIO(model_flags_path, mode="r").read())

    with tf.Graph().as_default():
        # convert feature_names and feature_sizes to lists of values
        feature_names, feature_sizes = utils.GetListOfFeatureNamesAndSizes(
            flags_dict["feature_names"], flags_dict["feature_sizes"])

        if flags_dict["frame_features"]:
            reader = readers.YT8MFrameFeatureReader(feature_names=feature_names,
                                                    feature_sizes=feature_sizes)
        else:
            reader = readers.YT8MAggregatedFeatureReader(feature_names=feature_names,
                                                         feature_sizes=feature_sizes)

        model = find_class_by_name(flags_dict["model"],
                                   [frame_level_models, video_level_models])()
        label_loss_fn = find_class_by_name(flags_dict["label_loss"], [losses])()

        if FLAGS.eval_data_pattern is "":
            raise IOError("'eval_data_pattern' was not specified. " +
                          "Nothing to evaluate.")

        build_graph(
            reader=reader,
            model=model,
            eval_data_pattern=FLAGS.eval_data_pattern,
            label_loss_fn=label_loss_fn,
            num_readers=FLAGS.num_readers,
            batch_size=FLAGS.batch_size)
        logging.info("built evaluation graph")

        summary_writer = tf.summary.FileWriter(
            FLAGS.train_dir, graph=tf.get_default_graph())

        evl_metrics = eval_util.EvaluationMetrics(reader.num_classes, FLAGS.top_k)
