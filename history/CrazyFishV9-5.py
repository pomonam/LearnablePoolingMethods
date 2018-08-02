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

""" Template for logging train / evaluation / inference methods. """

import os

####################################################################
# Configuration ####################################################
####################################################################

# Name and version of the model
MODEL_NAME = "CrazyFishV9"
MODEL_VERSION = "-5"

# Does it require frame-level models?
FRAME_LEVEL = True

# What features? e.g. RGB, audio
FEATURES = "rgb,audio"

# Batch size.
BATCH_SIZE = 256

# Base LR.
BASE_LEARNING_RATE = 0.0002

# Initialize a new model?
START_NEW_MODEL = False

EXTRA = "--fish9_iteration=300 " \
        "--fish9_video_cluster_size=256 " \
        "--fish9_audio_cluster_size=32 " \
        "--fish9_shift_operation=True " \
        "--fish9_filter_size=4 " \
        "--fish9_cluster_dropout=0.8 " \
        "--fish9_ff_dropout=0.9 " \
        "--fish9_linear_proj_dropout=0.9 " \
        "--fish9_l2_regularization_rate=1e-6 " \
        "--fish9_hidden_size=1024 " \
        "--moe_num_mixtures=4 " \
        "--learning_rate_decay_examples=2000000 " \
        "--learning_rate_decay=0.8 " \
        "--num_epochs=4 " \
        "--moe_l2=1e-6 " \
        "--max_step=400000 "


def main():
    # Start by defining a job name.
    local_command = "gcloud ml-engine local train "
    local_command += "--package-path=youtube-8m --module-name=youtube-8m.train "
    if FRAME_LEVEL:
        local_command += "-- --train_data_pattern='gs://youtube8m-ml-us-east1/2/frame/train/train*.tfrecord," \
                         "gs://youtube8m-ml-us-east1/2/frame/validate/validate*.tfrecord' "
        local_command += "--frame_features=True "
    else:
        local_command += "-- --train_data_pattern='gs://youtube8m-ml-us-east1/2/video/train/*.tfrecord' "
        local_command += "--frame_features=False "
    local_command += "--base_learning_rate={} ".format(str(BASE_LEARNING_RATE))
    local_command += "--model={} ".format(MODEL_NAME)
    local_command += "--feature_names='{}' ".format(FEATURES)
    local_command += "--feature_sizes='1024,128' "
    local_command += "--batch_size={} ".format(str(BATCH_SIZE))
    local_command += "--train_dir={} ".format(MODEL_NAME + str(MODEL_VERSION))
    local_command += "--base_learning_rate={} ".format(str(BASE_LEARNING_RATE))
    if START_NEW_MODEL:
        local_command += "--start_new_model "
    local_command += "--runtime-version=1.8 "
    local_command += EXTRA

    eval_command = "gcloud ml-engine local train "
    eval_command += "--package-path=youtube-8m --module-name=youtube-8m.eval "
    if FRAME_LEVEL:
        eval_command += "-- --eval_data_pattern='gs://youtube8m-ml-us-east1/2/frame/validate/validate*.tfrecord' "
        eval_command += "--frame_features=True "
    eval_command += "--model={} ".format(MODEL_NAME)
    eval_command += "--feature_names='{}' ".format(FEATURES)
    eval_command += "--feature_sizes='1024,128' "
    eval_command += "--batch_size={} ".format(str(BATCH_SIZE))
    eval_command += "--train_dir=/home/deeptopology2/JuhanTestModelV3-2/ ".format(MODEL_NAME + str(MODEL_VERSION))
    eval_command += "--base_learning_rate={} ".format(str(BASE_LEARNING_RATE))
    eval_command += "--run_once=True "
    eval_command += EXTRA

    inference_command = "gcloud ml-engine local train "
    inference_command += "--package-path=youtube-8m --module-name=youtube-8m.inference "
    if FRAME_LEVEL:
        inference_command += "-- --input_data_pattern='gs://youtube8m-ml-us-east1/2/frame/test/test*.tfrecord' "
        inference_command += "--frame_features=True "
    inference_command += "--model={} ".format(MODEL_NAME)
    inference_command += "--feature_names='{}' ".format(FEATURES)
    inference_command += "--feature_sizes='1024,128' "
    inference_command += "--batch_size=256 ".format(str(BATCH_SIZE))
    inference_command += "--train_dir=/home/deeptopology2/JuhanTestModelV3-2/ ".format(MODEL_NAME + str(MODEL_VERSION))
    inference_command += "--base_learning_rate={} ".format(str(BASE_LEARNING_RATE))
    inference_command += "--output_file=/home/deeptopology2/JuhanTestModelV3-2/predictions.csv ".format(MODEL_NAME + str(MODEL_VERSION))
    inference_command += EXTRA

    return local_command, eval_command, inference_command


if __name__ == "__main__":
    current_directory = os.getcwd()
    current_directory = "/".join(current_directory.split("/")[:-2])
    print("Run the following command here: {}".format(current_directory))
    tc, ec, ic = main()
    print("Train: \n{}".format(tc))
    print("Evaluation: \n{}".format(ec))
    print("Inference: \n{}".format(ic))
