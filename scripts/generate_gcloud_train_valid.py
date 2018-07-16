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

"""
Generate command line arguments for Google Cloud MLE training.
Run command: BUCKET_NAME=gs://dtp1_yt8m_train_bucket
"""


import os

####################################################################
# Configuration ####################################################
####################################################################
# yaml settings. cloudml-4gpu.yaml, cloudml-gpu.yaml, cloudml-gpu-distributed.yaml
CLOUD_GPU = "cloudml-gpu.yaml"
# Name and version of the model
MODEL_NAME = "TransformerEncoder"
MODEL_VERSION = ""
# Does it require frame-level models?
FRAME_LEVEL = True
# What features? e.g. RGB, audio
FEATURES = "rgb,audio"
# Batch size.
BATCH_SIZE = 64
# Base LR.
BASE_LEARNING_RATE = 0.00005
# Initialize a new model?
START_NEW_MODEL = True
EXTRA = "-learning_rate_decay=0.7"
# EXTRA = "-tembed_v3_video_anchor_size=64 " \
#         "-tembed_v3_audio_anchor_size=8 " \
#         "-tembed_v3_distrib_concat_hidden_size=4096 " \
#         "-tembed_v3_temporal_concat_hidden_size=4096 " \
#         "-tembed_v3_full_concat_hidden_size=8192"


def main():
    # Start by defining a job name.
    local_command = "gcloud ml-engine local train "
    command = "JOB_NAME=yt8m_train_$(date +%Y%m%d_%H%M%S); "
    command += "gcloud --verbosity=debug ml-engine jobs submit training $JOB_NAME "
    command += "--package-path=youtube-8m --module-name=youtube-8m.train "
    local_command += "--package-path=youtube-8m --module-name=youtube-8m.train "
    command += "--staging-bucket=$BUCKET_NAME --region=us-east1 "
    command += "--config=youtube-8m/cloudml_config/{} ".format(CLOUD_GPU)
    if FRAME_LEVEL:
        command += "-- --train_data_pattern='gs://youtube8m-ml-us-east1/2/frame/train/*.tfrecord' "
        command += "--frame_features=True "
        local_command += "-- --train_data_pattern='gs://youtube8m-ml-us-east1/2/frame/train/*.tfrecord' "
        local_command += "--frame_features=True "
    else:
        command += "-- --train_data_pattern='gs://youtube8m-ml-us-east1/2/video/train/*.tfrecord' "
        command += "--frame_features=False "
        local_command += "-- --train_data_pattern='gs://youtube8m-ml-us-east1/2/video/train/*.tfrecord' "
        local_command += "--frame_features=False "
    command += "--base_learning_rate={} ".format(str(BASE_LEARNING_RATE))
    local_command += "--base_learning_rate={} ".format(str(BASE_LEARNING_RATE))
    command += "--model={} ".format(MODEL_NAME)
    local_command += "--model={} ".format(MODEL_NAME)
    command += "--feature_names='{}' ".format(FEATURES)
    local_command += "--feature_names='{}' ".format(FEATURES)
    command += "--feature_sizes='1024,128' "
    local_command += "--feature_sizes='1024,128' "
    command += "--batch_size={} ".format(str(BATCH_SIZE))
    local_command += "--batch_size={} ".format(str(BATCH_SIZE))
    command += "--train_dir=$BUCKET_NAME/{} ".format(MODEL_NAME + str(MODEL_VERSION))
    local_command += "--train_dir=/tmp/yt8m_train "
    command += "--base_learning_rate={} ".format(str(BASE_LEARNING_RATE))
    local_command += "--base_learning_rate={} ".format(str(BASE_LEARNING_RATE))
    if START_NEW_MODEL:
        command += "--start_new_model "
        local_command += "--start_new_model "
    local_command += "--runtime-version=1.8"
    command += EXTRA
    local_command += EXTRA
    return command, local_command


if __name__ == "__main__":
    current_directory = os.getcwd()
    current_directory = "/".join(current_directory.split("/")[:-2])
    print("Run the following command here: {}".format(current_directory))
    c, lc = main()
    print("Local: \n{}".format(lc))
    print("Cloud: \n{}".format(c))
