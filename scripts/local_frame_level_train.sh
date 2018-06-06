@!/bin/sh

# Execute the following command
gcloud ml-engine local train \
--package-path=youtube-8m --module-name=youtube-8m.train -- \
--train_data_pattern='gs://youtube8m-ml-us-east1/2/frame/train/train*.tfrecord' \
--train_dir=/tmp/yt8m_train --frame_features=True --model=FrameLevelLogisticModel --feature_names="rgb" \
--feature_sizes="1024" --batch_size=128 \
--train_dir=$BUCKET_NAME/yt8m_train_frame_level_logistic_model --start_new_model
