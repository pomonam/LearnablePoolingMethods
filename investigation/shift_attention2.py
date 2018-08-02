import tensorflow as tf
import numpy as np

feature_size = 3
num_cluster = 3
batch_size = 2
max_frames = 4
output = tf.constant(np.arange(0, 18), dtype=tf.float32, shape=[batch_size, num_cluster, feature_size])
transposed_input = tf.transpose(output, perm=[0, 2, 1])

cluster_weight = tf.constant(np.arange(0, 3), dtype=tf.float32, shape=[num_cluster])

addition = transposed_input * cluster_weight

normalized = tf.nn.l2_normalize(addition, 1)

expand_normalized = tf.reshape(normalized, [-1, num_cluster * feature_size])
renormalized = tf.nn.l2_normalize(expand_normalized, 1)

sample = tf.constant(np.arange(0, 18), dtype=tf.float32, shape=[batch_size * num_cluster, feature_size])
normalized_sample = tf.nn.l2_normalize(sample, 1)


with tf.Session():
    print("original")
    print(output.eval())
    print(transposed_input.eval())
    print("add cluster")
    print(cluster_weight.eval())
    print(addition.eval())
    print(normalized.eval())
    print("expand")
    print(expand_normalized.eval())
    print(renormalized.eval())
    print("sample")
    print(sample.eval())
    print(normalized_sample.eval())
