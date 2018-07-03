"""
Investigate non-shareable CNN weights methods.

Input: (batch_size * max_frames) x anchor_size x feature_size
Weight: anchor_size x kernel_size x feature_size
Output: (batch_size * max_frames) x anchor_size x kernel_size
"""
import tensorflow as tf
import numpy as np

BATCH_SIZE = 4
MAX_FRAMES = 10
ANCHOR_SIZE = 8
FEATURE_SIZE = 10
KERNEL_SIZE = 6

inputs = tf.constant(
    np.arange(0, 3200),
    dtype=tf.float32,
    shape=[BATCH_SIZE * MAX_FRAMES, ANCHOR_SIZE, FEATURE_SIZE]
)
weight = tf.constant(
    np.arange(0, 480),
    dtype=tf.float32,
    shape=[ANCHOR_SIZE, KERNEL_SIZE, FEATURE_SIZE]
)

# Transpose the inputs.
tp_weight = tf.transpose(weight, perm=[0, 2, 1])
# -> weight: anchor_size x feature_size x kernel_size

# change
tp_inputs = tf.transpose(inputs, perm=[1, 0, 2])

output = tf.matmul(tp_inputs, tp_weight)

tp_output = tf.transpose(output, perm=[1, 0, 2])


with tf.Session():
    print("Initial Input:")
    print(inputs)
    print(inputs.eval())
    print("Initial Weight:")
    print(weight)
    print(weight.eval())
    print("Transposed Weight: ")
    print(tp_weight)
    print(tp_weight.eval())
    print("Output: ")
    # It should be (batch_size * max_frames) x anchor_size x kernel_size
    # 40 x 8 x 6
    print(output)
    print(output.eval())
    print("tp_output: ")
    print(tp_output)
