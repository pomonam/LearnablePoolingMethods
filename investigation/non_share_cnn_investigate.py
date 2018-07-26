"""
Investigate non-shareable CNN weights methods.

Input: (batch_size * max_frames) x anchor_size x feature_size
Weight: anchor_size x kernel_size x feature_size
Output: (batch_size * max_frames) x anchor_size x kernel_size
"""
import tensorflow as tf
import numpy as np

BATCH_SIZE = 1
MAX_FRAMES = 2
ANCHOR_SIZE = 3
FEATURE_SIZE = 4
KERNEL_SIZE = 2

inputs = tf.constant(
    np.arange(0, 24),
    dtype=tf.float32,
    shape=[BATCH_SIZE * MAX_FRAMES, ANCHOR_SIZE, FEATURE_SIZE]
)
weight = tf.constant(
    np.arange(0, 24),
    dtype=tf.float32,
    shape=[ANCHOR_SIZE, KERNEL_SIZE, FEATURE_SIZE]
)

normalized_inputs = tf.nn.l2_normalize(inputs, 2)

# Transpose the inputs.
tp_weight = tf.transpose(weight, perm=[0, 2, 1])
# -> weight: anchor_size x feature_size x kernel_size

# change
tp_inputs = tf.transpose(inputs, perm=[1, 0, 2])

output = tf.matmul(tp_inputs, tp_weight)

tp_output = tf.transpose(output, perm=[1, 0, 2])

# Get final features
final = tf.reshape(tp_output, [-1, ANCHOR_SIZE * KERNEL_SIZE])


with tf.Session():
    print("Initial Input:")
    print(inputs)
    print(inputs.eval())
    print(normalized_inputs.eval())
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
    print(tp_output.eval())
    print("final: ")
    print(final)
    print(final.eval())
