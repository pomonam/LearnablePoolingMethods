import tensorflow as tf
import numpy as np

feature_size = 5
anchor_size = 3
batch_size = 2
max_frames = 4
b = max_frames * batch_size
# inputs dim b * feature_size
inputs = tf.constant(np.arange(batch_size * max_frames, feature_size * anchor_size), dtype=tf.float32, shape=[b,
                                                                                                              feature_size *
                                                                                                              anchor_size])
modified_inputs = tf.expand_dims(inputs, -1)

with tf.Session():
    print("original")
    print(inputs.eval())
    print("modified")
    print(modified_inputs.eval())


print(inputs)
print(modified_inputs)
