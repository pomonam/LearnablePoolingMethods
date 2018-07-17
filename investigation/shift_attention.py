import tensorflow as tf
import numpy as np
import math

feature_size = 5
num_cluster = 3
batch_size = 2
max_frames = 4
inputs = tf.constant(np.arange(0, 40), dtype=tf.float32, shape=[batch_size * max_frames,
                                                                                                feature_size])
attention_weight = tf.constant(np.arange(0, feature_size * num_cluster), dtype=tf.float32, shape=[feature_size,
                                                                                   num_cluster])

activation = tf.matmul(inputs, attention_weight)
activation2 = tf.reshape(activation, [-1, max_frames, num_cluster])
activation3 = tf.scalar_mul(1 / math.sqrt(feature_size), activation2)
activation4 = tf.nn.softmax(activation3, dim=1)


reshaped_inputs = tf.reshape(inputs, [-1, max_frames, feature_size])
activation5 = tf.transpose(activation4, perm=[0, 2, 1])
att_activation = tf.matmul(activation5, reshaped_inputs)

reshaped_activation = tf.reshape(att_activation, [-1, feature_size])
reshaped_activation2 = tf.scalar_mul(3, reshaped_activation)
reshaped_activation3 = reshaped_activation + 0.1
reshaped_activation4 = tf.nn.l2_normalize(reshaped_activation3, 1)
reshaped_activation5 = tf.scalar_mul(1 / math.sqrt(num_cluster), reshaped_activation)

with tf.Session():
    print("original")
    print(inputs.eval())
    print(attention_weight.eval())
    print("1")
    print(activation.eval())
    print(activation2.eval())
    print(activation3.eval())
    print(activation4.eval())
    print("2")
    print(activation5.eval())
    print(att_activation.eval())
    print("3")
    print(reshaped_activation.eval())
    print(reshaped_activation2.eval())
    print(reshaped_activation3.eval())
    print(reshaped_activation4.eval())
    print(reshaped_activation5.eval())



print(inputs)
