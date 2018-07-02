import tensorflow as tf
import numpy as np

feature_size = 5
anchor_size = 3
batch_size = 2
max_frames = 4
b = max_frames * batch_size
# inputs dim b * feature_size
inputs = tf.constant(np.arange(feature_size * anchor_size, feature_size * anchor_size + b * feature_size), dtype=tf.float32, shape=[b, feature_size])
anchor_weights = tf.constant(np.arange(feature_size * anchor_size), dtype=tf.float32, shape=[feature_size, anchor_size])
anchor_weights_reshape = tf.reshape(tf.transpose(anchor_weights), [1, feature_size * anchor_size])

tiled_inputs = tf.tile(inputs, [1, anchor_size])
t_emb = tf.subtract(tiled_inputs, anchor_weights_reshape)
t_emb_reshape = tf.reshape(t_emb, [-1, anchor_size, feature_size])
t_emb_norm = tf.nn.l2_normalize(t_emb_reshape, 2)
t_emb_final = tf.reshape(t_emb_norm, [-1, feature_size * anchor_size])

t_emb_reshape2 = tf.reshape(t_emb_final, [-1, max_frames, feature_size * anchor_size])

t_emb_reshape3 = tf.reshape(t_emb_final, [-1, anchor_size])


t_emb_cloned = tf.identity(t_emb_reshape2)
cloned_input = tf.manip.roll(t_emb_cloned, shift=1, axis=1)
input_subtract = tf.subtract(t_emb_reshape2, cloned_input)
input_subtract_reshape = tf.reshape(input_subtract, [-1, anchor_size, feature_size])
input_subtract_norm = tf.nn.l2_normalize(input_subtract_reshape, 2)
input_subtract_norm_reshape = tf.reshape(input_subtract_norm, [-1, max_frames, feature_size * anchor_size])


stacks = tf.unstack(input_subtract_norm_reshape, axis=1)
del stacks[0]
temp_info = tf.stack(stacks, 1)

with tf.Session():
    # print("anchor_weights: ")
    # print(anchor_weights.eval())
    # print("anchor_weights after tf.reshape: ")
    # print(anchor_weights_reshape.eval())
    # print("inputs: ")
    # print(inputs.eval())
    # print("inputs after tf.tile: ")
    # print(tiled_inputs.eval())
    # print("temb after tr.subtract: ")
    # print(t_emb.eval())
    # print("t_emb after tf.reshape: ")
    # print(t_emb_reshape.eval())
    # print("te_emb after norm: ")
    # print(t_emb_norm.eval())
    print("t_emb after reshape: ")
    print(t_emb_final.eval())
    print("reshape 2: ")
    print(t_emb_reshape2.eval())
    print("roll: ")
    print(cloned_input.eval())
    # print("t_emb_shape2: ")
    # print(t_emb_reshape2.eval())
    print("after subtract: ")
    print(input_subtract.eval())
    print("subtract reshape: ")
    print(input_subtract_reshape.eval())
    print("subtract norm: ")
    print(input_subtract_norm_reshape.eval())
    print("temp info: ")
    print(temp_info.eval())

    print("test")
    print(t_emb_reshape3.eval())
    print("test2")
print(t_emb_final)
print(t_emb_reshape2)
print(input_subtract_norm_reshape)
print(temp_info)