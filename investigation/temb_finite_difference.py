import tensorflow as tf

# Using `Session.run()`.
sess = tf.Session()
cloned_inputs = tf.constant([[[1,2], [3,4], [13,14]], [[5,6], [7,8], [15, 16]],[[9, 10], [11, 12], [17, 18]], [[1,2], [3,4], [13,14]]])
a_vecs = tf.unstack(cloned_inputs, axis=1)
print(a_vecs)
del a_vecs[0]
a_new = tf.stack(a_vecs, 1)
with tf.Session():
    print(cloned_inputs.eval())
    print("ssss")
    print(a_new.eval())


print(cloned_inputs)
print(a_new)