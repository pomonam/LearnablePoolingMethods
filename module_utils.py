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

from tensorflow.python.platform import tf_logging as logging
from tensorflow.python.ops import standard_ops
from tensorflow.python.framework import ops
import tensorflow as tf
import numbers


def orthogonal_regularizer(scale, scope=None):
    """ Return a function that computes orthogonal regularization.
    :param scale: A scalar multiplier `Tensor`. 0.0 disables the regularizer.
    :param scope: An optional scope name.
    :return: A function with signature `orthogonal_sum(weights)` that applies orthogonal regularization.
    """
    if isinstance(scale, numbers.Integral):
        raise ValueError('scale cannot be an integer: %s' % (scale,))
    if isinstance(scale, numbers.Real):
        if scale < 0.:
            raise ValueError('Setting a scale less than 0 on a regularizer: %g.' %
                             scale)
        if scale == 0.:
            logging.info('Scale of 0 disables regularizer.')
            return lambda _: None

    def orthogonal_sum(weights):
        """ Applies orthogonal regularization to weights. """
        with ops.name_scope(scope, 'orthogonal_regularizer', [weights]) as name:
            tensor_scale = ops.convert_to_tensor(scale,
                                                 dtype=weights.dtype.base_dtype,
                                                 name='scale')

            norm_weights = tf.nn.l2_normalize(weights, axis=1)
            anchor_weights_t = tf.transpose(norm_weights)
            det_reg = tf.matmul(anchor_weights_t, norm_weights)
            identity = tf.eye(tf.shape(det_reg)[0])
            det_reg = tf.subtract(det_reg, identity)
            det_reg = tf.reduce_sum(tf.abs(det_reg))

            # Print sum value before scaling
            det_reg = tf.Print(det_reg, [det_reg], "Orthogonal sum for \"{}\" :".format(name))

            return standard_ops.multiply(tensor_scale, det_reg, name=name)

    return orthogonal_sum


def reduce_var(x, axis=None, keep_dim=False):
    """ Return variance of a tensor, alongside the specified axis.

    Reference:
    https://stackoverflow.com/questions/39354566/what-is-the-equivalent-of-np-std-in-tensorflow

    :param x: Tensor or variable
    :param axis: int
    :param keep_dim: bool
    :return: Tensor with the variance of elements of x
    """
    m = tf.reduce_mean(x, axis=axis, keep_dims=True)
    devs_squared = tf.square(x - m)
    return tf.reduce_mean(devs_squared, axis=axis, keep_dims=keep_dim)