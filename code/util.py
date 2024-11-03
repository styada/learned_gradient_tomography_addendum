"""Utilities needed for the networks."""

import numpy as np
import tensorflow as tf
import odl

def random_ellipse():
    return ((np.random.rand() - 0.3) * np.random.exponential(0.3),
            np.random.exponential() * 0.2, np.random.exponential() * 0.2,
            np.random.rand() - 0.5, np.random.rand() - 0.5,
            np.random.rand() * 2 * np.pi)


def random_phantom(spc):
    n = np.random.poisson(100)
    ellipses = [random_ellipse() for _ in range(n)]
    return odl.phantom.ellipsoid_phantom(spc, ellipses)


def conv2d(x, W, stride=(1, 1)):
    """
    The conv2d function performs a 2D convolution operation using TensorFlow with specified stride and
    padding.
    
    :param x: The parameter `x` in the `conv2d` function represents the input tensor to the
    convolutional layer. This tensor typically contains the input data or features that are being
    processed through the convolution operation
    :param W: The parameter `W` in the `conv2d` function represents the filter weights for the
    convolution operation. These weights are learned during the training process of a neural network.
    The filter weights determine how the convolution operation is applied to the input data `x` to
    extract features
    :param stride: The `stride` parameter in the `conv2d` function represents the stride length for
    moving the convolutional filter/kernel across the input tensor `x`
    :return: The function `conv2d` is returning the result of applying a 2D convolution operation using
    the input `x`, filter `W`, and the specified stride. The output of the convolution operation is
    returned.
    """
    return tf.nn.conv2d(x, W, strides=[1, stride[0], stride[1], 1], padding='SAME')
