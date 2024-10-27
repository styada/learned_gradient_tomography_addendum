"""Partially learned gradient descent without regularizer as input."""
"""
The code implements a partially learned gradient descent algorithm without regularizer for solving
an inverse problem using TensorFlow and ODL.

:param validation: The `validation` parameter in the code is used to determine whether to generate a
set of random data for validation purposes. When `validation` is set to `True`, the code generates a
single set of data for validation. Otherwise, it generates multiple sets of random data for
training, defaults to False (optional)
:return: The code is a TensorFlow implementation of a partially learned gradient descent
algorithm without a regularizer as input. It involves creating ODL (Operator Discretization Library)
data structures, generating random data, defining placeholders and variables, implementing an
iterative scheme, calculating loss, defining an optimizer, and training the network.
"""

import tensorflow as tf
import numpy as np
import odl
import odl.contrib.tensorflow
from util import random_phantom, conv2d

# Create ODL data structures
size = 128
space = odl.uniform_discr([-64, -64], [64, 64], [size, size], dtype='float32')

geometry = odl.tomo.parallel_beam_geometry(space, num_angles=30)
operator = odl.tomo.RayTransform(space, geometry)
pseudoinverse = odl.tomo.fbp_op(operator)

# Ensure operator has fixed operator norm for scale invariance
opnorm = odl.power_method_opnorm(operator)
operator = (1 / opnorm) * operator
pseudoinverse = pseudoinverse * opnorm

# Create tensorflow layer from odl operator
odl_op_layer = odl.contrib.tensorflow.as_tensorflow_layer(operator,'RayTransform')
odl_op_layer_adjoint = odl.contrib.tensorflow.as_tensorflow_layer(operator.adjoint, 'RayTransformAdjoint')

# User selected paramters
n_data = 20
n_memory = 5
n_iter = 10

# Remove the placeholders and directly use variables for your data
x_0 = tf.Variable(tf.zeros([None, size, size, 1]), dtype=tf.float32, name="x_0")
x_true = tf.Variable(tf.zeros([None, size, size, 1]), dtype=tf.float32, name="x_true")
y = tf.Variable(tf.zeros([None, operator.range.shape[0], operator.range.shape[1], 1]), dtype=tf.float32, name="y")
s = tf.Variable(tf.zeros([size, size, n_memory]), trainable=False)

def generate_data(validation=False):
    """Generate a set of random data."""
    n_iter = 1 if validation else n_data

    x_arr = np.zeros((n_iter, space.shape[0], space.shape[1], 1), dtype=np.float32)
    y_arr = np.zeros((n_iter, operator.range.shape[0], operator.range.shape[1], 1), dtype=np.float32)
    x_true_arr = np.zeros((n_iter, space.shape[0], space.shape[1], 1), dtype=np.float32)

    for i in range(n_iter):
        if validation:
            phantom = odl.phantom.shepp_logan(space, True)
        else:
            phantom = random_phantom(space)

        data = operator(phantom)
        noisy_data = data + odl.phantom.white_noise(operator.range) * np.mean(np.abs(data)) * 0.05
        fbp = pseudoinverse(noisy_data)

        x_arr[i, ..., 0] = fbp
        x_true_arr[i, ..., 0] = phantom
        y_arr[i, ..., 0] = noisy_data

    return x_arr, y_arr, x_true_arr

new_params = False

with tf.name_scope('variable_definitions'):
    if new_params:
        """
        Initialization with Keras Initializers: 
        The tf.keras.initializers.GlorotUniform() is used to replace tf.contrib.layers.xavier_initializer_conv2d, 
        providing a more straightforward way to initialize weights.
        """
        # Parameters if the network should be re-trained
        w1 = tf.Variable(tf.keras.initializers.GlorotUniform()(shape=[3, 3, n_memory + 2, 32]), name='w1')
        b1 = tf.Variable(tf.constant(0.01, shape=[1, 1, 1, 32]), name='b1')

        w2 = tf.Variable(tf.keras.initializers.GlorotUniform()(shape=[3, 3, 32, 32]), name='w2')
        b2 = tf.Variable(tf.constant(0.01, shape=[1, 1, 1, 32]), name='b2')

        w3 = tf.Variable(tf.keras.initializers.GlorotUniform()(shape=[3, 3, 32, n_memory + 1]), name='w3')
        b3 = tf.Variable(tf.constant(0.00, shape=[1, 1, 1, n_memory + 1]), name='b3')
    else:
        # If trained network is available, re-use as starting guess
        ld = np.load("code/partially_learned_gradient_descent_no_regularizer_parameters.npz")

        w1 = tf.Variable(tf.constant(ld['w1']), name='w1')
        b1 = tf.Variable(tf.constant(ld['b1']), name='b1')

        w2 = tf.Variable(tf.constant(ld['w2']), name='w2')
        b2 = tf.Variable(tf.constant(ld['b2']), name='b2')

        w3 = tf.Variable(tf.constant(ld['w3']), name='w3')
        b3 = tf.Variable(tf.constant(ld['b3']), name='b3')
    
@tf.function
def iterative_scheme(x_0, y, n_iter, s, w1, b1, w2, b2, w3, b3):
    # Implementation of the iterative scheme
    x_values = [x_0]
    x = x_0
    for i in range(n_iter):
        with tf.name_scope(f'iterate_{i}'):
            # Compute the adjoint gradient
            gradx = odl_op_layer_adjoint(odl_op_layer(x) - y)

            # Concatenate and process updates
            update = tf.concat([x, gradx, s], axis=3)

            # Apply convolutions and activations
            update = tf.nn.relu(conv2d(update, w1) + b1)
            update = tf.nn.relu(conv2d(update, w2) + b2)
            update = conv2d(update, w3) + b3

            # Split the update into s and dx
            s = tf.nn.relu(update[..., 1:])
            dx = update[..., 0:1]

            # Update x
            x = x + dx
            x_values.append(x)
    return x_values, x

@tf.function
def validate_step(x_arr_validate, y_arr_validate, x_true_arr_validate):
    # Get predictions using iterative_scheme
    x_pred = iterative_scheme(x_0, y_arr_validate, n_iter, s, w1, b1, w2, b2, w3, b3)[-1]
    # Compute loss
    loss_result = tf.reduce_mean(tf.reduce_sum((x_pred - x_true_arr_validate) ** 2, axis=(1, 2)))
    
    return x_pred, loss_result

# Learning rate
global_step = tf.Variable(0, trainable=False)
starter_learning_rate = 1e-3
learning_rate = tf.train.inverse_time_decay(starter_learning_rate,
                                            global_step=global_step,
                                            decay_rate=1.0,
                                            decay_steps=500,
                                            staircase=True,
                                            name='learning_rate')

# Create a learning rate schedule
learning_rate_schedule = tf.keras.optimizers.schedules.InverseTimeDecay(
    starter_learning_rate,
    decay_steps=500,
    decay_rate=1.0
)

# Define the optimizer using Keras
optimizer = tf.keras.optimizers.RMSprop(learning_rate=learning_rate_schedule)

# Solve with an ODL callback to see what happens in real time
callback = odl.solvers.CallbackShow(clim=[0.1, 0.4])

# Generate validation data
x_arr_validate, y_arr_validate, x_true_arr_validate = generate_data(validation=True)

new_train = False

if new_train:
    # Train the network
    n_train = 10
    for i in range(n_train):
        x_arr, y_arr, x_true_arr = generate_data()

        # Use GradientTape to record operations
        with tf.GradientTape() as tape:
            x_pred = iterative_scheme(x_0, y_arr, n_iter, s, w1, b1, w2, b2, w3, b3)[-1]  # Adjust as needed
            loss_training = tf.reduce_mean(tf.reduce_sum((x_pred - x_true_arr) ** 2, axis=(1, 2))) # Compute the loss

        gradients = tape.gradient(loss_training, [w1, w2, w3, b1, b2, b3])  # Compute gradients including any trainable variables
        optimizer.apply_gradients(zip(gradients, [w1, w2, w3, b1, b2, b3])) # Update weights

        # Validate on shepp-logan
        x_values_result, loss_result = validate_step(x_arr_validate, y_arr_validate, x_true_arr_validate)  # Validate

        print(
            f'iter={i}, training loss={loss_training.numpy()} validation loss={loss_result.numpy()}'
        )

        callback((space ** (n_iter + 1)).element([xv.squeeze() for xv in x_values_result]))
else:
    # Validate on shepp-logan
    x_values_result, loss_result = [x_values, loss], feed_dict={x_0: x_arr_validate, x_true: x_true_arr_validate, y: y_arr_validate}

    print(f'validation loss={loss_result}')

    callback((space ** (n_iter + 1)).element(
        [xv.squeeze() for xv in x_values_result]))
