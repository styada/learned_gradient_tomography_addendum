# Import necessary libraries
import tensorflow as tf
import numpy as np
import odl
from util import random_phantom, conv2d

# Using the eager execution mode is the default in TensorFlow 2.x.
# Removed the interactive session setup as it's not necessary.
# sess = tf.compat.v1.InteractiveSession()

# Create ODL data structures
size = 128

# creating a uniform discretization grid in a 2D space using the Operator Discretization Library
space = odl.uniform_discr([-64, -64], [64, 64], [size, size], dtype='float32')

# assigning the `parallel` attribute of the `odl.tomo.geometry` module to the variable `parallel`.
parallel = odl.tomo.geometry.parallel

# creating a geometry object for parallel beam projections in tomography with 30 angles
geometry = parallel.parallel_beam_geometry(space, num_angles=30)

# creating an instance of the `RayTransform`
operator = odl.tomo.RayTransform(space, geometry)

# creating a pseudoinverse operator using the Filtered Back Projection (FBP) method.
pseudoinverse = odl.tomo.fbp_op(operator)

# Ensure operator has fixed operator norm for scale invariance
# calculating the operator norm of the given operator using the power method.
opnorm = odl.power_method_opnorm(operator)

# scaling the operator by a factor of `1 / opnorm`.
operator = (1 / opnorm) * operator

# scaling the pseudoinverse operator by a factor of `opnorm`.
pseudoinverse = pseudoinverse * opnorm

# Create tensorflow layers from odl operators
# creating a TensorFlow layer from the ODL operator `operator` using the `as_tensorflow_layer` function
odl_op_layer = odl.contrib.tensorflow.as_tensorflow_layer(operator, 'RayTransform')

# creating a TensorFlow layer from the adjoint of the ODL operator
odl_op_layer_adjoint = odl.contrib.tensorflow.as_tensorflow_layer(operator.adjoint, 'RayTransformAdjoint')

# creating instances of the `PartialDerivative` class for each axis
partial0 = odl.PartialDerivative(space, axis=0)
partial1 = odl.PartialDerivative(space, axis=1)

# creating a TensorFlow layer from the regularization term defined by the sum of the squared partial derivatives along the two axes.
odl_op_regularizer = odl.contrib.tensorflow.as_tensorflow_layer(partial0.adjoint * partial0 + partial1.adjoint * partial1, 'Regularizer')

# User selected parameters
n_data = 20
n_memory = 5
n_iter = 10

def generate_data(validation=False):
    """
    The `generate_data` function creates a set of random data for training or validation purposes in a
    medical imaging context.
    
    :param validation: The `validation` parameter is a boolean flag that determines whether to generate
    data for validation purposes. If `validation` is set to `True`, the function will generate data for
    validation by using a Shepp-Logan phantom. Otherwise, if `validation` is `False`, random phantom
    data will, defaults to False (optional)
    :return: The function `generate_data` returns three NumPy arrays: `x_arr`, `y_arr`, and
    `x_true_arr`. These arrays contain generated random data for a given number of iterations.
    """
    n_iter = 1 if validation else n_data

    # initializing three NumPy arrays
    x_arr = np.empty((n_iter, space.shape[0], space.shape[1], 1), dtype='float32')
    y_arr = np.empty((n_iter, operator.range.shape[0], operator.range.shape[1], 1), dtype='float32')
    x_true_arr = np.empty((n_iter, space.shape[0], space.shape[1], 1), dtype='float32')

    for i in range(n_iter):
        if validation:
            # If validation is true create shepp_logan phantoms
            phantom = odl.phantom.shepp_logan(space, True)
        else:
            # If validation is false create random phantom with random ellipsoid phantoms
            phantom = random_phantom(space)
        
        # applying the operator to the phantom. this operation represents the forward projection of the phantom image.
        data = operator(phantom)
        
        # adding white noise to the data obtained from the forward projection operation.
        noisy_data = data + odl.phantom.white_noise(operator.range) * np.mean(np.abs(data)) * 0.05
        
        # performing a filtered back projection (FBP) operation on the noisy data obtained from the forward projection.
        fbp = pseudoinverse(noisy_data)

        x_arr[i, ..., 0] = fbp
        x_true_arr[i, ..., 0] = phantom
        y_arr[i, ..., 0] = noisy_data

    return x_arr, y_arr, x_true_arr

with tf.name_scope('placeholders'):
    # Define input shapes using tf.keras.Input
    # TensorFlow placeholders are now deprecated; the new version uses keras.Input
    x_0 = tf.keras.Input(shape=(size, size, 1), name="x_0")  # Initial input
    x_true = tf.keras.Input(shape=(size, size, 1), name="x_true")  # Ground truth
    y = tf.keras.Input(shape=(None, None, 1), name="y")  # Noisy measurement data (size can vary)

    # creating a TensorFlow tensor `s` filled with zeros of shape `[batch_size, size, size, n_memory]`.
    # Note: tf.zeros is preferred over tf.fill for clarity
    s = tf.zeros([tf.shape(x_0)[0], size, size, n_memory], dtype=tf.float32, name="s")

with tf.name_scope('variable_definitions'):
    if 0:
        # Parameters if the network should be re-trained
        # defining the weights and biases for a neural network model
        w1 = tf.keras.layers.Dense(32, kernel_initializer='he_normal')(tf.keras.Input(shape=(n_memory + 3,)))
        b1 = tf.Variable(tf.constant(0.01, shape=[1, 1, 1, 32]), name='b1')

        w2 = tf.keras.layers.Dense(32, kernel_initializer='he_normal')(tf.keras.Input(shape=(32,)))
        b2 = tf.Variable(tf.constant(0.01, shape=[1, 1, 1, 32]), name='b2')

        w3 = tf.keras.layers.Dense(n_memory + 1, kernel_initializer='he_normal')(tf.keras.Input(shape=(32,)))
        b3 = tf.Variable(tf.constant(0.00, shape=[1, 1, 1, n_memory + 1]), name='b3')
    else:
        # If trained network is available, re-use as starting guess
        # loading pre-trained parameters for a neural network model from a NumPy zip file
        ld = np.load('code/partially_learned_gradient_descent_parameters.npz')

        w1 = tf.Variable(tf.constant(ld['w1']), name='w1')
        b1 = tf.Variable(tf.constant(ld['b1']), name='b1')

        w2 = tf.Variable(tf.constant(ld['w2']), name='w2')
        b2 = tf.Variable(tf.constant(ld['b2']), name='b2')

        w3 = tf.Variable(tf.constant(ld['w3']), name='w3')
        b3 = tf.Variable(tf.constant(ld['b3']), name='b3')

# Define the ODL operations using tf.function
@tf.function
def odl_operations(x, y):
    return odl_op_layer_adjoint(odl_op_layer(x) - y)

# Implementation of the iterative scheme
# Wrapped in a function for better structure
def iterative_scheme(x_0, y, s, n_iter):
    """Run the iterative gradient descent scheme."""
    x = x_0
    x_values = [x]

    for i in range(n_iter):
        with tf.name_scope('iterate_{}'.format(i)):
            gradx = odl_operations(x, y)
            gradreg = odl_op_regularizer(x)

            update = tf.concat([x, gradx, gradreg, s], axis=3)

            update = tf.nn.relu(conv2d(update, w1) + b1)
            update = tf.nn.relu(conv2d(update, w2) + b2)

            update = conv2d(update, w3) + b3

            s = tf.nn.relu(update[..., 1:])
            dx = update[..., 0:1]

            x = x + dx
            x_values.append(x)

    return x_values

# Initialize loss and optimizer
with tf.name_scope('loss'):
    loss = tf.reduce_mean(tf.reduce_sum((x - x_true) ** 2, axis=(1, 2)))

with tf.name_scope('optimizer'):
    # Learning rate
    global_step = tf.Variable(0, trainable=False)
    starter_learning_rate = 1e-3
    learning_rate = tf.train.inverse_time_decay(
        starter_learning_rate,
        global_step=global_step,
        decay_rate=1.0,
        decay_steps=500,
        staircase=True,
        name='learning_rate'
    )

    optimizer = tf.train.RMSPropOptimizer(learning_rate).minimize(loss, global_step=global_step)

# Initialize all TF variables
tf.global_variables_initializer().run()

# Create a TensorFlow session
with tf.Session() as sess:
    for i in range(n_iter):
        # Generate training data
        x_arr, y_arr, x_true_arr = generate_data()

        # Run the optimization and get the final x value
        x_history = iterative_scheme(x_0, y, s, n_iter)
        x_final = x_history[-1]  # Get the last updated x

        # Run the optimization
        _, loss_val = sess.run([optimizer, loss], feed_dict={
            x_0: x_arr,
            x_true: x_true_arr,
            y: y_arr,
            x: x_final
        })
        
        print(f'Iteration {i}: Loss = {loss_val}')
