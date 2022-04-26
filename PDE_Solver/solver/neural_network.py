"""Script to solve PDE with a Neural Network algorithm."""

from PDE_Solver.solver import explicit_euler, parameters

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt


def make_kernel(a):
    """Transform a 2D array into a convolution kernel"""
    a = np.asarray(a)
    a = a.reshape(list(a.shape) + [1, 1])
    return tf.constant(a, dtype=1)


def simple_conv(x, k):
    """A simplified 2D convolution operation"""
    x = tf.expand_dims(tf.expand_dims(x, 0), -1)
    y = tf.nn.depthwise_conv2d(x, k, [1, 1, 1, 1], padding='SAME')
    return y[0, :, :, 0]


def laplace(x):
    """Compute the 2D laplacian of an array"""
    laplace_k = make_kernel([[0.5, 1.0, 0.5],
                             [1.0, -6., 1.0],
                             [0.5, 1.0, 0.5]])
    return simple_conv(x, laplace_k)


def set_initial_boundary_conditions(input_params):
    # Create grid
    x, t = explicit_euler.create_grid(input_params)
    # Initialize output array
    u = explicit_euler.initialize_output(input_params)
    # Apply initial conditions
    u = explicit_euler.define_initial_conditions(u)
    # Apply boundary conditions
    u = explicit_euler.define_boundary_conditions(u, x)

    return x, t, u


if __name__ == "__main__":
    # Load parameters
    params = parameters.Parameters()

    # sess = tf.InteractiveSession()

    u_init = set_initial_boundary_conditions(params)

    U = tf.Variable(u_init)

    plt.imshow(U.eval())
    plt.show()
