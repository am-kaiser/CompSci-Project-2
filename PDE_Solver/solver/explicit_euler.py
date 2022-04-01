"""Script to perform explicit Euler algorithm."""
from solver import parameters
from utils import plot_solution

import numpy as np
import matplotlib.pyplot as plt
import time


def initialize_output(input_params):
    output = np.empty([input_params.num_x_steps, input_params.num_time_steps])
    output[:] = np.nan
    return output


def create_grid(input_params):
    x_values = np.linspace(input_params.x_start, input_params.x_end_cm, num=input_params.num_x_steps).astype(int)
    t_values = np.linspace(input_params.t_start, input_params.t_end_s, num=input_params.num_time_steps).astype(int)

    return x_values, t_values


def define_initial_conditions(u_values):
    u_values[0, :] = 0.0
    u_values[-1, :] = 0.0
    return u_values


def define_boundary_conditions(u_values, x_values):
    u_values[:, 0] = np.sin(np.pi * x_values)
    return u_values


def perform_euler_algorithm(u_values, x_values, t_values, input_params):
    alpha = input_params.dt / input_params.dx

    t_indices = range(input_params.num_time_steps)
    x_indices = range(input_params.num_x_steps)

    for t_index in t_indices[:-1]:
        for x_index in x_indices[1:-2]:
            u_values[x_index, t_index + 1] = alpha * u_values[x_index + 1, t_index] +\
                                         (1-2*alpha) * u_values[x_index, t_index] +\
                                         alpha * u_values[x_index - 1, t_index]

            plot_solution.plot_3d_solution(t_values, x_values, u_values, 't [s]', 'x [cm]', 'u', value_range=[-10,10])
            plt.show()
            time.sleep(0.1)
            plt.close('all')

    return u_values


def solve_pde_euler():
    # Load parameters
    params = parameters.Parameters()

    # Initialize output array
    u = initialize_output(params)

    # Create grid
    x, t = create_grid(params)

    # Apply initial conditions
    u = define_initial_conditions(u)

    # Apply boundary conditions
    u = define_boundary_conditions(u, x)

    # Perform Euler algorithm
    u = perform_euler_algorithm(u, x, t, params)

    # Plot solution
    plot_solution.plot_3d_solution(t, x, u, 't [s]', 'x [cm]', 'u', value_range=[0,1])
    plt.show()